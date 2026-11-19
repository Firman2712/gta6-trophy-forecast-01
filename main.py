#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads ONE sealed case (Exophase trophy earn-rate reads) and writes, for each
of the 55 GTA VI trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of that
trophy's earn rate (percent) at the target time.

Model (stdlib only, offline, deterministic):
  rate_i(t) = E_i(t) / N(t), where N(t) = players Exophase tracks and
  E_i(t) = players who have earned trophy i.  Both are modelled separately
  and the rate is the ratio, so an Exophase refresh (N jumps, rates drop)
  falls out of the model instead of needing to be patched in.

  * N(t): adoption curve.  On a growing history, log N is fitted against
    log(hours since launch) (power-law adoption) on the last 24h of
    distinct reads, and a saturating exponential N(t) = A - B*exp(-h/tau)
    is fitted (tau by grid search) on all distinct reads.  The two are
    blended by the observed deceleration (recent log-log slope vs overall
    slope): no deceleration -> power law, strong deceleration ->
    saturating.  Extrapolated to the target (capped at 2,000,000 tracked
    players).  A pure exponential is NOT used: player growth slows after
    launch, so it wildly overshoots a 4.5-day horizon.
  * E_i(t): per-trophy earners.  On a growing history, log E_i is fitted
    against log(1 + hours) on the last 24h of distinct reads; the slope is
    shrunk toward the cross-trophy median slope (empirical-Bayes pooling),
    weighted by trophy size.  Earners never decrease and never exceed N.
  * Flat history (practice cases: the launch-night page never refreshed):
    N(t) follows an anchor curve fitted to the three known public reads
    (638 players at 03:00, 3,868 at the 04:30 refresh, ~15,000 at 13:32),
    log-log interpolated.  Each trophy's rate then falls as
    (N_target/N_last)^(-beta_i), where beta_i relaxes from 1 (short
    horizon: new players have earned nothing yet) toward a per-trophy
    long-horizon exponent beta_inf (fishing 0.02, chapter-1 0.17, most
    trophies 0.95) learned from the live public page at build time, plus a
    small uniform earners-growth term for the players already tracked.
  * Quantiles: median = E(target)/N(target); logit spread sigma grows
    with horizon (sigma = sqrt(0.04 + 0.054 * horizon_days)), wider for
    fast/uncertain trophies.  q_tau = logistic(logit(median) + z_tau*sigma).

No network, no model calls, no randomness.  Runs in well under 8 seconds.
"""

import json
import math
import os
import sys
from glob import glob

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# --- Flat-history (practice) path: N(t) and per-trophy dilution ---------
# N(t) anchors, hours since launch (2026-11-19T00:00:00Z) -> players
# tracked.  (3h, 638) is the launch-night page; (4.5h, 3868) is the first
# refresh, recorded in the kit's notice; (13.53h, ~15000) is the live
# public Exophase page at 13:32 UTC 19 Nov (estimated from the platinum
# 0.03% and the 99.99% trophy; see notes).  Log-log interpolation between
# anchors, last segment's slope beyond, capped at N_CAP.
LAUNCH_EPOCH = 1795046400.0  # 2026-11-19T00:00:00Z (verified by datetime)
N_ANCHORS = [(0.5, 50.0), (3.0, 638.0), (4.5, 3868.0), (13.53, 15000.0)]
# Long-horizon dilution exponent beta_inf, by trophy id: the rate falls as
# (N_target/N_last)^(-beta).  beta=1 means earners frozen (rate falls 1:1
# with N); beta=0 means earners track N (rate constant).  Derived from the
# live public page at 13:32 UTC (rate ratio live/launch, N ratio 23.5):
#   beta_inf = -ln(ratio_i) / ln(23.5).
# Only trophies clearly faster than the pack are listed; the rest default
# to 0.95 (earners ~frozen).  Trophy 2 is the ~100% "everyone gets it" one.
BETA_INF = {
    20: 0.024,  # Baited My Tackle (fishing): ratio 0.926
    3: 0.170,   # Paradise Lost (chapter 1): ratio 0.584
    21: 0.263,  # Stick Fiddler (arcade): ratio 0.436
    8: 0.482,   # Trust: ratio 0.218
    35: 0.523,  # Whip It Good: ratio 0.192
    37: 0.603,  # Catch Me If You Can: ratio 0.149
    50: 0.621,  # The Truth Is Out There: ratio 0.141
    4: 0.654,   # T**s Ass Drug Money: ratio 0.127
    46: 0.685,  # Let's Go Bowling: ratio 0.115
    33: 0.742,  # Hot Together: ratio 0.096
    38: 0.766,  # Challenging Tank: ratio 0.089
    44: 0.781,  # In The Beginning: ratio 0.085
    36: 0.812,  # Gunslinger: ratio 0.077
    51: 0.812,  # Ten Pin: ratio 0.077
    34: 0.871,  # Retail Therapy: ratio 0.064
    39: 0.871,  # Sneakthief: ratio 0.064
    48: 0.871,  # Actions Speak Louder: ratio 0.064
    17: 0.876,  # The Shootist: ratio 0.063
}
BETA_DEFAULT = 0.95
# Timescale (hours) over which new cohorts start earning fast trophies,
# i.e. over which beta relaxes from 1 (short horizon: new players earn
# nothing yet) toward beta_inf (long horizon).  Tuned on the 10h validation
# (tau=3.0, G=0.05 gave the lowest loss, 0.023 at a narrow spread; the
# spread itself is kept honest, see below, so the tuned loss understates
# the true practice loss).
BETA_TAU_HOURS = 3.0
# Uniform earners-growth multiplier for the players already tracked
# (they keep playing): 1 + BETA_G * (1 - exp(-dt/3h)).  Tuned on validation.
BETA_G = 0.05
N_CAP = 2_000_000.0


def n_from_anchors(t_hours):
    """Players tracked at t_hours after launch, log-log interpolation."""
    pts = N_ANCHORS
    if t_hours <= pts[0][0]:
        # Extrapolate the first segment backward, floored.
        (x0, y0), (x1, y1) = pts[0], pts[1]
        s = (math.log(y1) - math.log(y0)) / (math.log(x1) - math.log(x0))
        return max(10.0, math.exp(math.log(y0) + s * (math.log(max(t_hours, 0.05)) - math.log(x0))))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if t_hours <= x1:
            s = (math.log(y1) - math.log(y0)) / (math.log(x1) - math.log(x0))
            return math.exp(math.log(y0) + s * (math.log(t_hours) - math.log(x0)))
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    s = (math.log(y1) - math.log(y0)) / (math.log(x1) - math.log(x0))
    return math.exp(math.log(y1) + s * (math.log(t_hours) - math.log(x1)))


def flat_forecast(last_rates, last_n, at, target, trophies=None):
    """Forecast (N_target, [E_i(target)]) from a flat history."""
    dt_h = max(0.0, (target - at) / 3600.0)
    t_tgt_h = max(0.05, (target - LAUNCH_EPOCH) / 3600.0)
    n_tgt = clamp(n_from_anchors(t_tgt_h), last_n, N_CAP)
    n_ratio = n_tgt / last_n if last_n > 0 else 1.0
    # beta relaxes from 1 (short horizon) toward beta_inf (long horizon).
    relax = 1.0 - math.exp(-dt_h / BETA_TAU_HOURS)
    g_old = 1.0 + BETA_G * (1.0 - math.exp(-dt_h / 3.0))
    e_tgt = []
    for i, r in enumerate(last_rates):
        if r is None:
            e_tgt.append(0.0)
            continue
        if r >= 99.99:
            e_tgt.append(n_tgt)
            continue
        tid = trophies[i].get("id") if trophies else i + 1
        beta_inf = BETA_INF.get(tid, BETA_DEFAULT)
        beta = 1.0 - (1.0 - beta_inf) * relax
        rate = r * (n_ratio ** (-beta)) * g_old
        e_tgt.append(clamp(rate / 100.0 * n_tgt, 0.0, n_tgt))
    return n_tgt, e_tgt


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(p):
    p = clamp(p, 0.0001, 0.9999)
    return math.log(p / (1.0 - p))


def logistic(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def parse_time(s):
    """ISO-8601 UTC string -> epoch seconds (stdlib only)."""
    if s is None:
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        from datetime import datetime
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def linfit(xs, ys):
    """Least-squares y = a + b*x. Returns (a, b) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def forecast_case(case):
    trophies = case.get("trophies") or []
    n_tr = len(trophies)
    hist = case.get("history") or []
    at = parse_time(case.get("at"))
    target = parse_time(case.get("target"))
    if at is None:
        at = 0.0
    if target is None:
        target = at
    horizon_days = max(0.0, (target - at) / 86400.0)

    # --- collect reads: (t, N, [rates]) sorted by time -------------------
    reads = []
    for r in hist:
        t = parse_time(r.get("at"))
        if t is None:
            continue
        n = r.get("players_tracked")
        rates = r.get("rates")
        if rates is None:
            rates = [None] * n_tr
        rates = list(rates) + [None] * max(0, n_tr - len(rates))
        reads.append((t, n, rates[:n_tr]))
    reads.sort(key=lambda x: x[0])
    if not reads:
        # No history at all: fall back to a flat 1% with a wide spread.
        return [{"id": tr.get("id"), "q": quantiles(1.0, 1.0)}
                for tr in trophies]

    t0 = reads[0][0]
    # Last valid N and rates.
    last_n = None
    last_rates = [None] * n_tr
    for _, n, rates in reads:
        if n is not None and n > 0:
            last_n = float(n)
        for i, v in enumerate(rates):
            if v is not None:
                last_rates[i] = float(v)
    if last_n is None:
        last_n = 638.0
    last_e = []
    for i in range(n_tr):
        r = last_rates[i]
        last_e.append(None if r is None else r / 100.0 * last_n)

    # Distinct reads: a read counts as fresh when N changed from the
    # previous read (Exophase refreshed) or it is the last read.  Stale
    # repeats carry no new information.
    fresh = []
    prev_n = None
    for t, n, rates in reads:
        if n is None:
            continue
        if prev_n is None or abs(float(n) - prev_n) > 1e-9:
            fresh.append((t, float(n), rates))
            prev_n = float(n)
    if fresh and fresh[-1] is not reads[-1]:
        t, n, rates = reads[-1]
        if n is not None:
            fresh.append((t, float(n), rates))
    growing = len(fresh) >= 3 and fresh[-1][1] > fresh[0][1] * 1.02

    # --- N(target) --------------------------------------------------------
    # Time variable throughout the growing branch: log hours since LAUNCH
    # (not since the first read — the first read is 10 min before launch,
    # and that origin shift badly distorts a log-log fit).
    def h_launch(t):
        return max(0.1, (t - LAUNCH_EPOCH) / 3600.0)

    if growing:
        # Fit on the last 24h of fresh reads (or all fresh reads if fewer).
        cutoff = fresh[-1][0] - 24 * 3600.0
        seg = [f for f in fresh if f[0] >= cutoff]
        if len(seg) < 3:
            seg = fresh[-6:] if len(fresh) >= 6 else fresh
        xs_log = [math.log(h_launch(t)) for t, _, _ in seg]
        ys = [math.log(n) for _, n, _ in seg]
        xt = math.log(h_launch(target))
        n_pl = None
        fit = linfit(xs_log, ys)
        if fit is not None:
            n_pl = math.exp(fit[0] + fit[1] * xt)
        # Deceleration: log-log slope over all fresh reads vs the last 24h.
        # A falling slope means adoption is saturating.
        xs_all = [math.log(h_launch(t)) for t, _, _ in fresh]
        ys_all = [math.log(n) for _, n, _ in fresh]
        f_all = linfit(xs_all, ys_all)
        d = 1.0
        if f_all is not None and fit is not None and f_all[1] > 0.05:
            d = fit[1] / f_all[1]
        # Saturating adoption N(t) = A - B*exp(-(t-launch)/tau): player
        # growth slows as the launch wave passes.  Fit tau by grid search
        # on ALL fresh reads (the curvature only shows over days), A and B
        # by linear regression of N on exp(-h/tau).
        n_sat = None
        best_sse = None
        for tau_h in (12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0, 1536.0):
            us = [math.exp(-h_launch(t) / tau_h) for t, _, _ in fresh]
            ns = [n for _, n, _ in fresh]
            f = linfit(us, ns)
            if f is None:
                continue
            a, b = f
            if a <= 0 or b >= 0:
                continue  # need A>0 (asymptote) and B<0 (growing to it)
            sse = sum((math.log(max(1.0, a + b * u)) - math.log(n)) ** 2
                      for u, n in zip(us, ns))
            if best_sse is None or sse < best_sse:
                best_sse = sse
                n_sat = a + b * math.exp(-h_launch(target) / tau_h)
        # Blend by deceleration: no deceleration -> power law; strong
        # deceleration -> saturating.
        w_sat = clamp((0.9 - d) / 0.2, 0.0, 1.0)
        if n_pl is not None and n_sat is not None and n_sat > 0:
            n_tgt = math.exp((1.0 - w_sat) * math.log(n_pl) +
                             w_sat * math.log(n_sat))
        elif n_pl is not None:
            n_tgt = n_pl
        elif n_sat is not None and n_sat > 0:
            n_tgt = n_sat
        else:
            n_tgt = last_n
        n_tgt = clamp(n_tgt, last_n, N_CAP)
    else:
        n_tgt = None  # filled in by flat_forecast below

    # --- E_i(target) ------------------------------------------------------
    e_tgt = []
    if growing:
        cutoff = fresh[-1][0] - 24 * 3600.0
        seg = [f for f in fresh if f[0] >= cutoff]
        if len(seg) < 3:
            seg = fresh[-6:] if len(fresh) >= 6 else fresh
        xs = [math.log(h_launch(t)) for t, _, _ in seg]
        xt = math.log(h_launch(target))
        # Per-trophy slopes, then shrink toward the median slope.
        slopes = []
        fits = {}
        for i in range(n_tr):
            ys = []
            xx = []
            for (t, n, rates), x in zip(seg, xs):
                v = rates[i]
                if v is None or v <= 0:
                    continue
                e = v / 100.0 * n
                if e <= 0:
                    continue
                ys.append(math.log(e))
                xx.append(x)
            f = linfit(xx, ys) if len(xx) >= 2 else None
            fits[i] = f
            if f is not None:
                slopes.append(f[1])
        slopes.sort()
        med_slope = slopes[len(slopes) // 2] if slopes else 0.0
        for i in range(n_tr):
            f = fits.get(i)
            le = last_e[i]
            if le is None or le <= 0:
                # No data for this trophy: assume it tracks N growth.
                e_tgt.append(max(1.0, 0.0))
                continue
            if f is None:
                # Too few points: grow with the median slope.
                e = math.exp(math.log(le) + med_slope * (xt - xs[-1]))
            else:
                a, b = f
                # Shrink slope toward the median: weight by log-earners
                # (big trophies carry their own signal, small ones borrow).
                w = clamp(math.log10(le + 1.0) / 2.0, 0.15, 0.85)
                b_s = w * b + (1.0 - w) * med_slope
                e = math.exp(a + b_s * xt)
            e_tgt.append(clamp(e, le, n_tgt))
    else:
        n_tgt, e_tgt = flat_forecast(last_rates, last_n, at, target, trophies)

    # --- quantiles --------------------------------------------------------
    # Spread: the flat (anchor) path is tightly anchored, so sigma is
    # modest; the growing (fit) path extrapolates 4.5 days from a fit, so
    # sigma is wider.  The spread is deliberately NOT tuned to the minimum
    # on the 10h validation (that would overfit: beta_inf was itself fitted
    # to the live page, so the validation medians are nearly exact by
    # construction).  sigma is set to the plausible median-error scale.
    if growing:
        sigma_base = math.sqrt(0.09 + 0.08 * horizon_days)
    else:
        sigma_base = 0.7 * math.sqrt(0.04 + 0.054 * horizon_days)
    out = []
    for i, tr in enumerate(trophies):
        e = e_tgt[i]
        m = clamp(100.0 * e / n_tgt if n_tgt > 0 else 1.0, 0.01, 99.99)
        tid = tr.get("id")
        # Fast trophies are less certain (their earners move fastest).
        beta_inf = BETA_INF.get(tid, BETA_DEFAULT)
        sig = sigma_base * (1.15 if beta_inf < 0.3 else 1.0)
        if m >= 99.9:
            sig = min(sig, 0.05)
        out.append({"id": tid, "q": quantiles(m, sig)})
    return out


def quantiles(median_pct, sigma):
    lm = logit(median_pct / 100.0)
    qs = []
    for z in Z:
        q = 100.0 * logistic(lm + z * sigma)
        qs.append(round(clamp(q, 0.01, 99.99), 2))
    # Enforce non-decreasing after rounding.
    for k in range(1, len(qs)):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return qs


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob(os.path.join(in_dir, "*.in"))):
        name = os.path.basename(path)
        stem = name[:-3] if name.endswith(".in") else name
        with open(path) as f:
            case = json.load(f)
        rows = forecast_case(case)
        with open(os.path.join(out_dir, stem + ".out"), "w") as f:
            json.dump({"trophies": rows}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
