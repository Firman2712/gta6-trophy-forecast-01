#!/usr/bin/env python3
"""Sparse-refresh validation of the growing-history path (round 4).

The real final case differs from tools/synth_test.py in two ways that
matter:

1. Exophase refreshes are SPARSE and irregular.  The live page was
   unchanged 19 Nov 13:32 -> 20 Nov 03:32 (~14h, all 55 rates identical),
   after a ~4.7h gap to the first refresh (04:30) and <=9h to the second
   (<=13:32).  The final history will therefore contain only a handful
   of DISTINCT (N, rates) states, not the dense hourly refreshes the old
   synthetic test assumed.  With ~3-5 distinct points the N(t) and E_i(t)
   fits are fragile, and the old test does not exercise that regime.

2. The ANSWER is the first valid read in the hour from the target, i.e.
   the rates as of the LAST REFRESH before the target -- not the rates
   at the target instant.  With lengthening refresh gaps that lag can be
   12-30h, so the model is scored here against the truth as of the last
   refresh before the target (and, for reference, against truth at the
   target).

Truth families are calibrated to the two refresh-revealed states we
actually know: N ~= 3,868 at 04:30 UTC 19 Nov and N ~= 15,000 at
13:32 UTC 19 Nov (from the platinum 0.03% rate).  The launch-night
638 is a STALE page value (the page had not refreshed since 23:48),
so the truth's N(3h) is not forced to 638 -- only the DISPLAYED reads
show 638 until the 04:30 refresh.  Per-trophy earner growth exponents
alpha_i are calibrated from the real launch-night earners and the real
13:32 live page.

Model variants are loaded by textual patching of main.py (the entry
file itself is untouched by this test).
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

WS = "/home/hatch/workspace/makefun"
KIT = os.path.join(WS, "kits/gta6-trophy-rarity-forecast-01-kit/input")
REPO = os.path.join(WS, "gta6-trophy-forecast-01")
LIVE1332 = json.load(open(os.path.join(WS, "gta6-live-20261119-1332.json")))["rates"]
base = json.load(open(os.path.join(KIT, "p01.in")))
TROPHIES = base["trophies"]
LAUNCH_RATES = base["history"][-1]["rates"]  # launch-night rates, N=638 page

LAUNCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc).timestamp()
T_START = datetime(2026, 11, 18, 23, 50, tzinfo=timezone.utc).timestamp()
T_DEADLINE = datetime(2026, 11, 21, 11, 30, tzinfo=timezone.utc).timestamp()
T_TARGET = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc).timestamp()

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)


def iso(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def logit(p):
    p = min(max(p, 0.0001), 0.9999)
    return math.log(p / (1.0 - p))


def pinball(y_pct, qs):
    y = logit(y_pct / 100.0)
    tot = 0.0
    for tau, q in zip(LEVELS, qs):
        d = y - logit(q / 100.0)
        tot += 2.0 * max(tau * d, (tau - 1.0) * d)
    return tot / len(LEVELS)


def load_variant(patches):
    """Load main.py with textual replacements applied."""
    src = open(os.path.join(REPO, "main.py")).read()
    for old, new in patches:
        assert old in src, f"patch anchor not found: {old[:70]}"
        src = src.replace(old, new)
    mod = {}
    exec(compile(src, "main_variant.py", "exec"), mod)
    return mod


# --- calibrated per-trophy earner growth ---------------------------------
# alpha_i from real observations: E(3h) = launch rate * 638,
# E(13.5h) = live rate * 15000.  Clipped to [0, 3].
ALPHA = []
for i in range(55):
    if i == 1:  # the ~100% trophy
        ALPHA.append(0.0)
        continue
    e3 = max(LAUNCH_RATES[i] / 100.0 * 638.0, 0.5)
    e135 = max(LIVE1332[i] / 100.0 * 15000.0, 0.5)
    a = math.log(e135 / e3) / math.log(13.5 / 3.0) if e135 > e3 else 0.0
    ALPHA.append(min(max(a, 0.0), 3.0))
K_EARN = [max(LAUNCH_RATES[i] / 100.0 * 638.0, 0.5) for i in range(55)]


def make_truth(n_kind, e_kind):
    """Return (n_true, e_true, rate_true) closures for a regime."""
    if n_kind == "pl":  # power law through (4.5h, 3868), (13.5h, 15000)
        s = math.log(15000.0 / 3868.0) / math.log(13.5 / 4.5)

        def n_true(t):
            h = max(0.1, (t - LAUNCH) / 3600.0)
            return 3868.0 * (h / 4.5) ** s
    else:  # saturating N = A - B*exp(-h/tau), solved from the two points
        tau = {"sat24": 24.0, "sat48": 48.0, "sat96": 96.0, "sat200": 200.0}[n_kind]
        u1, u2 = math.exp(-4.5 / tau), math.exp(-13.5 / tau)
        B = (15000.0 - 3868.0) / (u1 - u2)
        A = 3868.0 + B * u1

        def n_true(t):
            h = max(0.1, (t - LAUNCH) / 3600.0)
            return A - B * math.exp(-h / tau)

    def e_true(i, t):
        if i == 1:
            return n_true(t)
        h = max(0.1, (t - LAUNCH) / 3600.0)
        if e_kind == "pl":
            return K_EARN[i] * (h / 3.0) ** ALPHA[i]
        # decelerating: slope halves after h=48 (continuous in log-log)
        a = ALPHA[i]
        if h <= 48.0:
            return K_EARN[i] * (h / 3.0) ** a
        return K_EARN[i] * (48.0 / 3.0) ** a * (h / 48.0) ** (a * 0.5)

    def rate_true(i, t):
        n = n_true(t)
        return 100.0 * min(e_true(i, t), n) / n

    return n_true, e_true, rate_true


# Refresh schedule (hours after launch): lengthening gaps, as observed.
REFRESH_H = [4.5, 12.5, 27.0, 48.0, 74.0, 104.0, 138.0]


def build_case(n_true, e_true, rate_true, target_t):
    """Final-like case: reads every 30 min (10 min last 6h), values frozen
    between refreshes; the launch-night page shows the stale 638 state
    until the 04:30 refresh."""
    refresh_t = [LAUNCH + h * 3600.0 for h in REFRESH_H]

    def displayed(t):
        # last refresh at or before t; before the first, the stale page
        ref = None
        for rt in refresh_t:
            if rt <= t:
                ref = rt
            else:
                break
        if ref is None:
            n = 638.0
            rates = list(LAUNCH_RATES)
        else:
            n = n_true(ref)
            rates = [round(rate_true(i, ref), 2) for i in range(55)]
        return n, rates

    reads = []
    t = T_START
    slot = 0
    while t <= T_DEADLINE:
        n, rates = displayed(t)
        tot = sum(r / 100.0 * n for r in rates)
        reads.append({"slot": slot, "at": iso(t), "players_tracked": int(n),
                      "trophies_earned": int(tot), "rates": rates})
        slot += 1
        t += 600 if t > T_DEADLINE - 6 * 3600 else 1800
    case = dict(base)
    case["id"] = "f00-sparse"
    case["at"] = iso(T_DEADLINE)
    case["target"] = iso(target_t)
    case["horizon_minutes"] = int((target_t - T_DEADLINE) / 60)
    case["history"] = reads
    return case


def answer_time():
    """First refresh at or before the target: the answer's effective time."""
    ref = None
    for h in REFRESH_H:
        rt = LAUNCH + h * 3600.0
        if rt <= T_TARGET:
            ref = rt
    return ref


def score(rows, rate_fn):
    return sum(pinball(rate_fn(r["id"] - 1), r["q"]) for r in rows) / len(rows)


def persistence_loss(case, rate_fn):
    rates = case["history"][-1]["rates"]
    sig = 0.5 * math.sqrt((T_TARGET - T_DEADLINE) / 86400.0)
    tot = 0.0
    for i in range(55):
        lm = logit(rates[i] / 100.0)
        qs = [round(min(max(100.0 / (1.0 + math.exp(-(lm + zz * sig))), 0.01), 99.99), 2)
              for zz in Z]
        tot += pinball(rate_fn(i), qs)
    return tot / 55


VARIANTS = {
    # main.py is now the V17 configuration (sat-only N, shrink cap 0.95,
    # growing sigma x0.8).  Variants below revert one choice at a time.
    "V0 current": [],
    "V2 pl-only": [("if n_sat is not None and n_sat > 0:\n            n_tgt = n_sat\n        elif n_pl is not None:\n            n_tgt = n_pl",
                    "if n_pl is not None:\n            n_tgt = n_pl\n        elif n_sat is not None and n_sat > 0:\n            n_tgt = n_sat")],
    "V3 lag-12h": "TARGET_SHIFT_12H",
    "V4 lag-24h": "TARGET_SHIFT_24H",
    "V6 sigma x1.0": [("sigma_base = 0.8 * math.sqrt(0.09 + 0.08 * horizon_days)",
                       "sigma_base = math.sqrt(0.09 + 0.08 * horizon_days)")],
    "V8 cap 0.85": [("w = clamp(math.log10(le + 1.0) / 2.0, 0.15, 0.95)",
                     "w = clamp(math.log10(le + 1.0) / 2.0, 0.15, 0.85)")],
    "V10 sigma x0.7": [("sigma_base = 0.8 * math.sqrt(0.09 + 0.08 * horizon_days)",
                        "sigma_base = 0.7 * math.sqrt(0.09 + 0.08 * horizon_days)")],
    "V11 sigma x0.6": [("sigma_base = 0.8 * math.sqrt(0.09 + 0.08 * horizon_days)",
                        "sigma_base = 0.6 * math.sqrt(0.09 + 0.08 * horizon_days)")],
    "V18 lag12h+sig.7": "TARGET_SHIFT_12H_SIG7",
}

REGIMES = [("sat24", "pl"), ("sat48", "pl"), ("sat96", "pl"),
           ("sat200", "pl"), ("pl", "pl"),
           ("sat48", "pld"), ("sat96", "pld"), ("pl", "pld")]


def main():
    t_ans = answer_time()
    print(f"answer effective time (last refresh before target): {iso(t_ans)} "
          f"({(T_TARGET - t_ans)/3600:.0f}h before target)")
    print(f"alpha_i range: {min(ALPHA):.2f}..{max(ALPHA):.2f} "
          f"(median {sorted(ALPHA)[27]:.2f}); fishing id20 alpha={ALPHA[19]:.2f}")
    for nk, ek in REGIMES:
        n_true, e_true, rate_true = make_truth(nk, ek)
        case = build_case(n_true, e_true, rate_true, T_TARGET)
        n_dl = n_true(T_DEADLINE)
        n_tg = n_true(T_TARGET)
        n_an = n_true(t_ans)
        distinct = sorted({r["players_tracked"] for r in case["history"]})
        print(f"\n== regime N={nk} E={ek}: N(deadline)={n_dl:,.0f} "
              f"N(answer)={n_an:,.0f} N(target)={n_tg:,.0f} "
              f"distinct N in history={distinct}")
        print(f"   persistence loss vs answer-truth: "
              f"{persistence_loss(case, lambda i: rate_true(i, t_ans)):.4f}")
        ans_fn = lambda i: rate_true(i, t_ans)  # noqa: E731
        tgt_fn = lambda i: rate_true(i, T_TARGET)  # noqa: E731
        for name, patches in VARIANTS.items():
            if isinstance(patches, str):
                shift = 12 * 3600 if "12H" in patches else 24 * 3600
                c2 = build_case(n_true, e_true, rate_true, T_TARGET - shift)
                extra = []
                if "SIG7" in patches:
                    extra.append(("sigma_base = 0.8 * math.sqrt(0.09 + 0.08 * horizon_days)",
                                  "sigma_base = 0.7 * math.sqrt(0.09 + 0.08 * horizon_days)"))
                mod = load_variant(extra)
            else:
                c2 = case
                mod = load_variant(patches)
            rows = mod["forecast_case"](c2)
            # what N did the model predict? recover from median of trophy 1
            # (slow trophy, E ~ frozen): N_pred ~ E1/median1 -- instead
            # report the model's implied N via the flat/growing branch is
            # internal; approximate from the platinum median vs its E.
            la = sum(pinball(rate_true(r["id"] - 1, t_ans), r["q"])
                     for r in rows) / len(rows)
            lt = sum(pinball(rate_true(r["id"] - 1, T_TARGET), r["q"])
                     for r in rows) / len(rows)
            print(f"   {name:14s} loss vs answer {la:.4f} | vs target {lt:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
