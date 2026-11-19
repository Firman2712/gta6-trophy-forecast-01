#!/usr/bin/env python3
"""Grid-search the flat-path parameters on the 10h validation.

Tunes BETA_TAU_HOURS, BETA_G and the sigma scale by scoring the pseudo-case
(p01 history -> live page at 13:32 UTC).  Also reports the implied 90-min
practice factor (median rate ratio at the real p01 target 05:00) so the
tuned values can be sanity-checked against the house baseline's ~1.51
practice loss (which implies an overall 90-min factor of ~0.20).
"""
import itertools
import json
import math
import os
import sys

WS = "/home/hatch/workspace/makefun"
sys.path.insert(0, os.path.join(WS, "gta6-trophy-forecast-01"))
import main as M

KIT = os.path.join(WS, "kits/gta6-trophy-rarity-forecast-01-kit/input")
LIVE_RATES = json.load(open(os.path.join(WS, "gta6-live-20261119-1332.json")))["rates"]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def logit(p):
    p = min(max(p, 0.0001), 0.9999)
    return math.log(p / (1.0 - p))


def pinball(y_pct, qs):
    y = logit(y_pct / 100.0)
    return sum(2.0 * max(t * (y - logit(q / 100.0)), (t - 1.0) * (y - logit(q / 100.0)))
               for t, q in zip(LEVELS, qs)) / len(LEVELS)


def score(rows):
    return sum(pinball(y, r["q"]) for r, y in zip(rows, LIVE_RATES)) / len(LIVE_RATES)


def build_case(target, horizon_min):
    case = json.load(open(os.path.join(KIT, "p01.in")))
    case["at"] = case["history"][-1]["at"]
    case["target"] = target
    case["horizon_minutes"] = horizon_min
    return case


# Pseudo-case: 10h horizon, ground truth = live page.
val_case = build_case("2026-11-19T13:32:00Z", 631)
# Real practice case: 90-min horizon, for the implied-factor check.
prac_case = json.load(open(os.path.join(KIT, "p01.in")))

results = []
for tau, g, sig_scale in itertools.product(
        [3.0, 4.0, 6.0, 8.0, 12.0, 24.0],
        [0.0, 0.1, 0.15, 0.2, 0.3],
        [0.7, 1.0, 1.3]):
    M.BETA_TAU_HOURS = tau
    M.BETA_G = g
    # Patch the sigma by scaling inside forecast via a wrapper: easiest is
    # to monkey-patch quantiles to scale sigma.
    orig_q = M.quantiles

    def scaled_q(med, sig, _o=orig_q, _s=sig_scale):
        return _o(med, sig * _s)
    M.quantiles = scaled_q
    rows = M.forecast_case(val_case)
    loss = score(rows)
    # Implied 90-min factor on the real practice case.
    prow = M.forecast_case(prac_case)
    last = prac_case["history"][-1]["rates"]
    fac = []
    for r, lr in zip(prow, last):
        if lr and lr < 99.99:
            fac.append(r["q"][2] / lr)
    fac.sort()
    med_fac = fac[len(fac) // 2]
    M.quantiles = orig_q
    results.append((loss, tau, g, sig_scale, med_fac))

results.sort()
print("loss    tau_h  G     sig_x  median 90-min factor")
for loss, tau, g, sig_scale, med_fac in results[:12]:
    print(f"{loss:.4f}  {tau:5.1f}  {g:.2f}  {sig_scale:.1f}   {med_fac:.3f}")
print("...")
for loss, tau, g, sig_scale, med_fac in results[-3:]:
    print(f"{loss:.4f}  {tau:5.1f}  {g:.2f}  {sig_scale:.1f}   {med_fac:.3f}")
