#!/usr/bin/env python3
"""Local validation for the GTA VI trophy forecaster.

Builds a pseudo-case from a practice case (flat launch-night history) with
the target set to the live Exophase page read at 2026-11-19T13:32:00Z
(rates saved in gta6-live-20261119-1332.json), runs main.py on it, and
scores the pinball loss against the live rates.  Also scores the house
persistence baseline and a pure-dilution variant for reference.

Usage: python3 validate.py [p00|p01] [--tune]
"""
import json
import math
import os
import subprocess
import sys

WS = "/home/hatch/workspace/makefun"
KIT = os.path.join(WS, "kits/gta6-trophy-rarity-forecast-01-kit/input")
LIVE = json.load(open(os.path.join(WS, "gta6-live-20261119-1332.json")))
LIVE_RATES = LIVE["rates"]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def logit(p):
    p = min(max(p, 0.0001), 0.9999)
    return math.log(p / (1.0 - p))


def pinball(y_pct, qs):
    y = logit(y_pct / 100.0)
    tot = 0.0
    n = 0
    for tau, q in zip(LEVELS, qs):
        d = y - logit(q / 100.0)
        tot += 2.0 * max(tau * d, (tau - 1.0) * d)
        n += 1
    return tot / n


def score(rows):
    losses = []
    for row, y in zip(rows, LIVE_RATES):
        losses.append(pinball(y, row["q"]))
    return sum(losses) / len(losses)


def baseline_persistence(case):
    """House baseline: q_tau = logistic(logit(r) + z_tau * 0.5*sqrt(days))."""
    from datetime import datetime
    def parse(s):
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    horizon_days = (parse(case["target"]) - parse(case["at"])) / 86400.0
    sig = 0.5 * math.sqrt(horizon_days)
    z = (-1.6448536269514722, -0.6744897501960817, 0.0,
         0.6744897501960817, 1.6448536269514722)
    rates = case["history"][-1]["rates"]
    rows = []
    for i, tr in enumerate(case["trophies"]):
        r = rates[i]
        lm = logit(r / 100.0)
        qs = [round(min(max(100.0 / (1.0 + math.exp(-(lm + zz * sig))), 0.01), 99.99), 2)
              for zz in z]
        rows.append({"id": tr["id"], "q": qs})
    return rows


def main():
    which = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("p00", "p01") else "p01"
    case = json.load(open(os.path.join(KIT, which + ".in")))
    # Pseudo-case: same flat history, target = live page read 13:32 UTC.
    case["at"] = case["history"][-1]["at"]
    case["target"] = "2026-11-19T13:32:00Z"
    from datetime import datetime
    def parse(s):
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    case["horizon_minutes"] = int((parse(case["target"]) - parse(case["at"])) / 60)
    os.makedirs("/tmp/gta6-val/in", exist_ok=True)
    os.makedirs("/tmp/gta6-val/out", exist_ok=True)
    with open("/tmp/gta6-val/in/val.in", "w") as f:
        json.dump(case, f)
    env = dict(os.environ, MAKEFUN_INPUT_DIR="/tmp/gta6-val/in",
               MAKEFUN_OUTPUT_DIR="/tmp/gta6-val/out")
    r = subprocess.run([sys.executable, os.path.join(WS, "gta6-trophy-forecast-01/main.py")],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print("main.py failed:", r.stderr)
        return 1
    rows = json.load(open("/tmp/gta6-val/out/val.out"))["trophies"]
    print(f"pseudo-case {which}: checkpoint {case['at']} -> target 13:32 UTC "
          f"({case['horizon_minutes']} min, {case['horizon_minutes']/1440:.2f} days)")
    print(f"  my model loss      : {score(rows):.4f}")
    print(f"  persistence base   : {score(baseline_persistence(case)):.4f}")
    # Pure dilution (f = 638/3868, no earner growth), same spread as model.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gta6main", os.path.join(WS, "gta6-trophy-forecast-01/main.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rates = case["history"][-1]["rates"]
    sig = math.sqrt(0.04 + 0.054 * (case["horizon_minutes"] / 1440.0))
    rows_d = []
    for i, tr in enumerate(case["trophies"]):
        med = rates[i] * (638.0 / 3868.0)
        rows_d.append({"id": tr["id"], "q": m.quantiles(med, sig)})
    print(f"  pure dilution f=.165: {score(rows_d):.4f}")
    # Per-trophy worst offenders.
    losses = [(pinball(y, row["q"]), row["id"]) for row, y in zip(rows, LIVE_RATES)]
    losses.sort(reverse=True)
    print("  worst trophies (loss, id):", [(round(l, 2), i) for l, i in losses[:6]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
