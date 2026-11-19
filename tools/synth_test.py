#!/usr/bin/env python3
"""Synthetic test of the growing-history path.

Generates a final-like case (history 18 Nov 23:50 -> 21 Nov 11:30, reads
every 30 min, then every 10 min; target 26 Nov 00:00) from a KNOWN
data-generating process that differs from the program's fit family
(saturating-exponential N, per-trophy power-law earners), then checks the
program's forecast against the true rates at the target.  Also exercises
edge cases: empty history, single read, missing (null) rates.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-trophy-forecast-01")
import main as M

KIT = "/home/hatch/workspace/makefun/kits/gta6-trophy-rarity-forecast-01-kit/input"
base = json.load(open(os.path.join(KIT, "p01.in")))
TROPHIES = base["trophies"]


def epoch(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def iso(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- truth: N(t) saturating exponential, E_i(t) power law in time -------
LAUNCH = epoch("2026-11-19T00:00:00Z")
N_INF = 400_000.0
N_TAU_H = 60.0


def n_true(t):
    h = max(0.0, (t - LAUNCH) / 3600.0)
    return 638.0 + (N_INF - 638.0) * (1.0 - math.exp(-h / N_TAU_H))


def beta_i(i):
    # per-trophy dilution exponent, spread like the real ones
    return 0.95 - 0.93 * ((i * 7) % 10) / 10.0


def e_true(i, t):
    # earners grow as a power of time since launch, anchored at the
    # launch-night counts (k_i earners among 638 players)
    h = max(0.1, (t - LAUNCH) / 3600.0)
    k = base["history"][-1]["rates"][i] / 100.0 * 638.0
    if k <= 0:
        return 0.0
    # power-law growth with exponent tied to beta: fast trophies (low
    # beta) grow fast
    alpha = 1.2 * (1.0 - beta_i(i)) + 0.15
    return k * (h / 3.0) ** alpha


def rate_true(i, t):
    n = n_true(t)
    return 100.0 * min(e_true(i, t), n) / n


# --- build a final-like case ---------------------------------------------
t_start = epoch("2026-11-18T23:50:00Z")
t_deadline = epoch("2026-11-21T11:30:00Z")
t_target = epoch("2026-11-26T00:00:00Z")
reads = []
t = t_start
slot = 0
while t <= t_deadline:
    # Exophase refreshes ~hourly: rates are stale between refreshes.
    # Simulate: the read shows the values as of the last refresh (floor to
    # the previous hour).
    t_ref = math.floor((t - LAUNCH) / 3600.0) * 3600.0 + LAUNCH
    n = n_true(t_ref)
    rates = []
    tot = 0
    for i in range(55):
        r = round(rate_true(i, t_ref), 2)
        rates.append(r)
        tot += r / 100.0 * n
    reads.append({"slot": slot, "at": iso(t), "players_tracked": int(n),
                  "trophies_earned": int(tot), "rates": rates})
    slot += 1
    # every 30 min early, every 10 min in the last 6h (in-cluster recorder)
    t += 600 if t > t_deadline - 6 * 3600 else 1800

case = dict(base)
case["id"] = "f00-synth"
case["at"] = iso(t_deadline)
case["target"] = iso(t_target)
case["horizon_minutes"] = int((t_target - t_deadline) / 60)
case["history"] = reads
rows = M.forecast_case(case)

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def logit(p):
    p = min(max(p, 0.0001), 0.9999)
    return math.log(p / (1.0 - p))


def pinball(y, qs):
    yy = logit(y / 100.0)
    return sum(2.0 * max(t * (yy - logit(q / 100.0)), (t - 1.0) * (yy - logit(q / 100.0)))
               for t, q in zip(LEVELS, qs)) / 5


losses = []
for r in rows:
    y = rate_true(r["id"] - 1, t_target)
    losses.append(pinball(y, r["q"]))
print(f"synthetic final-like case: {len(reads)} reads, "
      f"history {iso(t_start)} -> {iso(t_deadline)}, target {iso(t_target)}")
print(f"  N at deadline {n_true(t_deadline):,.0f}, N at target {n_true(t_target):,.0f}")
print(f"  mean pinball loss vs truth: {sum(losses)/len(losses):.4f}")
print(f"  median |logit error|: {sorted(abs(logit(r['q'][2]/100.0)-logit(rate_true(r['id']-1,t_target)/100.0)) for r in rows)[27]:.3f}")
# persistence baseline on the same synthetic case
last = reads[-1]["rates"]
sig = 0.5 * math.sqrt((t_target - t_deadline) / 86400.0)
z = (-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722)
bl = []
for i in range(55):
    lm = logit(last[i] / 100.0)
    qs = [round(min(max(100.0 / (1.0 + math.exp(-(lm + zz * sig))), 0.01), 99.99), 2) for zz in z]
    bl.append(pinball(rate_true(i, t_target), qs))
print(f"  persistence baseline loss: {sum(bl)/len(bl):.4f}")

# --- edge cases -----------------------------------------------------------
print("edge cases:")
print("  empty history:", M.forecast_case({"trophies": TROPHIES, "history": [], "at": "2026-11-19T03:00:00Z", "target": "2026-11-19T04:30:00Z"})[0])
c1 = dict(base); c1["history"] = [base["history"][0]]
print("  single read ok:", len(M.forecast_case(c1)) == 55)
c2 = json.loads(json.dumps(base))
for r in c2["history"]:
    r["rates"] = [None if (i % 3 == 0) else v for i, v in enumerate(r["rates"])]
print("  null rates ok:", len(M.forecast_case(c2)) == 55)
c3 = json.loads(json.dumps(base))
c3["history"] = [dict(r, rates=[100.0] * 55) for r in c3["history"]]
print("  all-100% rates ok:", M.forecast_case(c3)[0])
