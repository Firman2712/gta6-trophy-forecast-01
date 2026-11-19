# Method — GTA VI trophy rarity forecast

`main.py` forecasts, for each of the 55 GTA VI trophies, five quantiles of
its Exophase earn rate at the target time. Score is pinball loss on the
logit scale. The program is standard-library Python only, offline,
deterministic, and runs in well under a second per case.

## The core idea: model players and earners separately

A trophy's earn rate is `rate_i(t) = E_i(t) / N(t)`, where `N(t)` is the
players Exophase tracks and `E_i(t)` is the players who have earned trophy
`i`. Both are modelled separately and the rate is the ratio, so an
Exophase refresh — `N` jumps, every rate drops — falls out of the model
instead of needing to be patched in. The two regimes in the data need two
paths:

- **Growing history** (the final case: ~2.5 days of reads up to the
  deadline). `N(t)` is fitted two ways and blended by the observed
  deceleration (recent log-log slope vs overall slope): a power law,
  `log N` against `log(hours since launch)`, on the last 24h of distinct
  reads, and a saturating exponential `N = A - B·exp(-h/τ)` (τ by grid
  search) on all distinct reads. No deceleration → power law; strong
  deceleration → saturating. A pure exponential is never used: it wildly
  overshoots a 4.5-day horizon. `E_i(t)` is fitted per trophy as
  `log E_i` against `log(hours since launch)` on the last 24h of distinct
  reads, with the slope shrunk toward the cross-trophy median slope
  (empirical-Bayes pooling, weighted by trophy size). Earners never
  decrease and never exceed `N`. Only reads where `N` changed count as
  fresh — stale repeats between refreshes carry no new information.
- **Flat history** (the practice cases: the launch-night page never
  refreshed, so the history is one stale snapshot repeated). `N(t)`
  follows an anchor curve fitted to the three known public reads — 638
  players at 03:00, 3,868 at the 04:30 refresh (recorded in the kit's
  notice), ~15,000 at 13:32 — log-log interpolated. Each trophy's rate
  then falls as `(N_target/N_last)^(-β_i)`, where `β_i` relaxes from 1
  (short horizon: new players have earned nothing yet) toward a
  per-trophy long-horizon exponent `β_inf` (fishing 0.02, chapter-1 0.17,
  most trophies 0.95), plus a small uniform earners-growth term for the
  players already tracked. The earners' growth spans the last refresh
  (the first read's time — the history's rates are stale, the page last
  refreshed before the first read) to the target, not the checkpoint to
  the target. `β_inf` is learned from the live public
  Exophase page at build time (13:32 UTC 19 Nov 2026): most trophies'
  rates fell ~20× as `N` grew ~20× (earners ~flat), while fishing held
  at 0.93×, chapter-1 at 0.58×, arcade at 0.44×.

## Quantiles

Median = `E(target)/N(target)`. Logit spread σ grows with horizon and
with how the case was forecast: the flat (anchor) path is tightly
anchored, so σ is modest (`0.85·√(0.04 + 0.054·days)`); the growing
(fit) path extrapolates 4.5 days from a fit, so σ is wider
(`√(0.09 + 0.08·days)`). Fast trophies (low `β_inf`) get a 1.15× wider
spread. `q_τ = logistic(logit(median) + z_τ·σ)` with the standard normal
quantiles. The ~100% trophy is forecast at 99.99% with a tight spread.

## Round 3 (19 Nov 2026, after the second submission's practice score)

Round 2 scored **practice loss 0.0809, rank 25/3,689** (typical miss
0.03 pp, coverage_90 0.9545) — just outside the top 20 (line 0.0784).
Coverage 0.9545 vs the nominal 0.90 means the flat-path spread was now
~20% too wide, so the multiplier went 1.0 → 0.85 (σ ≈ 0.18 at 90 min).
Round 3 touches only the flat path, so the final-case forecast is
unchanged. 10h pseudo-case loss 0.0457; synthetic final-like 0.1882.

## How it was validated

No local grading exists — the server grades every entry — so validation
used the live public Exophase page as ground truth:

- **10-hour pseudo-case** (practice history → live page at 13:32 UTC
  19 Nov, 631 min horizon): pinball loss **0.046**, vs 2.38 for the
  house persistence baseline and 1.01 for pure dilution. The flat-path
  parameters (β relaxation τ = 3h, earners-growth 0.10) were tuned on
  this; the spread was deliberately *not* tuned to the minimum (that
  would overfit, since `β_inf` was itself fitted to the same live page).
- **Synthetic final-like cases** (2.5-day history, 4.5-day horizon,
  known truth, two regimes): saturating-`N` truth **0.19** vs persistence
  0.25; power-law-`N` truth **0.21** vs persistence 0.36. The model beats
  persistence in both regimes; the earlier power-law-only `N` ensemble
  lost badly on the power-law truth, which is what motivated the
  deceleration-blended ensemble.
- **Edge cases**: empty history, single read, missing (`null`) rates, and
  all-rates-100% all produce valid output. Two runs are byte-identical.

## What the practice score implies

The first submission (round 1) scored **practice loss 0.0955, rank
40/3,588** (typical miss 0.04 pp, coverage_90 0.7727) — outside the top
20 (line 0.0804). That score diagnosed two problems, both fixed in
round 2 (this commit):

- The history's rates are stale (the page last refreshed before the
  first read), so the earners' growth must span the last refresh →
  target (~4.7h), not checkpoint → target (1.5h). Growing over only
  1.5h understated the earners.
- The flat-path spread was too narrow: coverage 0.77 vs the nominal
  0.90 implies a true error scale of ~0.20 in logit, but σ was 0.146
  at 90 min. The 0.7 multiplier was removed, so σ ≈ 0.21 at 90 min.
  The median factor was raised only ~5% (earners-growth 0.05 → 0.10):
  the loss 0.0955 bounds the median error to ~0.07, so a large raise
  would overshoot if the medians were already accurate.

The practice cut is the iteration signal only; the final case (26 Nov
00:00 UTC) decides the prizes, and it is a growth-extrapolation
problem, not a refresh shock — which is why the growing-history path,
not the flat path, carries the final. Round 2 touches only the flat
path, so the final-case forecast is unchanged.

## Files

- `main.py` — the entrypoint (repo root, `python3 main.py`).
- `tools/validate.py` — builds the 10h pseudo-case and scores it.
- `tools/tune.py` — grid-searches the flat-path parameters on it.
- `tools/synth_test.py` — synthetic final-like cases + edge cases.
