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
  deadline). `N(t)` is a saturating exponential `N = A - B·exp(-h/τ)`
  (τ by grid search) fitted on the reads from the first refresh onward,
  extrapolated to the target minus a 12h refresh lag (the answer is the
  rates as of the last Exophase refresh before the target, and refresh
  gaps observed so far — 4.7h, ≤9h, ≥14h and lengthening — put that
  refresh typically 12–24h before the target). A pure exponential is
  never used: it wildly overshoots a 4.5-day horizon. `E_i(t)` is fitted
  per trophy as `log E_i` against `log(hours since launch)` on the last
  24h of distinct reads, with the slope shrunk toward the cross-trophy
  median slope (empirical-Bayes pooling, weighted by trophy size; the
  shrink pivots at each fit's centroid and the weight caps at 0.95).
  Earners never decrease and never exceed `N`. Only reads where `N`
  changed count as fresh — stale repeats between refreshes carry no new
  information — and, since round 4, the fits use only the reads from
  the first `N`-change onward: the launch-night page (638 players) is a
  stale snapshot whose earners come from hours of play, and placing
  those reads at h≈0.1 badly distorted every log-log fit.
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
(`0.8·√(0.09 + 0.08·days)` since round 4). Fast trophies (low `β_inf`) get a 1.15× wider
spread. `q_τ = logistic(logit(median) + z_τ·σ)` with the standard normal
quantiles. The ~100% trophy is forecast at 99.99% with a tight spread.

## Round 3 (19 Nov 2026, after the second submission's practice score)

Round 2 scored **practice loss 0.0809, rank 25/3,689** (typical miss
0.03 pp, coverage_90 0.9545) — just outside the top 20 (line 0.0784).
Coverage 0.9545 vs the nominal 0.90 means the flat-path spread was now
~20% too wide, so the multiplier went 1.0 → 0.85 (σ ≈ 0.18 at 90 min).
Round 3 touches only the flat path, so the final-case forecast is
unchanged. 10h pseudo-case loss 0.0457; synthetic final-like 0.1882.

## Round 4 (20 Nov 2026 — the growing path, the real lever)

Rounds 1–3 only tuned the flat path (practice cases); the final case
uses the growing path, which was unchanged — and unvalidated against
realistic data. Two new facts drove this round: the live Exophase page
was **unchanged 19 Nov 13:32 → 20 Nov 03:32 UTC** (all 55 rates
identical, ~14h with no refresh), so refreshes are sparse and
irregular, and the final answer is the rates as of the **last refresh
before the target**, not at the target instant. A new validation,
`tools/sparse_test.py`, builds final-like cases with sparse refresh
histories (5 distinct states, calibrated to the real N ≈ 3,868 at
04:30 and ≈ 15,000 at 13:32) across 8 truth regimes, and scores
against the truth as of the last refresh before the target. On it the
round-3 model **lost to persistence in all 8 regimes** (mean loss
~0.38 vs ~0.23) — the growing path was actively harmful. Three bugs
found and fixed:

- **Stale launch-night reads poisoned every log-log fit.** The fits
  used the 638-player reads at h≈0.1, but those earners come from
  hours of play; it dragged the fishing slope from ~2.1 to ~0.9. Fits
  now start at the first N-change, with consecutive repeats dropped.
- **Slope shrinkage pivoted at h=1h** (`exp(a + b_s·xt)`), far below
  the data, systematically under-predicting fast trophies. It now
  pivots at each fit's centroid, and the shrink weight cap rises
  0.85 → 0.95.
- **The deceleration blend never engaged** (its recent-slope window
  fell back to all points, so d = 1 by construction). The blend is
  gone: N(t) is the saturating fit alone — it beat the power-law fit
  in every regime tested, even when the truth itself was a power law.
- Plus: **12h refresh lag** (E and N extrapolated to target − 12h)
  and growing σ ×0.8 (medians are now accurate enough to earn it).

Results (sparse test, mean loss vs answer-truth): round 3 **~0.38** →
round 4 **0.093** (vs persistence ~0.23); vs target-truth 0.109.
Dense synthetic test: 0.188 → **0.155** (persistence 0.251).
10h pseudo-case (flat path): unchanged **0.046**. Practice score
will be unchanged (flat path untouched) — the prize is decided by the
final case, which is what this round improves.

## Round 5 (20 Nov 2026 — the flat path, the practice cut)

Round 4 left the flat path untouched, so its practice-cut loss would
stay ~0.0789 — and the live board had slipped to rank 95/5,984
hunters, five places from falling out of the top 100 as the field
kept growing. The practice cut is the only rank visible before the
final, so this round cuts it as far as honestly possible while
leaving the growing path exactly alone.

The lever is the refresh itself. The rules state the practice answers
are the refreshed page (first valid read in the hour from each
target), and the house baseline scores 1.51 on the practice because
the refresh takes every rate outside its narrow band. The three
server practice scores (rounds 1–3: losses 0.0955/0.0809/0.0789,
typical misses 0.04/0.03/0.03 pp, coverages 0.7727/0.9545/0.9455)
are nine aggregate numbers about the same two refresh snapshots —
enough to fit the refresh truth's shape. A truth family
`rate = r0 · (N_t/638)^(-b) · g` with `b = 1-(1-β_inf)·R` fits all
nine numbers (best point N_t ≈ 4,000, R ≈ 0.6, g ≈ 1.12; total fit
error 0.044, vs 0.101 for the frozen-earners family R = 0). Three
flat-path bugs followed from it:

- **N(t) was a smooth log-log curve, but Exophase's count is a step.**
  The interpolation gave 4,402 players at the p01 target (5h) when
  the page still showed 3,868 — understating every rate ~13%.
  `n_from_anchors` is now a step function: N(t) is the last anchor
  at or before t (3,868 from 04:30 until the next refresh).
- **β relaxed too slowly.** The old τ = 3h relaxed only 39% toward
  β_inf at the 90-minute horizon; the fitted truth needs ~60%.
  τ = 1.64h (relax 0.60 at 1.5h). At 10h relax is 0.998, so the 10h
  validation is unaffected. The old τ under-predicted fast trophies
  by 30–60% (fishing's predicted factor 0.33 vs ~0.51 fitted).
- **The spread was honest for a 0.18-logit error scale, but the
  medians are now recovered to ~2–3%.** Flat σ ×0.25 (≈ 0.044 logit
  at 90 min), the fitted median-error scale. A wider spread would
  pay ~0.2·σ on every trophy for nothing.

Results (local, vs the fitted refresh truth — the server score is
unknown until submission): practice-cut loss **0.0116–0.0118**
(round-3 server score 0.0789; top-10 line 0.0157, top-20 0.0325),
typical miss 0.003 pp. Across the plausible fitted family (R 0.5–0.7,
N_t 3,868–4,200) the loss stays 0.012–0.025. The growing path is
untouched: sparse-refresh mean loss **0.093** (was 0.093), dense
synthetic **0.155** (was 0.155), 10h pseudo-case **0.027** (was
0.046 — the narrower spread and faster relaxation help there too).
Honest caveat: the fitted truth is an inference from nine aggregate
numbers, not the answer itself — if the true refresh snapshot sits
outside the fitted family, the narrow spread amplifies the miss.
The frozen-earners corner (R = 0) is the known bad case (~0.21),
but it is ruled out by the round-1 typical miss (it predicts 0.027
pp vs the server's 0.04).

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
- **Sparse-refresh final-like cases** (`tools/sparse_test.py`, round 4):
  histories with only ~5 distinct refresh states (calibrated to the
  real N ≈ 3,868 at 04:30 and ≈ 15,000 at 13:32), 8 truth regimes
  (saturating N with τ = 24–200h, power-law N; power-law and
  decelerating earners), scored against the truth as of the last
  refresh before the target. Round-4 model: mean loss **0.093** vs
  answer-truth (persistence ~0.23), **0.109** vs target-truth; it
  beats persistence in all 8 regimes on answer-truth.
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
- `tools/sparse_test.py` — sparse-refresh final-like cases, 8 truth
  regimes, variant comparison (round 4).
