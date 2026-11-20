# gta6-trophy-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Firman2712/gta6-trophy-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Firman2712/gta6-trophy-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for GTA VI's trophy earn rates on Exophase: given one sealed case of trophy-page reads, it forecasts five quantiles (5%, 25%, 50%, 75%, 95%) of each of the 55 trophies' earn rate at the target time, scored by pinball loss on the logit scale. It models players-tracked and earners separately, so an Exophase refresh (players jump, rates drop) falls out of the model instead of needing to be patched in. See `METHOD.md` for the model and how it was validated.

Practice cut (server): round 1 scored loss **0.0955, rank 40/3,588**; round 2 scored loss **0.0809, rank 25/3,689**; round 3 scored loss **0.0789, rank 25/3,768** (flat-path spread fix). Round 4 rebuilds the growing path (the final case): sparse-refresh validation mean loss 0.38 → 0.093 vs persistence ~0.23 — see `METHOD.md`.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every `*.in` case in the input directory and writes one `*.out` per case to the output directory, each `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`. Without those variables it uses `/input` and `/output`, as the harness provides. It is deterministic: the same case bytes in give the same bytes out, and it runs in well under a second per case.

---

<sub>Entered on [make.fun](https://make.fun/?ref=2nb4vgz3&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
