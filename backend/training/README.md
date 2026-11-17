# Training

Put training and evaluation scripts here, in one folder per model (`training/facial/`, and so on). Save weights to `backend/artifacts/`, which is gitignored, and load them in the model's `load()`.

## Statistics model (the live tracker)

The tracker scores bets in `api/_lib/scoring.js`, not in this backend. Training only produces the numbers in `api/_lib/statisticsWeights.js`. The Python `StatisticsModel` in `app/ml/statistics/` is not part of this and no longer matches `scoring.js`.

```
phh_to_rows.py  public hand histories  ─▶ artifacts/statistics/public_rows.jsonl
export.mjs      rows or tracker bets   ─▶ artifacts/statistics/{public,tracker}.csv
train.py        fit and compare        ─▶ api/_lib/statisticsWeights.js (only with --write)
```

1. Download hand histories in PHH format, for example the Pluribus hands from [phh-dataset](https://github.com/uoftcprg/phh-dataset) (`data/pluribus`). Keep them outside the repo.
2. From `backend/`: `python training/statistics/phh_to_rows.py <folder> artifacts/statistics/public_rows.jsonl`
3. From the repo root: `node backend/training/statistics/export.mjs public backend/artifacts/statistics/public_rows.jsonl`
4. Optional, from the repo root: `node --env-file=.env.local backend/training/statistics/export.mjs tracker` (reads labelled tracker bets from Neon, select only).
5. From `backend/`, after `pip install -e ".[training]"`: `python training/statistics/train.py artifacts/statistics/public.csv --tracker artifacts/statistics/tracker.csv`

`train.py` prints the fitted model against the current weights on held-out bets. Add `--write` to save the weights. It refuses unless there are at least 200 bets and 50 bluffs, the held-out log loss beats the current weights, and (with 30 or more tracker bets) the tracker log loss is no worse. Review the diff to `statisticsWeights.js` and open a PR.

Things to know:

- `export.mjs` calls the same `statisticsFeatures()` as live scoring, so there is one definition of each feature. Labels come from `api/_lib/bluffRule.js`, which still needs team sign-off.
- Only postflop bets have labels, so the two preflop position weights and the nut-flush blocker weight stay hand-set. The prior also stays as it is; the fit's intercept is folded into the street weights.
- A bet's size is the total bet or raised to, and the pot is the pot before the bet, in both the tracker and the public data.
- The Pluribus hands are five professionals and a bot playing six-handed, with every player's cards known. Home games differ, so check against tracker bets before trusting a fit.
- Datasets, CSVs and anything else in `artifacts/` are gitignored. Never commit them.

## Training data for the decision model

Each row of this query is one labelled decision with its model outputs:

```sql
SELECT d.id, d.game_state, d.was_bluff, d.outcome_source,
       s.model_name, s.model_version, s.status, s.score, s.confidence, s.evidence
FROM decisions d
JOIN signal_outputs s ON s.decision_id = d.id
WHERE d.was_bluff IS NOT NULL;
```

## Rules for evaluation

- Never use `revealed_target_cards` as a model input. Use it only to work out the label.
- Split by session, not by row, so one player's game can't appear in both training and test data.
- Report results against a statistics-only baseline. If the facial and speech models don't beat it on held-out data, say so.
- Set `calibrated = True` in a decision model only after fitting and checking it on held-out data.
