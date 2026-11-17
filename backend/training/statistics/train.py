"""Fits the statistics model's weights and compares them with the current hand-set ones.

Usage, from backend/ (needs `pip install -e ".[training]"`):
    python training/statistics/train.py artifacts/statistics/public.csv
    python training/statistics/train.py artifacts/statistics/public.csv \
        --tracker artifacts/statistics/tracker.csv --write

The CSVs come from export.mjs. The model is a logistic regression over the feature columns,
so each coefficient is one feature's log-odds and the result can be read by scoring.js.
Results are measured on held-out groups (hands, or players with --group-by player).

Without --write nothing is saved. With --write, api/_lib/statisticsWeights.js is rewritten
only if the gate passes: enough data, and a better held-out log loss than the current weights.
"""

import argparse
import csv
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

WEIGHTS_FILE = Path(__file__).resolve().parents[3] / "api" / "_lib" / "statisticsWeights.js"
WEIGHTS_MARKER = "export default "
META_COLUMNS = ["source", "group", "player", "street", "was_bluff", "baseline"]
# No labelled data exists for these (preflop bets, the observer's own cards), so they stay hand-set.
HAND_SET = {"late_position_raise", "early_position_raise", "observer_blocks_nut_flush"}
# Every postflop bet is on exactly one street. The flop is the reference level during the fit,
# and the fit's intercept is folded into the three street weights when saving (see write_weights).
STREETS = ["street_flop", "street_turn", "street_river"]

MIN_BETS = 200
MIN_BLUFFS = 50
MIN_TRACKER_BETS = 30
FOLDS = 5
RELIABILITY_BINS = 5


def load(path: Path) -> dict:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        sys.exit(f"{path} has no rows")
    codes = [name for name in rows[0] if name not in META_COLUMNS]
    return {
        "codes": codes,
        "x": np.array([[float(row[code]) for code in codes] for row in rows]),
        "y": np.array([int(row["was_bluff"]) for row in rows]),
        "baseline": np.array([float(row["baseline"]) for row in rows]),
        "group": np.array([row["group"] for row in rows]),
        "player": np.array([row["player"] or row["group"] for row in rows]),
    }


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    both_classes = 0 < y.sum() < len(y)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if both_classes else None,
    }


def show(title: str, fitted: dict, baseline: dict) -> None:
    print(f"\n{title}")
    print(f"  {'':10}{'log loss':>10}{'Brier':>10}{'AUC':>10}")
    for name, m in (("fitted", fitted), ("hand-set", baseline)):
        auc = f"{m['auc']:.3f}" if m["auc"] is not None else "n/a"
        print(f"  {name:10}{m['log_loss']:>10.4f}{m['brier']:>10.4f}{auc:>10}")


def reliability(y: np.ndarray, p: np.ndarray) -> None:
    print("\nReliability of the fitted model on held-out bets (estimate vs outcome)")
    edges = np.quantile(p, np.linspace(0, 1, RELIABILITY_BINS + 1))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, RELIABILITY_BINS - 1)
    for b in range(RELIABILITY_BINS):
        mask = bins == b
        if mask.any():
            estimate, bluffs = p[mask].mean(), y[mask].mean()
            print(f"  estimate {estimate:.2f}   bluffs {bluffs:.2f}   bets {mask.sum()}")


def write_weights(model, codes: list[str], trained_on: dict, results: dict) -> str:
    text = WEIGHTS_FILE.read_text()
    start = text.index(WEIGHTS_MARKER)
    weights = json.loads(text[start + len(WEIGHTS_MARKER) :])
    fitted = dict(zip(codes, (float(c) for c in model.coef_[0]), strict=True))
    # The prior stays as it is (it also drives preflop reads and the opponent-history term).
    # The gap between the fit's intercept and that prior goes into the street weights instead,
    # which is exact because every postflop bet has exactly one street feature.
    prior = weights["prior"]
    gap = float(model.intercept_[0]) - math.log(prior / (1 - prior))
    for street in STREETS:
        fitted[street] = fitted.get(street, 0.0) + gap
    # Live scoring multiplies every statistics term by the postflop confidence, so divide it out.
    scale = weights["confidence"]["postflop"]
    for code, log_odds in fitted.items():
        weights["weights"][code] = round(log_odds / scale, 4)
    weights["version"] = f"fitted-{date.today():%Y%m%d}"
    weights["trainedOn"] = trained_on
    weights["metrics"] = results
    body = json.dumps(weights, indent=2)
    WEIGHTS_FILE.write_text(text[:start] + WEIGHTS_MARKER + body + "\n")
    return weights["version"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("public", type=Path, help="CSV to fit on (export.mjs public)")
    parser.add_argument("--tracker", type=Path, help="CSV of tracker bets, used only as a check")
    parser.add_argument("--group-by", choices=["hand", "player"], default="hand")
    parser.add_argument("--c", type=float, default=1.0, help="inverse L2 strength")
    parser.add_argument("--write", action="store_true", help="save the weights if the gate passes")
    args = parser.parse_args()

    data = load(args.public)
    x, y = data["x"], data["y"]
    skipped = HAND_SET | {STREETS[0]}
    keep = [i for i, c in enumerate(data["codes"]) if c not in skipped and x[:, i].std() > 0]
    codes = [data["codes"][i] for i in keep]
    x = x[:, keep]
    groups = data["player"] if args.group_by == "player" else data["group"]
    group_count = len(set(groups))
    print(f"{len(y)} bets, {int(y.sum())} bluffs ({y.mean():.1%}), {group_count} {args.group_by}s")

    problems = []
    if len(y) < MIN_BETS or y.sum() < MIN_BLUFFS:
        problems.append(f"needs at least {MIN_BETS} bets and {MIN_BLUFFS} bluffs")
    if group_count < FOLDS or y.sum() == 0 or y.sum() == len(y):
        sys.exit(f"Not enough data to fit: need {FOLDS}+ {args.group_by}s, bluffs and value bets")

    def fit(features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
        return LogisticRegression(C=args.c, max_iter=1000).fit(features, labels)

    held_out = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=FOLDS).split(x, y, groups):
        held_out[test] = fit(x[train], y[train]).predict_proba(x[test])[:, 1]
    fitted, baseline = metrics(y, held_out), metrics(y, data["baseline"])
    show(f"Held-out public bets ({FOLDS} folds, split by {args.group_by})", fitted, baseline)
    reliability(y, held_out)
    if fitted["log_loss"] >= baseline["log_loss"]:
        problems.append("held-out log loss is not better than the hand-set weights")

    model = fit(x, y)
    print("\nFitted log-odds per feature (all public bets)")
    ranked = sorted(zip(codes, model.coef_[0], strict=True), key=lambda item: -abs(item[1]))
    for code, coefficient in ranked:
        print(f"  {code:26}{coefficient:+.3f}")
    print(f"  {'(intercept)':26}{model.intercept_[0]:+.3f}")

    results = {"held_out": fitted, "hand_set": baseline, "tracker": None}
    tracker_bets = 0
    if args.tracker:
        tracker = load(args.tracker)
        tracker_bets = len(tracker["y"])
        columns = [tracker["codes"].index(code) for code in codes]
        if tracker_bets >= MIN_TRACKER_BETS:
            ours = metrics(tracker["y"], model.predict_proba(tracker["x"][:, columns])[:, 1])
            theirs = metrics(tracker["y"], tracker["baseline"])
            show(f"Tracker bets ({tracker_bets})", ours, theirs)
            results["tracker"] = {"fitted": ours, "hand_set": theirs}
            if ours["log_loss"] > theirs["log_loss"]:
                problems.append("log loss on tracker bets is worse than the hand-set weights")
        else:
            print(f"\nTracker check skipped: {tracker_bets} labelled bets, needs {MIN_TRACKER_BETS}")  # noqa: E501

    print()
    if problems:
        print("Gate failed, weights not saved:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1 if args.write else 0)
    if not args.write:
        print("Gate passed. Run again with --write to save the weights.")
        return
    trained_on = {
        "bets": int(len(y)),
        "bluffs": int(y.sum()),
        "groups": group_count,
        "group_by": args.group_by,
        "tracker_bets_checked": tracker_bets if results["tracker"] else 0,
    }
    version = write_weights(model, codes, trained_on, results)
    print(f"Saved {WEIGHTS_FILE} as {version}. Review the diff and open a PR.")


if __name__ == "__main__":
    main()
