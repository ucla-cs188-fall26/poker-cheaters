// The numbers behind the statistics signal in scoring.js. Everything after "export default"
// is strict JSON, because backend/training/statistics/train.py rewrites this file.
// Weights are log-odds (positive means toward a bluff). A weight of 0 switches a feature off.
// These values are hand-set; "trainedOn" and "metrics" are filled in when a fit replaces them.
export default {
  "version": "baseline-0.2",
  "prior": 0.35,
  "confidence": { "preflop": 0.5, "postflop": 0.8 },
  "weights": {
    "overbet": 0.6,
    "small_bet": -0.3,
    "large_bet_on_dry_board": 0.5,
    "missed_flush_draw": 0.4,
    "multiway_pot": -0.5,
    "late_position_raise": 0.4,
    "early_position_raise": -0.5,
    "sudden_aggression": 0.3,
    "consistent_aggression": -0.2,
    "observer_blocks_nut_flush": 0.35,
    "log_bet_to_pot": 0,
    "barrel_count": 0,
    "raise": 0,
    "scare_card": 0,
    "flush_completed": 0,
    "paired_board": 0,
    "street_flop": 0,
    "street_turn": 0,
    "street_river": 0
  },
  "history": { "weight": 1.0, "strength": 4 },
  "trainedOn": null,
  "metrics": null
}
