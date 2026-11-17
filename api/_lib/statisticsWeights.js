// The numbers behind the statistics signal in scoring.js. Everything after "export default"
// is strict JSON, because backend/training/statistics/train.py rewrites this file.
// Weights are log-odds (positive means toward a bluff). A weight of 0 switches a feature off.
// These values are hand-set; "trainedOn" and "metrics" are filled in when a fit replaces them.
export default {
  "version": "fitted-20261117",
  "prior": 0.35,
  "confidence": {
    "preflop": 0.5,
    "postflop": 0.8
  },
  "weights": {
    "overbet": -0.1295,
    "small_bet": -0.0366,
    "large_bet_on_dry_board": 0.1259,
    "missed_flush_draw": 0.0277,
    "multiway_pot": -0.4716,
    "late_position_raise": 0.4,
    "early_position_raise": -0.5,
    "sudden_aggression": 0.12,
    "consistent_aggression": -0.0204,
    "observer_blocks_nut_flush": 0.35,
    "log_bet_to_pot": 0.2152,
    "barrel_count": -0.0547,
    "raise": -1.3407,
    "scare_card": 0.1085,
    "flush_completed": -0.5222,
    "paired_board": 0.2902,
    "street_flop": 0.4668,
    "street_turn": -0.7607,
    "street_river": -0.5973
  },
  "history": {
    "weight": 1.0,
    "strength": 4
  },
  "trainedOn": {
    "bets": 7626,
    "bluffs": 2245,
    "groups": 4919,
    "group_by": "hand",
    "tracker_bets_checked": 0
  },
  "metrics": {
    "held_out": {
      "log_loss": 0.5826267486278927,
      "brier": 0.19824043010284292,
      "auc": 0.6324623593117581
    },
    "hand_set": {
      "log_loss": 0.6355031408817161,
      "brier": 0.22183156848401447,
      "auc": 0.4868935448449526
    },
    "tracker": null
  }
}
