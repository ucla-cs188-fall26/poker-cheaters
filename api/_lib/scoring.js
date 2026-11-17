// Turns one bet or raise into a bluff estimate: features of the bet and the board, weighted by
// statisticsWeights.js, plus the browser's face score, fused as log-odds. The weights are
// hand-set until a fit replaces them, so the output is an uncalibrated estimate and must never
// be described as a probability.
//
// Positive log-odds mean toward a bluff. Every reason shown to the user is a real term of the sum.
import { RANKS } from './cards.js'
import weights from './statisticsWeights.js'

export const SCORING_VERSION = weights.version
// Every statistics feature, in a fixed order. The training export uses this as its columns.
export const STATISTICS_CODES = Object.keys(weights.weights)

const PRIOR = weights.prior
const WEIGHTS = { statistics: 1.0, live_facial: 0.5 }
const MAX_REASONS = 3
const MIN_CONTRIBUTION = 0.01

const LARGE_BET_RATIO = 0.75
const OVERBET_RATIO = 1.25
const SMALL_BET_RATIO = 0.33
const STREET_ORDER = ['preflop', 'flop', 'turn', 'river']

const logit = (p) => Math.log(p / (1 - p))
const sigmoid = (x) => 1 / (1 + Math.exp(-x))

const rankIndex = (card) => RANKS.indexOf(card[0])
const countBy = (items) => {
  const counts = {}
  for (const item of items) counts[item] = (counts[item] ?? 0) + 1
  return counts
}
const isAggressive = (a) => a.action === 'bet' || a.action === 'raise'

// Three board cards within a five-rank window let two hole cards make a straight.
function straightPossible(ranks) {
  const values = new Set(ranks)
  if (values.has(RANKS.indexOf('A'))) values.add(-1) // The ace also plays low.
  for (let start = -1; start < 9; start += 1) {
    if ([...values].filter((v) => v >= start && v < start + 5).length >= 3) return true
  }
  return false
}

// Unpaired, with no flush or straight possible.
function isDry(board) {
  const ranks = board.map(rankIndex)
  const unpaired = new Set(ranks).size === ranks.length
  const maxSuited = Math.max(...Object.values(countBy(board.map((c) => c[1]))))
  return unpaired && maxSuited <= 2 && !straightPossible(ranks)
}

// A flush draw was possible on the turn, and the river did not complete it.
function missedFlushDraw(board) {
  const turnSuits = countBy(board.slice(0, 4).map((c) => c[1]))
  return Math.max(...Object.values(turnSuits)) === 2 && (turnSuits[board[4][1]] ?? 0) < 2
}

// A feature is one input to the sum: its value times its weight is its log-odds.
// Rules that fire have value 1; the numeric features carry their own value.
const feature = (code, description, value = 1) => ({ code, description, value })

function betSizeFeatures(state, ratio) {
  const items = []
  if (ratio >= OVERBET_RATIO) {
    items.push(feature('overbet', `Bet is ${ratio.toFixed(1)}x the pot, a size often used to force folds`))
  } else if (ratio <= SMALL_BET_RATIO) {
    items.push(feature('small_bet', `Bet is only ${Math.round(ratio * 100)}% of the pot, which pressures few hands`))
  }
  if (state.street !== 'preflop' && ratio >= LARGE_BET_RATIO && isDry(state.board)) {
    items.push(feature('large_bet_on_dry_board', 'Large bet on a dry board, where few strong hands are likely'))
  }
  if (state.street === 'river' && ratio >= LARGE_BET_RATIO && missedFlushDraw(state.board)) {
    items.push(feature('missed_flush_draw', 'Large river bet after a flush draw missed'))
  }
  if (state.street !== 'preflop') {
    items.push(feature('log_bet_to_pot', `Bet is ${Math.round(ratio * 100)}% of the pot`, Math.log(ratio)))
  }
  return items
}

function tableFeatures(state) {
  const items = []
  if (state.activePlayers >= 3) {
    items.push(
      feature(
        'multiway_pot',
        `${state.activePlayers} players are still in, and bluffs rarely work against several opponents`,
      ),
    )
  }
  if (state.street === 'preflop') {
    if (state.position === 'late') {
      items.push(feature('late_position_raise', 'Raise from late position, a common spot to steal the blinds'))
    } else if (state.position === 'early') {
      items.push(feature('early_position_raise', 'Raise from early position, which usually shows a strong range'))
    }
  }
  return items
}

function lineFeatures(state, ratio) {
  const items = []
  const earlier = state.priorActions.filter((a) => STREET_ORDER.indexOf(a.street) < STREET_ORDER.indexOf(state.street))
  const barrels = new Set(earlier.filter(isAggressive).map((a) => a.street)).size
  if (barrels > 0) {
    items.push(
      feature('barrel_count', `Bet or raised on ${barrels} earlier street${barrels === 1 ? '' : 's'} this hand`, barrels),
    )
  }
  if (state.action === 'raise' && state.street !== 'preflop') items.push(feature('raise', 'Raised instead of betting first'))

  const postflop = state.priorActions.filter((a) => a.street !== 'preflop')
  if (postflop.length === 0 || ratio < LARGE_BET_RATIO) return items
  if (postflop.every((a) => a.action === 'check' || a.action === 'call')) {
    items.push(feature('sudden_aggression', 'Large bet after only checking or calling on earlier streets'))
  } else if (postflop.every(isAggressive)) {
    items.push(feature('consistent_aggression', 'Has bet or raised on every earlier street, a consistent story'))
  }
  return items
}

// What the board looks like when the bet is made. Postflop only.
function boardFeatures(state) {
  const { board, street } = state
  if (street === 'preflop' || board.length === 0) return []
  const items = []
  const ranks = board.map(rankIndex)
  if (new Set(ranks).size !== ranks.length) items.push(feature('paired_board', 'The board is paired'))
  items.push(feature(`street_${street}`, `Bet came on the ${street}`))
  if (street === 'turn' || street === 'river') {
    const latest = board[board.length - 1]
    const earlier = ranks.slice(0, -1)
    if ('AK'.includes(latest[0]) || rankIndex(latest) > Math.max(...earlier)) {
      items.push(feature('scare_card', `The ${street} brought a high card that changes the board`))
    }
    if (countBy(board.map((c) => c[1]))[latest[1]] === 3) {
      items.push(feature('flush_completed', `The ${street} put three of one suit on the board`))
    }
  }
  return items
}

function blockerFeatures(state) {
  if (state.myCards.length === 0 || state.board.length === 0) return []
  const counts = countBy(state.board.map((c) => c[1]))
  const [suit, count] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]
  if (count >= 3 && state.myCards.includes(`A${suit}`)) {
    return [feature('observer_blocks_nut_flush', "You hold the flush suit's ace, so they can't have the nut flush")]
  }
  return []
}

// Every feature of one bet. Live scoring and the training export both call this, so the
// model is trained on exactly what it is served. Never add the bettor's own cards here.
export function statisticsFeatures(state) {
  const ratio = state.betSize / state.potSize
  return [
    ...betSizeFeatures(state, ratio),
    ...tableFeatures(state),
    ...lineFeatures(state, ratio),
    ...boardFeatures(state),
    ...blockerFeatures(state),
  ]
}

// How often this opponent's earlier shown bets were bluffs, pulled toward the prior so one
// or two showdowns can't swing the read. No history means no term.
function historyEvidence(history) {
  if (!history || !(history.labeled > 0)) return []
  const { weight, strength } = weights.history
  const rate = (history.bluffs + strength * PRIOR) / (history.labeled + strength)
  return [
    evidence(
      'opponent_bluff_rate',
      `Has bluffed ${history.bluffs} of ${history.labeled} shown bets in earlier hands`,
      weight * (logit(rate) - logit(PRIOR)),
    ),
  ]
}

const evidence = (code, description, logOdds) => ({ code, description, logOdds })

// The bet and the board, weighted by statisticsWeights.js, plus the opponent's history.
function statisticsSignal(state) {
  return {
    name: 'statistics',
    confidence: state.street === 'preflop' ? weights.confidence.preflop : weights.confidence.postflop,
    evidence: [
      ...statisticsFeatures(state).map((f) => evidence(f.code, f.description, f.value * (weights.weights[f.code] ?? 0))),
      ...historyEvidence(state.history),
    ],
  }
}

// --- Face signal -----------------------------------------------------------------------
// The browser measures the video; the server only judges the derived numbers.

const NEUTRAL_SCORE = 3.0
const LOG_ODDS_PER_POINT = 0.15
const MIN_FACE_LOG_ODDS = -0.3
const MAX_FACE_LOG_ODDS = 1.0
const MIN_SAMPLES = 3
const MIN_TRACKED_RATIO = 0.3
const TELL_Z = 2.0
const MAX_TELLS = 3

const FACE_GROUPS = [
  ['brow movement', /brow/],
  ['lip movement', /mouthPress|mouthPucker|mouthFunnel|mouthRoll|mouthShrug/],
  ['smile movement', /mouthSmile|mouthDimple/],
  ['jaw movement', /jaw/],
  ['eye narrowing', /eyeSquint|eyeWide/],
]

function faceLabel(name) {
  for (const [label, pattern] of FACE_GROUPS) if (pattern.test(name)) return label
  return name.replace(/([A-Z])/g, ' $1').toLowerCase().trim()
}

function tellsOf(face) {
  return [...face.deviations]
    .filter((d) => Math.abs(d.z) >= TELL_Z)
    .sort((a, b) => Math.abs(b.z) - Math.abs(a.z))
    .slice(0, MAX_TELLS)
    .map((d) => {
      const label = faceLabel(d.name)
      return evidence(`tell:${d.name}`, `${label[0].toUpperCase()}${label.slice(1)} ${d.z > 0 ? 'up' : 'down'} vs resting`, 0)
    })
}

// Reads the expression-change score the browser measured against the player's own baseline.
// Tell evidence has log-odds 0: it feeds the opponent profile but never changes the score.
function faceSignal(face) {
  if (!face || face.samples < MIN_SAMPLES || face.trackedRatio < MIN_TRACKED_RATIO) return null
  const logOdds = Math.max(
    MIN_FACE_LOG_ODDS,
    Math.min(MAX_FACE_LOG_ODDS, (face.meanScore - NEUTRAL_SCORE) * LOG_ODDS_PER_POINT),
  )
  let description = 'Face stayed close to their own resting baseline'
  if (face.meanScore >= NEUTRAL_SCORE) {
    const strongest = [...face.deviations].sort((a, b) => Math.abs(b.z) - Math.abs(a.z))[0]
    description = `Face moved away from their own resting baseline${strongest ? `, mostly ${faceLabel(strongest.name)}` : ''}`
  }
  return {
    name: 'live_facial',
    confidence: face.trackedRatio * Math.min(1, face.samples / 6),
    evidence: [evidence('expression_change', description, logOdds), ...tellsOf(face)],
  }
}

// state: { street, action, board, betSize, potSize, activePlayers, position, priorActions, myCards,
//          history: { labeled, bluffs } or absent }
// face:  { samples, trackedRatio, meanScore, deviations: [{ name, z }] } or null.
// Returns { probability, reasons, tells, version }. Reasons are the strongest terms of the sum.
export function scoreBet(state, face) {
  const signals = [statisticsSignal(state), faceSignal(face)].filter(Boolean)
  let total = logit(PRIOR)
  const reasons = []
  const tells = []

  for (const signal of signals) {
    const scale = WEIGHTS[signal.name] * signal.confidence
    for (const item of signal.evidence) {
      const contribution = scale * item.logOdds
      total += contribution
      if (item.code.startsWith('tell:')) {
        tells.push({ code: item.code, description: item.description })
      } else if (Math.abs(contribution) >= MIN_CONTRIBUTION) {
        reasons.push({
          source: signal.name,
          code: item.code,
          description: item.description,
          direction: contribution > 0 ? 'toward_bluff' : 'away_from_bluff',
          contribution,
        })
      }
    }
  }

  reasons.sort(
    (a, b) =>
      Math.abs(b.contribution) - Math.abs(a.contribution) ||
      a.source.localeCompare(b.source) ||
      a.code.localeCompare(b.code),
  )
  return { probability: sigmoid(total), reasons: reasons.slice(0, MAX_REASONS), tells, version: SCORING_VERSION }
}
