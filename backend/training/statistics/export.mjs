// Builds the training table for the statistics model: one row per labelled postflop bet, with
// the same features live scoring uses (statisticsFeatures in api/_lib/scoring.js).
//
// Usage, from the repo root:
//   node backend/training/statistics/export.mjs public <rows.jsonl> [out.csv]
//   node --env-file=.env.local backend/training/statistics/export.mjs tracker [out.csv]
//
// "public" reads the rows written by phh_to_rows.py. "tracker" reads labelled bets from Neon
// (select only). Output goes to backend/artifacts/statistics/, which is gitignored: never
// commit it. Tracker ids are hashed and no names are written.
import { createHash } from 'node:crypto'
import { createReadStream, createWriteStream, mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { createInterface } from 'node:readline'
import { fileURLToPath } from 'node:url'
import { wasBluff } from '../../../api/_lib/bluffRule.js'
import { sql } from '../../../api/_lib/db.js'
import { STATISTICS_CODES, scoreBet, statisticsFeatures } from '../../../api/_lib/scoring.js'

const OUT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '../../artifacts/statistics')
const COLUMNS = ['source', 'group', 'player', 'street', 'was_bluff', 'baseline', ...STATISTICS_CODES]

const opaque = (id) => createHash('sha256').update(String(id)).digest('hex').slice(0, 12)
const csvCell = (value) => (/[",\n]/.test(String(value)) ? `"${String(value).replaceAll('"', '""')}"` : String(value))

// One CSV line: ids, the label, what the current weights say, then every feature value.
function line(source, group, player, state, label) {
  const values = Object.fromEntries(statisticsFeatures(state).map((f) => [f.code, f.value]))
  const baseline = scoreBet(state, null).probability
  const cells = [source, group, player ?? '', state.street, label ? 1 : 0, baseline, ...STATISTICS_CODES.map((c) => values[c] ?? 0)]
  return `${cells.map(csvCell).join(',')}\n`
}

function open(path) {
  mkdirSync(dirname(path), { recursive: true })
  const out = createWriteStream(path)
  out.write(`${COLUMNS.join(',')}\n`)
  return out
}

async function exportPublic(rowsPath, outPath) {
  const out = open(outPath)
  let written = 0
  for await (const text of createInterface({ input: createReadStream(rowsPath) })) {
    if (!text.trim()) continue
    const row = JSON.parse(text)
    const label = wasBluff(row.hole, row.board, row.street)
    if (label === null) continue
    const state = { ...row, position: null, myCards: [] }
    out.write(line('public', row.hand, row.player, state, label))
    written += 1
  }
  out.end()
  return written
}

async function exportTracker(outPath) {
  const rows = await sql`
    select a.hand_id, a.session_id, a.opponent_id, a.street, a.action, a.bet_size, a.pot_size,
           a.active_players, a.position, a.board, a.probability, a.was_bluff, h.my_cards
    from tracker_actions a join tracker_hands h on h.id = a.hand_id
    order by a.created_at
  `
  const out = open(outPath)
  const earlier = new Map() // "hand:opponent" -> that opponent's actions so far in the hand
  let written = 0
  for (const row of rows) {
    const key = `${row.hand_id}:${row.opponent_id}`
    const priorActions = earlier.get(key) ?? []
    if (row.probability !== null && row.was_bluff !== null && row.street !== 'preflop') {
      const state = {
        street: row.street,
        action: row.action,
        board: row.board,
        betSize: Number(row.bet_size),
        potSize: Number(row.pot_size),
        activePlayers: row.active_players ?? 2,
        position: row.position,
        priorActions,
        myCards: row.my_cards ?? [],
      }
      out.write(line('tracker', opaque(row.session_id), opaque(row.opponent_id), state, row.was_bluff))
      written += 1
    }
    earlier.set(key, [...priorActions, { street: row.street, action: row.action }])
  }
  out.end()
  return written
}

const [mode, first, second] = process.argv.slice(2)
if (mode === 'public' && first) {
  const outPath = resolve(second ?? `${OUT_DIR}/public.csv`)
  console.log(`${await exportPublic(resolve(first), outPath)} labelled bets written to ${outPath}`)
} else if (mode === 'tracker') {
  if (!process.env.DATABASE_URL) throw new Error('DATABASE_URL is not set. Run with: node --env-file=.env.local ...')
  const outPath = resolve(first ?? `${OUT_DIR}/tracker.csv`)
  console.log(`${await exportTracker(outPath)} labelled bets written to ${outPath}`)
} else {
  console.error('Usage: export.mjs public <rows.jsonl> [out.csv]  |  export.mjs tracker [out.csv]')
  process.exit(1)
}
