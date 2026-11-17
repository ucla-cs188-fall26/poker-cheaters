"""Turns public poker hand histories (PHH files) into one JSON row per postflop bet or raise.

Usage, from backend/:
    python training/statistics/phh_to_rows.py <folder or file> \
        artifacts/statistics/public_rows.jsonl

Reads .phh (one hand) and .phhs (many hands) files of no-limit hold'em. Each row holds what
was visible when the bet was made, plus the bettor's hole cards when the file reveals them.
The cards are only used later to work out the label, never as a model input.

A bet's size is the total the player bet or raised to on that street, and the pot is the pot
before that bet. api/actions.js records tracker bets the same way.
"""

import json
import sys
import tomllib
from pathlib import Path

STREET_BY_BOARD_SIZE = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}
NO_LIMIT_HOLDEM = "NT"


def split_cards(text: str) -> list[str]:
    return [text[i : i + 2] for i in range(0, len(text), 2)]


def known(cards: list[str]) -> bool:
    return len(cards) == 2 and all("?" not in card for card in cards)


def rows_from_hand(hand: dict, hand_id: str) -> list[dict]:
    """Replays one hand and returns its postflop bets by players whose cards are known."""
    if hand.get("variant") != NO_LIMIT_HOLDEM:
        return []
    stacks = [float(s) for s in hand["starting_stacks"]]
    count = len(stacks)
    names = hand.get("players") or [None] * count
    antes = [min(float(a), stacks[i]) for i, a in enumerate(hand.get("antes", [0] * count))]
    stacks = [stacks[i] - antes[i] for i in range(count)]
    blinds = hand.get("blinds_or_straddles", [0] * count)
    street_bets = [min(float(b), stacks[i]) for i, b in enumerate(blinds)]
    stacks = [stacks[i] - street_bets[i] for i in range(count)]
    pot = sum(antes) + sum(street_bets)

    folded = [False] * count
    holes: list[list[str]] = [[] for _ in range(count)]
    hole_source = [None] * count
    history: list[list[dict]] = [[] for _ in range(count)]
    board: list[str] = []
    rows = []

    for raw in hand["actions"]:
        parts = raw.split("#")[0].split()
        if len(parts) < 2:
            continue
        actor, verb = parts[0], parts[1]

        if actor == "d":
            if verb == "dh":
                player = int(parts[2][1:]) - 1
                holes[player] = split_cards(parts[3])
                if known(holes[player]):
                    hole_source[player] = "dealt"
            elif verb == "db":
                board = board + split_cards(parts[2])
                street_bets = [0.0] * count
            continue

        player = int(actor[1:]) - 1
        street = STREET_BY_BOARD_SIZE.get(len(board))
        if verb == "f":
            folded[player] = True
        elif verb == "cc":
            owed = min(max(street_bets) - street_bets[player], stacks[player])
            history[player].append({"street": street, "action": "call" if owed > 0 else "check"})
            street_bets[player] += owed
            stacks[player] -= owed
            pot += owed
        elif verb == "cbr":
            total = float(parts[2])
            action = "bet" if max(street_bets) == 0 else "raise"
            if street in ("flop", "turn", "river") and pot > 0:
                rows.append(
                    {
                        "hand": hand_id,
                        "player": names[player],
                        "seat": player,
                        "street": street,
                        "action": action,
                        "board": list(board),
                        "betSize": total,
                        "potSize": pot,
                        "activePlayers": folded.count(False),
                        "priorActions": list(history[player]),
                    }
                )
            history[player].append({"street": street, "action": action})
            paid = total - street_bets[player]
            street_bets[player] = total
            stacks[player] -= paid
            pot += paid
        elif verb == "sm" and len(parts) > 2 and not known(holes[player]):
            shown = split_cards(parts[2])
            if known(shown):
                holes[player] = shown
                hole_source[player] = "shown"

    labelled = []
    for row in rows:
        seat = row.pop("seat")
        if known(holes[seat]):
            labelled.append({**row, "hole": holes[seat], "holeSource": hole_source[seat]})
    return labelled


def hands_in(path: Path):
    """Yields (hand id, hand) for a .phh file or every hand in a .phhs file."""
    with path.open("rb") as file:
        data = tomllib.load(file)
    name = f"{path.parent.name}/{path.name}"  # File names repeat across folders.
    if path.suffix == ".phhs":
        for key, hand in data.items():
            yield f"{name}#{key}", hand
    else:
        yield name, data


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    found = [*source.rglob("*.phh"), *source.rglob("*.phhs")]
    files = [source] if source.is_file() else sorted(found)
    target.parent.mkdir(parents=True, exist_ok=True)
    hands = written = skipped = 0
    with target.open("w") as out:
        for path in files:
            try:
                for hand_id, hand in hands_in(path):
                    hands += 1
                    for row in rows_from_hand(hand, hand_id):
                        out.write(json.dumps(row) + "\n")
                        written += 1
            except (tomllib.TOMLDecodeError, KeyError, ValueError, IndexError) as error:
                skipped += 1
                print(f"Skipped {path}: {error}", file=sys.stderr)
    print(f"{hands} hands read, {written} bets with known cards written, {skipped} files skipped")


if __name__ == "__main__":
    main()
