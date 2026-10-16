"""Procedural scenarios: a hidden day timeline → clues → backpack placement.

The timeline is ground truth and never shown to the agent. Clues are derived from it with
difficulty-dependent noise:
  * cancelled meetings that still sit on the calendar (a chat message cancels them)
  * room changes — announced in chat (medium) or only on a note inside the original room (hard)
  * a misleading guess from the owner (hard)
  * look-alike decoy bags, placed inside rooms the clues point to (hard)
  * real bystander bags already present in the photos

Each scenario also records which rooms the clues *justify* visiting. The verifier grades every
move against that knowable set, not against the answer, so lucky guesses don't score as reasoning.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field

from .building import DATA_DIR, Building

VENUES = {
    "meeting":     ["F1_N03", "F1_N04", "F2_N05", "F2_N06"],
    "one_on_one":  ["F2_N03", "F2_N07", "F1_N06"],
    "coffee_chat": ["F2_N03", "F1_N06", "F2_N02", "F1_N07"],
    "lunch":       ["F1_N07", "F2_N05", "F2_N06", "F1_N08"],
    "focus":       ["F2_N02", "F2_N08", "F1_N08", "F2_N06"],
    "social":      ["F2_N03", "F2_N02", "F1_N08"],
}
TITLES = {
    "meeting": ["Sprint Planning", "Design Review", "Customer Sync", "Q4 Roadmap", "Hiring Debrief", "Incident Retro"],
    "one_on_one": ["1:1 with {mgr}"],
    "coffee_chat": ["Coffee chat with {peer}", "Catch-up with {peer}"],
    "focus": ["Focus block", "Heads-down: eval harness"],
    "social": ["Team happy hour"],
}
ON_CALENDAR = {"meeting", "one_on_one", "coffee_chat", "focus", "social"}  # lunch never is
SITS_DOWN = {"meeting", "one_on_one", "lunch", "focus", "coffee_chat"}     # where a bag gets left

FIRST = ["Maya", "Jordan", "Priya", "Sam", "Alex", "Noah", "Lena", "Omar", "Kai", "Rosa"]
LAST = ["Chen", "Okafor", "Patel", "Rivera", "Kim", "Novak", "Haddad", "Silva"]
COLORS = ["black", "navy", "charcoal grey", "olive green"]
FEATURES = ["a red keychain", "a cat sticker on the front pocket", "white zipper pulls", "a luggage tag",
            "a reflective strip", "a carabiner clip"]

DIFFICULTY = {
    # gap: events between last confirmed sighting and the drop.
    # missing: events after the drop until they notice (None = only at the end of the day).
    # budget: search budget; sized so a full sweep of the building cannot fit (see run_baselines).
    "easy":   dict(events=3, gap=(0, 0), missing=None, cancelled=0, moves=0, note_moves=False,
                   decoys=0, decoys_in_window=False, guess=False, budget=15.0),
    "medium": dict(events=5, gap=(1, 2), missing=(1, 1), cancelled=1, moves=1, note_moves=False,
                   decoys=1, decoys_in_window=False, guess=False, budget=22.0),
    "hard":   dict(events=7, gap=(2, 3), missing=(1, 2), cancelled=1, moves=1, note_moves=True,
                   decoys=2, decoys_in_window=True, guess=True, budget=30.0),
}


@dataclass
class Event:
    start: str
    end: str
    kind: str
    node: str                            # where it actually happened
    title: str | None
    calendar_node: str | None = None     # room the calendar claims (set on a room change)
    happened: bool = True                # False for cancelled meetings (calendar-only)
    move_via: str | None = None          # "chat" | "note"


@dataclass
class Scenario:
    id: str                              # opaque; safe to show
    seed: int                            # hidden
    difficulty: str
    budget: float
    start_node: str
    employee: dict
    backpack: dict                       # {"description", "view", "node", "spot"}  (hidden location)
    decoys: list[dict]                   # [{"description", "view", "node"}]
    bystanders: list[dict]               # real bags already in the photos: [{"description", "view"}]
    notes: dict                          # view id → note text visible when that room is searched
    timeline: list[Event]                # hidden ground truth
    clues: dict                          # calendar / badge_log / chat, returned by tools
    window: dict                         # hidden: last_seen, drop, missing event indices
    knowable: dict                       # hidden: feasible / clue / ruled_out node sets
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def public_task(self) -> str:
        if self.meta.get("task_text"):
            return self.meta["task_text"]
        e = self.employee
        return (f"{e['name']} ({e['team']}) left their backpack somewhere in the building today. "
                f"It is a {self.backpack['description']}. Find it and report it. "
                f"Other people's bags may look similar — only report a bag you have inspected.")


def _fmt(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _mins(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _unique_title(rng, kind, timeline, mgr, peer) -> str:
    """Titles must be unique in a day, or a cancellation notice would be ambiguous."""
    taken = {e.title for e in timeline}
    options = [t.format(mgr=mgr, peer=peer) for t in TITLES[kind]]
    free = [t for t in options if t not in taken]
    if free:
        return rng.choice(free)
    base = rng.choice(options)
    return f"{base} (follow-up {sum(1 for t in taken if t and t.startswith(base)) + 1})"


def _bag(rng, color=None, keep=(), avoid=None) -> tuple[str, frozenset]:
    """Random bag description; `keep` features are forced, and it never equals `avoid`."""
    while True:
        c = color or rng.choice(COLORS)
        feats = list(keep) + rng.sample([f for f in FEATURES if f not in keep], 2 - len(keep))
        key = frozenset([c, *feats])
        if key != avoid:
            return f"{c} backpack with {feats[0]} and {feats[1]}", key


def _bystander_views() -> list[str]:
    path = DATA_DIR / "annotations" / "anchors.json"
    if not path.exists():
        return []
    views = json.loads(path.read_text())["views"]
    return [v for v, dets in views.items() for d in dets if d["class"] == "backpack"]


def generate(seed: int, difficulty: str = "medium", building: Building | None = None) -> Scenario:
    b = building or Building()
    cfg = DIFFICULTY[difficulty]
    rng = random.Random(f"{difficulty}:{seed}")

    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    mgr, peer = rng.sample([f for f in FIRST if f not in name], 2)
    color = rng.choice(COLORS)
    desc, target_key = _bag(rng, color)
    target_feats = [f for f in FEATURES if f in desc]

    # --- hidden timeline ------------------------------------------------------------------
    n = cfg["events"]
    pool = ["meeting", "coffee_chat", "focus", "one_on_one", "meeting", "coffee_chat"]
    rng.shuffle(pool)
    social = n >= 5 and rng.random() < 0.6
    rest = pool[:n - 1 - social]
    kinds = rest[:len(rest) // 2 + 1] + ["lunch"] + rest[len(rest) // 2 + 1:] + (["social"] if social else [])
    used: set[str] = set()
    t = 9 * 60 + rng.choice([0, 15, 30])
    timeline: list[Event] = []
    for k in kinds:
        node = rng.choice([v for v in VENUES[k] if v not in used] or VENUES[k])
        used.add(node)
        dur = {"lunch": 45, "focus": 90, "meeting": 60, "social": 60}.get(k, 30)
        title = _unique_title(rng, k, timeline, mgr, peer) if k in TITLES else None
        timeline.append(Event(_fmt(t), _fmt(t + dur), k, node, title))
        t += dur + rng.choice([15, 30, 45])

    drop = rng.choice([i for i, e in enumerate(timeline) if e.kind in SITS_DOWN and i >= cfg["gap"][0]])
    last_seen = max(0, drop - rng.randint(*cfg["gap"]))
    # They notice it's missing somewhere other than where they were since the last sighting.
    window_nodes = {e.node for e in timeline[last_seen:drop + 1]}
    after = [i for i in range(drop + 1, len(timeline)) if timeline[i].node not in window_nodes]
    missing = after[min(len(after), rng.randint(*cfg["missing"])) - 1] if cfg["missing"] and after else None

    # Room changes: pick meetings inside the search window so the change actually matters.
    chat: list[dict] = []
    notes: dict[str, str] = {}
    window_end = missing if missing is not None else len(timeline)
    for _ in range(cfg["moves"]):
        cands = [i for i in range(last_seen, window_end) if timeline[i].kind == "meeting" and not timeline[i].calendar_node]
        if not cands:
            break
        e = timeline[rng.choice(cands)]
        spare = [v for v in VENUES["meeting"] if v != e.node and v not in used] or \
                [v for v in sorted(b.nodes) if v not in used and b.nodes[v].room_type != "hallway"]
        e.calendar_node = rng.choice(spare)
        used.add(e.calendar_node)
        if cfg["note_moves"]:
            e.move_via = "note"
            v = next((v for v in b.nodes[e.calendar_node].views if v.clue_surface), b.nodes[e.calendar_node].views[0])
            notes[v.id] = f"Note taped here: '{e.title} moved to \"{b.nodes[e.node].room}\" (this room is booked)'"
        else:
            e.move_via = "chat"
            chat.append({"time": _fmt(_mins(e.start) - 10), "from": mgr,
                         "text": f"moving {e.title} to {b.nodes[e.node].room}, the other room is taken"})

    # Cancelled meetings: still on the calendar, in a plausible room nobody used.
    for _ in range(cfg["cancelled"]):
        slot = rng.randrange(last_seen, window_end)
        venue = rng.choice([v for v in VENUES["meeting"] if v not in used] or
                           [v for v in sorted(b.nodes) if v not in used and b.nodes[v].room_type != "hallway"])
        used.add(venue)
        ghost = Event(timeline[slot].start, timeline[slot].end, "meeting", venue,
                      _unique_title(rng, "meeting", timeline, mgr, peer), happened=False)
        timeline.insert(slot, ghost)
        drop += slot <= drop
        last_seen += slot <= last_seen
        missing = missing + (slot <= missing) if missing is not None else None
        chat.append({"time": _fmt(_mins(ghost.start) - 20), "from": peer,
                     "text": f"heads up, {ghost.title} is cancelled today"})

    window_end = missing if missing is not None else len(timeline)
    drop_view = rng.choice(b.spot_views(timeline[drop].node))

    # --- clues ----------------------------------------------------------------------------
    calendar = [{"start": e.start, "end": e.end, "title": e.title,
                 "room": b.nodes[e.calendar_node or e.node].room,
                 "floor": b.nodes[e.calendar_node or e.node].floor}
                for e in timeline if e.kind in ON_CALENDAR]
    badge, floor = [], None
    for e in timeline:
        if e.happened and b.nodes[e.node].floor != floor:
            floor = b.nodes[e.node].floor
            badge.append({"time": _fmt(_mins(e.start) - 5), "event": f"badge in, floor {floor}"})
    badge.append({"time": _fmt(_mins(timeline[-1].end) + 10), "event": "badge out, building exit"})

    seen = timeline[last_seen]
    chat.append({"time": seen.end, "from": name, "text": _seen_line(seen, b, rng)})
    lunch = next((e for e in timeline if e.kind == "lunch"), None)
    if lunch:
        chat.append({"time": lunch.start, "from": name,
                     "text": f"grabbing lunch in the {b.nodes[lunch.node].room.lower()}, anyone want to join?"})
    if missing is not None:
        e = timeline[missing]
        chat.append({"time": e.start, "from": name,
                     "text": f"at the {b.nodes[e.node].room.lower()} and realized my bag isn't with me, "
                             f"must have left it somewhere earlier"})
    chat.append({"time": _fmt(_mins(timeline[-1].end) + 40), "from": name,
                 "text": "just got to the train and my backpack isn't with me 😩 can someone help find it?"})
    if cfg["guess"]:
        outside = [e for i, e in enumerate(timeline) if e.happened and not (last_seen <= i < window_end)]
        wrong = rng.choice(outside or [e for e in timeline if e.happened and e is not timeline[drop]])
        chat.append({"time": _fmt(_mins(timeline[-1].end) + 42), "from": name,
                     "text": f"maybe the {b.nodes[wrong.node].room.lower()}? honestly not sure"})
    chat.sort(key=lambda m: m["time"])

    # --- what the clues justify (hidden; used for process grading) --------------------------
    in_window = [e for i, e in enumerate(timeline) if last_seen <= i < window_end and e.happened]
    feasible = sorted({e.node for e in in_window})
    clue_nodes = sorted({e.calendar_node for e in in_window if e.move_via == "note"} - set(feasible))
    mentioned = {e.node for e in timeline} | {e.calendar_node for e in timeline if e.calendar_node}
    ruled_out = sorted(mentioned - set(feasible) - set(clue_nodes))

    # --- bags ---------------------------------------------------------------------------------
    taken = {drop_view.id}
    decoys = []
    for i in range(cfg["decoys"]):
        pool_nodes = feasible if cfg["decoys_in_window"] else [e.node for e in timeline if e.happened]
        views = [v for nd in pool_nodes for v in b.spot_views(nd) if v.id not in taken] or \
                [v for v in b.views.values() if v.candidate_spots and v.id not in taken]
        v = rng.choice(views)
        taken.add(v.id)
        # Same colour and one shared feature: only a close inspection tells them apart.
        d, _ = _bag(rng, color, keep=[target_feats[i % 2]], avoid=target_key)
        decoys.append({"description": d, "view": v.id, "node": v.node_id})
    bystanders = [{"description": _bag(rng, avoid=target_key)[0], "view": v} for v in _bystander_views()]

    sid = hashlib.sha1(f"office-search:{difficulty}:{seed}".encode()).hexdigest()[:10]
    return Scenario(
        id=sid, seed=seed, difficulty=difficulty, budget=cfg["budget"], start_node="F1_N01",
        employee={"name": name, "team": rng.choice(["Platform", "Research", "Sales", "Design"]),
                  "manager": mgr, "habits": _habits(rng)},
        backpack={"description": desc, "view": drop_view.id, "node": drop_view.node_id,
                  "spot": rng.choice(drop_view.candidate_spots)},
        decoys=decoys, bystanders=bystanders, notes=notes, timeline=timeline,
        clues={"calendar": calendar, "badge_log": badge, "chat": chat},
        window={"last_seen": last_seen, "drop": drop, "missing": missing},
        knowable={"feasible": feasible, "clue": clue_nodes, "ruled_out": ruled_out},
        meta={"window_size": len(feasible)},
    )


def _seen_line(e: Event, b: Building, rng: random.Random) -> str:
    where = b.nodes[e.node].room.lower()
    return rng.choice([
        f"pulled my laptop out of my backpack in the {where}",
        f"just finished up in the {where}, bag and all",
        f"in the {where}, charger's in my backpack if anyone needs it",
    ])


def _habits(rng: random.Random) -> list[str]:
    return rng.sample([
        "usually hangs their bag on the back of their chair",
        "tends to drop their bag under the table",
        "often works from the lounges in the afternoon",
        "prefers window seats",
    ], 2)
