"""Campus tasks: find one real, photographed backpack among look-alikes on the Berkeley floors (3–6).

Every bag in the photos is the same black Databricks backpack in one of four configurations, and every
configuration appears at least twice, so appearance alone never identifies the owner's bag. A task is
built backwards from the answer:

  1. pick the owner's placement (one of the 11 real bag photos) → fixes the owner's variant
  2. write a hidden day that puts the owner at that place inside the "had it → noticed it gone" window
  3. plant clues so that exactly one bag survives every elimination:
       variant facts     split across messages (cap? keys?), sometimes with a later correction
       window            last confirmed sighting and the moment the loss was noticed
       floor access      look-alikes on floors the owner's badge can't open → restricted floors (badge log)
       attribution       a coworker claims a look-alike ("that one's mine")
       early sighting    a look-alike was seen in a room before the owner ever got there (compare times)
       never sat down    the owner only passed through a room
       traps             cancelled meetings still on the calendar, meetings moved by notes taped in rooms
                         (chains of up to two notes), and the owner's own wrong guess
  4. verify: re-derive the answer from the hidden facts and reject the task unless exactly one bag survives;
     the shortest correct plan must fit the step cap and the budget must not cover a sweep.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field

from .building import DATA_DIR, Building
from .scenario import Event, Scenario, _fmt, _mins

CAMPUS_FLOORS = (3, 4, 5, 6)
START_NODE = "F4_N01"            # Commons — Northeast Entrance (no bag there)
MAX_OPTIMAL_STEPS = 10           # the harness caps runs at 12 tool calls
AREA = {3: "Meeting Wing", 4: "Commons", 5: "Studios", 6: "Atrium"}

# Places the owner can spend time in (a place may span several photo stops).
PLACES = {
    "Window Walkway":    {"nodes": ["F3_N01"], "kinds": ["coffee_chat", "call"]},
    "Focus Room":        {"nodes": ["F3_N03"], "kinds": ["focus"]},
    "Meeting Room A":    {"nodes": ["F3_N04", "F3_N05"], "kinds": ["meeting", "one_on_one"]},
    "Meeting Room B":    {"nodes": ["F3_N06"], "kinds": ["meeting", "one_on_one"]},
    "Meeting Room C":    {"nodes": ["F3_N07"], "kinds": ["meeting", "one_on_one"]},
    "Commons — Center":  {"nodes": ["F4_N02"], "kinds": ["coffee_chat", "study"]},
    "Kitchen Bar":       {"nodes": ["F4_N03"], "kinds": ["lunch", "coffee_chat"]},
    "Green Lounge":      {"nodes": ["F4_N04"], "kinds": ["coffee_chat", "study"]},
    "Commons NE Corner": {"nodes": ["F4_N05"], "kinds": ["study", "coffee_chat"]},
    "Kitchen Island":    {"nodes": ["F4_N06"], "kinds": ["lunch"]},
    "Studio A":          {"nodes": ["F5_N01"], "kinds": ["section", "meeting"]},
    "Studio B":          {"nodes": ["F5_N02"], "kinds": ["section", "meeting"]},
    "Lime Booths":       {"nodes": ["F6_N01"], "kinds": ["study", "lunch"]},
    "Atrium Overlook":   {"nodes": ["F6_N02"], "kinds": ["study", "coffee_chat"]},
}
NODE_PLACE = {n: p for p, d in PLACES.items() for n in d["nodes"]}
SITS_DOWN = {"meeting", "one_on_one", "focus", "section", "lunch", "study"}
TITLES = {
    "meeting": ["Lab meeting", "Paper reading group", "Project sync", "Eval design review", "Benchmark standup"],
    "one_on_one": ["1:1 with {adv}"],
    "focus": ["Focus block", "Heads-down: thesis draft"],
    "section": ["CS 288 section", "Data 100 office hours", "Vision reading group"],
    "study": ["Study session"],
    "coffee_chat": ["Coffee with {peer}"],
    "call": ["Call with {peer}"],
}
ON_CALENDAR = {"meeting", "one_on_one", "focus", "section", "study", "coffee_chat", "call"}
DURATION = {"meeting": 60, "one_on_one": 30, "focus": 90, "section": 50, "study": 75, "coffee_chat": 20,
            "call": 20, "lunch": 40}
FIRST = ["Ana", "Ben", "Chloe", "Dev", "Esha", "Felix", "Grace", "Hiro", "Ines", "Jae", "Kofi", "Lina", "Mateo",
         "Noor", "Owen", "Priya", "Quinn", "Rafa", "Sana", "Theo"]
LAST = ["Alvarez", "Brooks", "Chen", "Desai", "Eze", "Fischer", "Gupta", "Hassan", "Ito", "Jung", "Kumar", "Lopez"]

VARIANT_FACTS = {"plain": (False, False), "hat": (True, False), "keychain": (False, True), "hat_keychain": (True, True)}
VARIANT_TEXT = {
    "plain": "black Databricks backpack, no cap and no keychain on it",
    "hat": "black Databricks backpack with a black cap strapped on (neon-green ring logo), no keychain",
    "keychain": "black Databricks backpack with a keychain hanging off it (pink tag, keys, brass padlock), no cap",
    "hat_keychain": "black Databricks backpack with a black cap strapped on and a keychain with a pink tag",
}

DIFFICULTY = {
    #           window places, look-alike traps, note chain, cancelled, guess, variant correction, budget slack
    "hard":    dict(window=(3, 4), events=(7, 8), chain=(0, 1), cancelled=1, guess=True, correction=False, slack=1.35),
    "extreme": dict(window=(4, 5), events=(8, 10), chain=(1, 2), cancelled=2, guess=True, correction=True, slack=1.25),
}


@dataclass
class Placement:
    id: str
    variant: str
    spot: str
    views: list[str]
    place: str
    floor: int


def load_placements(include_p1: bool = True) -> list[Placement]:
    raw = json.loads((DATA_DIR / "bag_labels.json").read_text())["placements"]
    out = []
    for p in raw:
        if p["id"] == "P1" and not include_p1:
            continue
        views = list(p["views"])
        node = views[0].rsplit("_v", 1)[0]
        out.append(Placement(p["id"], p["variant"], p["spot"], views, NODE_PLACE[node], int(node[1])))
    return out


def generate_campus(seed: int, difficulty: str = "extreme", building: Building | None = None,
                    max_tries: int = 400) -> Scenario:
    """Deterministic per (seed, difficulty). Retries internally until the task verifies."""
    b = building or Building()
    for attempt in range(max_tries):
        rng = random.Random(f"campus:{difficulty}:{seed}:{attempt}")
        s = _try_generate(rng, seed, difficulty, b)
        if s is not None:
            s.meta["attempt"] = attempt
            return s
    raise RuntimeError(f"could not generate a verifiable campus task for seed {seed}")


def _try_generate(rng: random.Random, seed: int, difficulty: str, b: Building) -> Scenario | None:
    cfg = DIFFICULTY[difficulty]
    placements = load_placements(include_p1=rng.random() < 0.7)
    target = rng.choice(placements)
    variant = target.variant
    lookalikes = [p for p in placements if p.variant == variant and p is not target]

    owner = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    first = owner.split()[0]
    peers = rng.sample([f for f in FIRST if f != first], 6)
    adv, peer = peers[0], peers[1]

    # --- floors: look-alikes on other floors go on the owner's restricted list --------------------------
    other_floor_lookalikes = {p.floor for p in lookalikes if p.floor != target.floor}
    restricted = sorted(other_floor_lookalikes)
    allowed_floors = [f for f in CAMPUS_FLOORS if f not in restricted]
    if target.floor in restricted:
        return None

    # --- hidden day --------------------------------------------------------------------------------------
    n_window = rng.randint(*cfg["window"])
    allowed_places = [p for p, d in PLACES.items() if b.nodes[d["nodes"][0]].floor in allowed_floors]
    if len(allowed_places) < n_window + 2:
        return None
    # Window places: the target's place plus others — prefer places holding same-floor look-alikes (harder).
    same_floor_traps = [p.place for p in lookalikes if p.floor == target.floor and p.place != target.place]
    window_places = [target.place]
    for pl in rng.sample(same_floor_traps, len(same_floor_traps)):
        if len(window_places) < n_window and pl not in window_places:
            window_places.append(pl)
    others = [p for p in allowed_places if p not in window_places]
    rng.shuffle(others)
    while len(window_places) < n_window and others:
        window_places.append(others.pop())
    # Places before the last sighting / after noticing (outside the window).
    n_events = rng.randint(*cfg["events"])
    n_before = rng.randint(1, max(1, n_events - n_window - 1))
    n_after = max(1, n_events - n_window - n_before)
    before = [others.pop() for _ in range(min(n_before, len(others)))]
    after = [others.pop() for _ in range(min(n_after, len(others)))]
    if not before or not after:
        return None

    # Order inside the window: target somewhere after the first window event (bag still "had" at window start).
    rest = [p for p in window_places if p != target.place]
    rng.shuffle(rest)
    k = rng.randint(1, len(rest)) if rest else 0
    window_order = rest[:k] + [target.place] + rest[k:] if rest else [target.place]
    if window_order[0] == target.place and len(window_order) > 1:
        window_order = window_order[1:2] + [target.place] + window_order[2:]
    day_places = before + window_order + after

    events: list[Event] = []
    t = 9 * 60 + rng.choice([0, 10, 20, 30])
    used_titles: set[str] = set()
    for pl in day_places:
        kinds = list(PLACES[pl]["kinds"])
        kind = rng.choice(kinds)
        if pl == target.place and kind not in SITS_DOWN:
            kind = next((k for k in kinds if k in SITS_DOWN), kind)
        if kind not in SITS_DOWN and pl == target.place:
            return None
        title = _title(rng, kind, used_titles, adv, peer)
        dur = DURATION[kind]
        events.append(Event(_fmt(t), _fmt(t + dur), kind, PLACES[pl]["nodes"][0], title))
        t += dur + rng.choice([10, 15, 20, 30, 45])
    if t > 21 * 60:
        return None
    i_last = len(before)                          # last sighting at the end of the first window event
    i_drop = len(before) + window_order.index(target.place)
    i_noticed = len(before) + len(window_order)   # first event after the window
    if i_drop == i_last and i_drop != len(before):
        return None

    chat: list[dict] = []
    notes: dict[str, str] = {}
    elim: dict[str, str] = {}                     # placement id → how it's ruled out

    # --- meetings moved by notes (chains end at a window place, possibly the target) ---------------------
    chain_len = rng.randint(*cfg["chain"])
    chain_nodes: list[str] = []
    moved_event = None
    if chain_len:
        movable = [i for i in range(i_last, i_noticed) if events[i].kind in ("meeting", "section", "one_on_one")]
        if not movable:
            return None
        i_move = rng.choice(movable)
        moved_event = events[i_move]
        real_place = NODE_PLACE[moved_event.node]
        decoy_rooms = [p for p in allowed_places if p not in day_places and _has_note_surface(b, p)
                       and any(k in ("meeting", "section", "one_on_one") for k in PLACES[p]["kinds"])]
        if len(decoy_rooms) < chain_len:
            return None
        hops = rng.sample(decoy_rooms, chain_len)            # calendar says hops[0]; notes lead to real_place
        moved_event.calendar_node = PLACES[hops[0]]["nodes"][0]
        dests = hops[1:] + [real_place]
        for src, dst in zip(hops, dests):
            view = _note_view(b, src, rng)
            more = " — ask at the front if it's full" if dst != real_place else ""
            notes[view] = f"Note taped here: '{moved_event.title} moved to {dst}{more}.'"
            chain_nodes.append(PLACES[src]["nodes"][0])

    # --- cancelled meetings still on the calendar (traps; prefer rooms holding a look-alike) ------------
    cancel_slots = sorted(rng.sample(range(len(events)), min(cfg["cancelled"], len(events))))
    trap_rooms = [p.place for p in lookalikes if p.floor in allowed_floors and p.place not in day_places]
    plain_rooms = [p for p in allowed_places if p not in day_places
                   and any(k in ("meeting", "section") for k in PLACES[p]["kinds"])]
    ghosts = []
    for slot in cancel_slots:
        pool = [p for p in trap_rooms if p not in [NODE_PLACE[g.node] for g in ghosts]] or \
               [p for p in plain_rooms if p not in [NODE_PLACE[g.node] for g in ghosts]]
        pool = [p for p in pool if p not in [NODE_PLACE[n] for n in chain_nodes]]
        if not pool:
            break
        pl = rng.choice(pool)
        ghost = Event(events[slot].start, events[slot].end, "meeting", PLACES[pl]["nodes"][0],
                      _title(rng, "meeting", used_titles, adv, peer), happened=False)
        ghosts.append(ghost)
        chat.append({"time": _fmt(_mins(ghost.start) - 25), "from": rng.choice(peers[2:]),
                     "text": f"{ghost.title} is cancelled today, sorry for the late notice"})

    # --- eliminate every look-alike ---------------------------------------------------------------------
    window_set = set(window_order)
    owner_presence = {NODE_PLACE[e.node]: e for e in events}
    for p in lookalikes:
        if p.floor in restricted:
            elim[p.id] = "restricted_floor"
        elif p.place not in window_set:
            elim[p.id] = "outside_window"       # owner was never there during the window (or never at all)
        else:
            e = owner_presence[p.place]
            how = rng.choice(["attribution", "early_sighting", "early_sighting"] +
                             (["passed_through"] if e.kind not in SITS_DOWN else []))
            claimer = rng.choice(peers[2:])
            if how == "attribution":
                chat.append({"time": _fmt(_mins(e.end) + rng.choice([5, 25, 50])), "from": claimer,
                             "text": f"if anyone sees a Databricks backpack {p.spot} in {p.place} — that's mine, "
                                     f"leaving it there till tomorrow"})
            elif how == "early_sighting":
                seen = _mins(e.start) - rng.choice([35, 50, 70, 95])
                if seen < 8 * 60:
                    seen = 8 * 60 + 5
                if seen >= _mins(e.start):
                    return None
                chat.append({"time": _fmt(seen), "from": claimer,
                             "text": f"someone's Databricks backpack has been sitting {p.spot} in {p.place} "
                                     f"since I got here — anyone missing one?"})
            else:
                chat.append({"time": e.end, "from": owner,
                             "text": f"only passed through {p.place} for a minute, didn't even put my stuff down"})
            elim[p.id] = how

    # --- variant facts (never stated in one place) --------------------------------------------------------
    cap, keys = VARIANT_FACTS[variant]
    cap_line = ("my cap is still clipped to the bag 🙃" if cap else "I'm wearing my cap, so nothing's clipped to the bag")
    keys_line = ("and my keys are hanging off it (pink tag), so I'm locked out of the lab" if keys
                 else "keys are in my jacket pocket at least")
    t_end = _mins(events[-1].end)
    vf = [{"time": _fmt(t_end + 35), "from": owner, "text": cap_line}]
    if cfg["correction"] and rng.random() < 0.6:
        wrong = ("keys are in my jacket pocket at least" if keys else
                 "pretty sure my keys are on the bag too")
        vf.append({"time": _fmt(t_end + 37), "from": owner, "text": wrong})
        fix = ("wait no — jacket pocket is empty, the keys are on the bag" if keys
               else "nvm, found my keys in my jacket — they're NOT on the bag")
        vf.append({"time": _fmt(t_end + 52), "from": owner, "text": fix})
    else:
        vf.append({"time": _fmt(t_end + 38), "from": owner, "text": keys_line})
    chat += vf

    # --- window clues ---------------------------------------------------------------------------------------
    seen_e, miss_e = events[i_last], events[i_noticed]
    seen_place = NODE_PLACE[seen_e.node]
    chat.append({"time": seen_e.end, "from": owner,
                 "text": rng.choice([f"pulled my laptop out of my backpack in {seen_place}",
                                     f"wrapping up in {seen_place}, bag and all",
                                     f"in {seen_place}, charger's in my backpack if anyone needs it"])})
    chat.append({"time": miss_e.start, "from": owner,
                 "text": f"at {NODE_PLACE[miss_e.node]} and just realized my bag isn't with me — it was definitely "
                         f"with me earlier"})
    lunch = next((e for e in events if e.kind == "lunch"), None)
    if lunch:
        chat.append({"time": lunch.start, "from": owner, "text": f"grabbing lunch at {NODE_PLACE[lunch.node]}"})
    if cfg["guess"]:
        wrong_pool = [p.place for p in lookalikes if elim.get(p.id) in ("outside_window", "restricted_floor")] or \
                     [pl for pl in before + after]
        chat.append({"time": _fmt(t_end + 60), "from": owner,
                     "text": f"maybe {rng.choice(wrong_pool)}?? honestly no idea"})
    chat.sort(key=lambda m: m["time"])

    # --- calendar and badge log -------------------------------------------------------------------------------
    timeline = sorted(events + ghosts, key=lambda e: (_mins(e.start), e.happened))
    pos = {id(e): i for i, e in enumerate(timeline)}             # window indices into the merged timeline
    w_last, w_drop, w_missing = pos[id(events[i_last])], pos[id(events[i_drop])], pos[id(events[i_noticed])]
    calendar = [{"start": e.start, "end": e.end, "title": e.title,
                 "room": NODE_PLACE[e.calendar_node or e.node]}
                for e in timeline if e.kind in ON_CALENDAR]
    badge = {"access": {"allowed": [AREA[f] for f in allowed_floors], "no_access": [AREA[f] for f in restricted]},
             "entries": []}
    area = None
    for e in events:
        a = AREA[b.nodes[e.node].floor]
        if a != area:
            badge["entries"].append({"time": _fmt(_mins(e.start) - 3), "event": f"badge in: {a}"})
            area = a
    badge["entries"].append({"time": _fmt(t_end + 8), "event": "badge out: building exit"})

    # --- verify and plan ---------------------------------------------------------------------------------------
    if difficulty == "extreme":
        # Adversarial: walking nearest-first from the start must reach a look-alike before the owner's bag.
        dist = lambda p: b.route(START_NODE, p.views[0].rsplit("_v", 1)[0])[1]
        if not any(dist(p) < dist(target) for p in lookalikes):
            return None
    survivors = [p for p in [target] + lookalikes if p.id not in elim]
    if [p.id for p in survivors] != [target.id]:
        return None
    needs_badge = bool(restricted)
    plan = _optimal_plan(b, target, chain_nodes, needs_badge)
    if len(plan["actions"]) > MAX_OPTIMAL_STEPS:
        return None
    budget = round(plan["cost"] * cfg["slack"] + 1.0, 1)
    sweep = _sweep_cost(b, placements)
    if sweep < 1.8 * budget:
        return None

    others = [p for p in placements if p is not target]
    sid = hashlib.sha1(f"campus:{difficulty}:{seed}".encode()).hexdigest()[:10]
    feasible = sorted({n for pl in window_order for n in PLACES[pl]["nodes"]})
    clue_nodes = sorted(set(chain_nodes) - set(feasible))
    eliminated_nodes = sorted({n for p in lookalikes for n in [v.rsplit("_v", 1)[0] for v in p.views]} - set(feasible))
    ghost_nodes = {g.node for g in ghosts}
    mentioned = {e.node for e in events} | ghost_nodes | set(chain_nodes)
    ruled_out = sorted((mentioned | set(eliminated_nodes)) - set(feasible) - set(clue_nodes))
    s = Scenario(
        id=sid, seed=seed, difficulty=difficulty, budget=budget, start_node=START_NODE,
        employee={"name": owner, "team": rng.choice(["Vision Lab", "NLP Lab", "Systems Lab", "Robotics Lab"]),
                  "advisor": adv, "restricted_floors": [AREA[f] for f in restricted]},
        backpack={"description": VARIANT_TEXT[variant], "view": target.views[0],
                  "node": target.views[0].rsplit("_v", 1)[0], "spot": target.spot,
                  "placement": target.id, "views": target.views, "variant": variant},
        decoys=[{"description": VARIANT_TEXT[p.variant], "view": p.views[0], "node": p.views[0].rsplit("_v", 1)[0],
                 "placement": p.id, "views": p.views, "variant": p.variant,
                 "lookalike": p.variant == variant, "eliminated_by": elim.get(p.id)} for p in others],
        bystanders=[], notes=notes, timeline=timeline,
        clues={"calendar": calendar, "badge_log": badge, "chat": chat},
        window={"last_seen": w_last, "drop": w_drop, "missing": w_missing},
        knowable={"feasible": feasible, "clue": clue_nodes, "ruled_out": ruled_out},
        meta={
            "suite": "campus", "scope_floors": list(CAMPUS_FLOORS), "max_steps": 12,
            "placements_present": [p.id for p in placements],
            "image_overrides": ({"F3_N03_v0": "images/variants/F3_N03_v0__bag_plain.jpg"}
                                if any(p.id == "P1" for p in placements) else {}),
            "variant": variant, "restricted_floors": restricted, "eliminations": elim,
            "chain_nodes": chain_nodes, "ghost_rooms": sorted({NODE_PLACE[g.node] for g in ghosts}),
            "optimal_plan": plan, "sweep_cost": round(sweep, 1),
            "solution": _explain(target, variant, window_order, seen_place, NODE_PLACE[miss_e.node],
                                 lookalikes, elim, restricted, moved_event, chain_nodes),
        },
    )
    s.meta["task_text"] = campus_public_task(s)
    if not verify_campus(s):
        return None
    return s


def campus_public_task(s: Scenario) -> str:
    e = s.employee
    return (f"{e['name']} ({e['team']}) lost their backpack somewhere on the Berkeley floors today. It's the black "
            f"Databricks backpack the whole lab got at the retreat, so several identical ones are lying around. "
            f"Work out which one is theirs and report it. Read the clues carefully: details about their bag, where "
            f"they were, and where their badge works are spread across the calendar, chat, badge log and notes "
            f"taped up in rooms. Only report a bag you have inspected.")


def verify_campus(s: Scenario) -> bool:
    """Independent re-check: exactly one present bag is consistent with every planted fact."""
    present = {p.id: p for p in load_placements(include_p1="P1" in s.meta["placements_present"])}
    variant = s.meta["variant"]
    window_places = {NODE_PLACE[e.node] for i, e in enumerate(s.timeline)
                     if e.happened and s.window["last_seen"] <= i < s.window["missing"]}
    restricted = set(s.meta["restricted_floors"])
    alive = []
    for p in present.values():
        if p.variant != variant or p.floor in restricted or p.place not in window_places:
            continue
        if s.meta["eliminations"].get(p.id) in ("attribution", "early_sighting", "passed_through"):
            continue
        alive.append(p.id)
    if alive != [s.backpack["placement"]]:
        return False
    # Every elimination must be backed by visible text.
    text = json.dumps(s.clues) + json.dumps(s.notes)
    for pid, how in s.meta["eliminations"].items():
        p = present[pid]
        if how in ("attribution", "early_sighting", "passed_through") and p.place not in text:
            return False
        if how == "restricted_floor" and AREA[p.floor] not in json.dumps(s.clues["badge_log"]):
            return False
    # The owner's day never enters a restricted floor.
    return all(int(e.node[1]) not in restricted for e in s.timeline if e.happened)


# --- helpers ----------------------------------------------------------------------------------------------------
def _title(rng, kind, used, adv, peer) -> str:
    opts = [t.format(adv=adv, peer=peer) for t in TITLES.get(kind, ["Lunch"])]
    free = [t for t in opts if t not in used] or [f"{rng.choice(opts)} ({len(used) + 1})"]
    t = rng.choice(free)
    used.add(t)
    return t


def _has_note_surface(b: Building, place: str) -> bool:
    return any(v.clue_surface for n in PLACES[place]["nodes"] for v in b.nodes[n].views)


def _note_view(b: Building, place: str, rng) -> str:
    node = PLACES[place]["nodes"][0]
    views = [v for v in b.nodes[node].views if v.clue_surface] or b.nodes[node].views
    return rng.choice(views).id


def _optimal_plan(b: Building, target: Placement, chain_nodes: list[str], needs_badge: bool) -> dict:
    actions = [{"tool": "read_calendar"}, {"tool": "read_chat"}] + ([{"tool": "read_badge_log"}] if needs_badge else [])
    cost = 0.25 * len(actions)
    here = START_NODE
    target_nodes = [v.rsplit("_v", 1)[0] for v in target.views]
    for n in chain_nodes:
        if n in target_nodes:
            continue
        if n != here:
            cost += b.route(here, n)[1]
            actions.append({"tool": "go_to", "room": n})
            here = n
        cost += 2.0
        actions.append({"tool": "search"})
    t_node = min(target_nodes, key=lambda n: b.route(here, n)[1] if n != here else 0)
    if t_node != here:
        cost += b.route(here, t_node)[1]
        actions.append({"tool": "go_to", "room": t_node})
    cost += 2.0 + 0.5
    actions += [{"tool": "search"}, {"tool": "inspect", "placement": target.id}, {"tool": "report"}]
    return {"actions": actions, "cost": round(cost, 2), "steps": len(actions)}


def _sweep_cost(b: Building, placements: list[Placement]) -> float:
    """Greedy tour that searches every bag's room — what brute force would pay."""
    todo = {p.views[0].rsplit("_v", 1)[0] for p in placements}
    here, cost = START_NODE, 0.0
    while todo:
        nxt = min(todo, key=lambda n: b.route(here, n)[1] if n != here else 0)
        cost += (b.route(here, nxt)[1] if nxt != here else 0) + 2.0 + 0.5
        todo.discard(nxt)
        here = nxt
    return cost


def _explain(target, variant, window_order, seen_place, miss_place, lookalikes, elim, restricted, moved, chain) -> list[str]:
    cap, keys = VARIANT_FACTS[variant]
    steps = [f"Variant: cap {'on' if cap else 'off'}, keys {'on' if keys else 'off'} → '{variant}'.",
             f"Window: from the last sighting in {seen_place} until the loss was noticed in {miss_place}; "
             f"places in between: {', '.join(window_order)}."]
    if moved is not None:
        steps.append(f"'{moved.title}' is on the calendar elsewhere; {len(chain)} taped note(s) lead to "
                     f"{NODE_PLACE[moved.node]}.")
    if restricted:
        steps.append(f"Badge has no access to {', '.join(AREA[f] for f in restricted)} → look-alikes there are not theirs.")
    for p in lookalikes:
        steps.append(f"{p.id} ({p.place}, {p.variant}) eliminated: {elim.get(p.id)}.")
    steps.append(f"Only survivor: {target.id} — {target.spot} in {target.place}.")
    return steps
