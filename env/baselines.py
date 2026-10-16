"""Scripted agents.

Reference agents bracket model scores:
    RandomAgent      wanders, reports the first bag it inspects
    ExhaustiveAgent  sweeps every room in order and checks every bag (no reasoning)
    ClueAgent        reads clues, drops cancelled meetings, follows room changes (chat and on-site notes),
                     searches only between the last sighting and when the loss was noticed

Exploit agents try to game the verifier (tests/test_verifier.py asserts they all fail):
    BlindReporter, UninspectedReporter, FirstBagReporter, GuessAndGo, InspectSpammer
"""
from __future__ import annotations

import random
import re

from .office_env import OfficeSearchEnv


class Agent:
    name = "agent"

    def run(self, env: OfficeSearchEnv, obs: dict) -> dict:
        self.env, self.obs, self.final = env, obs, {}
        m = re.search(r"It is a (.+?)\. Find", obs["task"])
        self.target = m.group(1) if m else None      # campus tasks never state the bag's details
        self.rooms = {r["name"].lower(): r["room"] for r in obs["rooms"]}
        self.last_search = {}
        self.search_episode()
        if not env.done:
            self.act({"tool": "give_up"})
        return self.final

    def act(self, action: dict):
        if self.env.done:
            return None
        r = self.env.step(action)
        if r.done:
            self.final = r.info
        return r

    def visit_and_check(self, room: str) -> bool:
        """Go to a room, search it, inspect bags; report a match. Returns True when the episode ended."""
        if self.env.node != room and self.act({"tool": "go_to", "room": room}) is None:
            return True
        if self.env.done:
            return True
        r = self.act({"tool": "search"})
        if r is None or self.env.done:
            return True
        self.last_search = r.observation
        for view in r.observation.get("views", []):
            for o in view.get("objects", []):
                if o["class"] != "backpack":
                    continue
                ins = self.act({"tool": "inspect", "object": o["id"]})
                if ins is None or self.env.done:
                    return True
                if self.accept(ins.observation):
                    self.act({"tool": "report", "object": o["id"]})
                    return True
        return False

    def accept(self, inspected: dict) -> bool:
        return inspected.get("details") == self.target


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def accept(self, inspected):
        return True

    def search_episode(self):
        rooms = [r["room"] for r in self.obs["rooms"]]
        while not self.env.done:
            if self.visit_and_check(self.rng.choice([r for r in rooms if r != self.env.node])):
                return


class ExhaustiveAgent(Agent):
    name = "exhaustive"

    def search_episode(self):
        for r in self.obs["rooms"]:
            if self.visit_and_check(r["room"]):
                return


class ClueAgent(Agent):
    name = "clue_reasoner"

    def search_episode(self):
        cal = self.act({"tool": "read_calendar"}).observation["calendar"]
        chat = self.act({"tool": "read_chat"}).observation["chat"]
        cancelled = {m.group(1) for c in chat if (m := re.search(r"heads up, (.+) is cancelled", c["text"]))}
        moved = {m.group(1): m.group(2).lower() for c in chat
                 if (m := re.search(r"moving (.+) to (.+), the other", c["text"]))}
        events = [(e["start"], e["end"], moved.get(e["title"], e["room"].lower())) for e in cal
                  if e["title"] not in cancelled]
        events += [(c["time"], _plus(c["time"], 45), m.group(1)) for c in chat
                   if (m := re.search(r"lunch in the (.+?),", c["text"]))]
        events.sort()
        # Window by time: from the last sighting (message sent when that event ended) until they noticed.
        seen_t = next((c["time"] for c in chat
                       if any(k in c["text"] for k in ("laptop out", "bag and all", "charger's in"))), "00:00")
        miss_t = next((c["time"] for c in chat if "realized my bag" in c["text"]), "99:99")
        rooms = [r for start, end, r in events if end >= seen_t and start < miss_t]
        queue = [self.rooms[r] for r in rooms if r in self.rooms]
        done: set[str] = set()
        while queue:
            # Nearest candidate first: avoids bouncing between floors.
            queue = [q for q in dict.fromkeys(queue) if q not in done]
            if not queue:
                break
            room = min(queue, key=lambda r: self.env.b.path_cost(self.env.node, r))
            queue.remove(room)
            done.add(room)
            if self.visit_and_check(room):
                return
            # Follow on-site notes ("X moved to <room> — ...").
            for v in self.last_search.get("views", []):
                if m := re.search(r'moved to "(.+?)"', v.get("note", "")):
                    nxt = self.rooms.get(m.group(1).lower())
                    if nxt and nxt not in done:
                        queue.insert(0, nxt)
        self.act({"tool": "give_up"})


def _plus(hhmm: str, minutes: int) -> str:
    h, m = map(int, hhmm.split(":"))
    t = h * 60 + m + minutes
    return f"{t // 60:02d}:{t % 60:02d}"


# --- exploit agents (red team) ---------------------------------------------------------------------
class BlindReporter(Agent):
    """Reports made-up object ids and view ids without looking."""
    name = "exploit_blind"

    def search_episode(self):
        for guess in ("o00000", "o12345", self.env.s.start_node + "_v0"):
            self.act({"tool": "report", "object": guess, "view": guess})


class UninspectedReporter(Agent):
    """Searches, then reports a bag it saw but never inspected."""
    name = "exploit_uninspected"

    def search_episode(self):
        for r in self.obs["rooms"]:
            if self.env.node != r["room"] and self.act({"tool": "go_to", "room": r["room"]}) is None:
                return
            res = self.act({"tool": "search"})
            if res is None:
                return
            bags = [o["id"] for v in res.observation.get("views", []) for o in v.get("objects", [])
                    if o["class"] == "backpack"]
            if bags and self.act({"tool": "report", "object": bags[0]}) is None:
                return


class FirstBagReporter(ClueAgent):
    """Follows the clues, then reports the first bag it inspects without checking the details."""
    name = "exploit_first_bag"

    def accept(self, inspected):
        return True


class GuessAndGo(Agent):
    """Trusts the owner's last-minute guess (a trap on hard) and skips the calendar."""
    name = "exploit_trust_guess"

    def search_episode(self):
        chat = self.act({"tool": "read_chat"}).observation["chat"]
        guess = next((m.group(1) for c in chat if (m := re.search(r"maybe (?:the )?(.+?)\?", c["text"]))), None)
        g = (guess or "").lower()
        hit = [r["room"] for r in self.obs["rooms"] if g and (r["name"].lower() == g or r["name"].lower().startswith(g))]
        order = hit[:1] + [r["room"] for r in self.obs["rooms"]]
        for room in dict.fromkeys(order):
            if self.visit_and_check(room):
                return


class InspectSpammer(Agent):
    """Inspects every object in every room, hoping the budget is loose."""
    name = "exploit_inspect_all"

    def search_episode(self):
        for r in self.obs["rooms"]:
            if self.env.node != r["room"] and self.act({"tool": "go_to", "room": r["room"]}) is None:
                return
            res = self.act({"tool": "search"})
            if res is None:
                return
            for v in res.observation.get("views", []):
                for o in v.get("objects", []):
                    ins = self.act({"tool": "inspect", "object": o["id"]})
                    if ins is None:
                        return
                    if ins.observation.get("details") == self.target:
                        self.act({"tool": "report", "object": o["id"]})
                        return


REFERENCE_AGENTS = {"random": RandomAgent, "exhaustive": ExhaustiveAgent, "clue_reasoner": ClueAgent}
EXPLOIT_AGENTS = {a.name: a for a in (BlindReporter, UninspectedReporter, FirstBagReporter, GuessAndGo, InspectSpammer)}


# --- campus agents -----------------------------------------------------------------------------------------------
def _owner_variant(chat: list[dict], owner: str) -> tuple[bool | None, bool | None]:
    """Cap/keys facts from the owner's own messages; later messages override earlier ones."""
    cap = keys = None
    for m in sorted(chat, key=lambda m: m["time"]):
        if m["from"] != owner:
            continue
        t = m["text"]
        if "cap" in t:
            cap = "clipped to the bag" in t and "nothing's clipped" not in t
        if "keys" in t:
            keys = ("on the bag" in t or "hanging off it" in t) and "NOT on the bag" not in t
    return cap, keys


def _matches(details: str, cap, keys) -> bool:
    has_cap, has_keys = "cap strapped on" in details, "keychain" in details and "no keychain" not in details
    return (cap is None or has_cap == cap) and (keys is None or has_keys == keys)


class CampusOraclePlan(Agent):
    """Privileged ceiling: executes the generator's verified optimal plan. Proves every task is solvable
    within the step cap and budget; not a fair competitor."""
    name = "oracle_plan"

    def search_episode(self):
        target = next(oid for oid, o in self.env.objects.items() if o["bag"] == "target")
        for a in self.env.s.meta["optimal_plan"]["actions"]:
            a = {"tool": "inspect", "object": target} if a["tool"] == "inspect" else \
                {"tool": "report", "object": target} if a["tool"] == "report" else a
            if self.act(a) is None:
                return


class CampusAppearanceAgent(Agent):
    """Learns the bag's variant from chat, then sweeps rooms nearest-first and reports the first bag
    that LOOKS right. Ignores schedules, notes and badge access — the look-alike trap."""
    name = "appearance_only"

    def search_episode(self):
        owner = self.env.s.employee["name"]
        chat = self.act({"tool": "read_chat"}).observation["chat"]
        cap, keys = _owner_variant(chat, owner)
        self.accept = lambda ins: _matches(ins.get("details", ""), cap, keys)
        todo = [r["room"] for r in self.obs["rooms"]]
        while todo and not self.env.done:
            nxt = min(todo, key=lambda r: self.env.b.path_cost(self.env.node, r) if r != self.env.node else 0)
            todo.remove(nxt)
            if self.visit_and_check(nxt):
                return


class CampusRandom(Agent):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def accept(self, inspected):
        return True

    def search_episode(self):
        rooms = [r["room"] for r in self.obs["rooms"]]
        while not self.env.done:
            if self.visit_and_check(self.rng.choice([r for r in rooms if r != self.env.node])):
                return


CAMPUS_AGENTS = {"oracle_plan": CampusOraclePlan, "appearance_only": CampusAppearanceAgent, "random": CampusRandom}
