"""Gym-style environment: an agent searches the building for a lost backpack using tools.

Tools (one per step), passed to `step` as dicts:
    {"tool": "read_calendar"} | {"tool": "read_badge_log"} | {"tool": "read_chat"}
    {"tool": "go_to", "room": "F2_N05"}      walk there (cost = route length; floors cost extra)
    {"tool": "search"}                       look around the current room: 4 views, objects, notes
    {"tool": "inspect", "object": "o3f1"}     close look at an object found by `search` in this room
    {"tool": "report", "object": "o3f1"}      final answer (must be an object you inspected)
    {"tool": "report", "view": "F2_N05_v0"}   final answer in `vlm` perception mode (must be a view you searched)
    {"tool": "give_up"}

Anti-reward-hacking rules enforced here (see tests/test_verifier.py for the red-team suite):
  * One report per episode. Reports must cite an object the agent itself inspected (or, in vlm mode,
    a view it searched). Object ids are random per episode, so they cannot be guessed or memorised.
  * Observations never contain the seed, the answer, spot hints, or file names (images are opaque refs).
  * Every invalid call costs budget, so spamming the API is never free.
  * Budget is hard: running out ends the episode with zero reward.
  * Reward combines outcome, efficiency and a process score computed against what the clues justify,
    so a lucky guess or a brute-force sweep scores below genuine reasoning.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .building import DATA_DIR, Building
from .scenario import Scenario, generate

COST = {"read_calendar": 0.25, "read_badge_log": 0.25, "read_chat": 0.25,
        "search": 2.0, "inspect": 0.5, "zoom": 0.5, "report": 0.0, "give_up": 0.0}
INVALID_COST = 0.5
ANCHORS_PATH = DATA_DIR / "annotations" / "anchors.json"
VARIANTS_DIR = DATA_DIR / "images" / "variants"   # <view>__target.jpg / <view>__decoy.jpg
PERCEPTION_MODES = ("oracle", "vlm", "rfdetr")
IMAGE_MAX_SIDE = 768            # full views and zoom crops
COLLAGE_TILE = (288, 384)       # per view in the 2x2 search collage → 576x768 total

TOOL_SCHEMAS = [
    {"name": "read_calendar", "description": "Read the owner's calendar for today.", "args": {}},
    {"name": "read_badge_log", "description": "Read the owner's building badge log for today.", "args": {}},
    {"name": "read_chat", "description": "Read today's team chat messages involving the owner.", "args": {}},
    {"name": "go_to", "description": "Walk to a room (by room id). Cost = route length; changing floors costs extra.",
     "args": {"room": "room id, e.g. F2_N05"}},
    {"name": "search", "description": "Look around the current room (4 views). Lists objects and any notes. "
     "In photo modes you get ONE 2x2 collage: v0 top-left, v1 top-right, v2 bottom-left, v3 bottom-right.", "args": {}},
    {"name": "zoom", "description": "Photo modes only: a close-up of part of one view in the current (searched) room.",
     "args": {"view": "view id, e.g. F2_N05_v0", "region": "x0,y0,x1,y1 as fractions 0-1 of that view, e.g. 0.4,0.5,0.8,0.9"}},
    {"name": "inspect", "description": "Look closely at an object found by search in the current room.",
     "args": {"object": "object id from search"}},
    {"name": "report", "description": "Final answer. Ends the episode. Report the object id of the owner's bag "
     "(vlm mode: the view id where it is).", "args": {"object": "object id", "view": "view id (vlm mode only)"}},
    {"name": "give_up", "description": "End the episode without an answer.", "args": {}},
]


@dataclass
class StepResult:
    observation: dict
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class OfficeSearchEnv:
    def __init__(self, building: Building | None = None, perception: str = "oracle"):
        assert perception in PERCEPTION_MODES, perception
        self.b = building or Building()
        self.perception = perception
        anchors = json.loads(ANCHORS_PATH.read_text())["views"] if ANCHORS_PATH.exists() else {}
        # Non-bag scene objects per view; bags come from the scenario (bystanders/decoys/target).
        self.scene_objects = {v: [d["class"] for d in dets if d["class"] != "backpack"] for v, dets in anchors.items()}

    # --- episode lifecycle ------------------------------------------------------------------
    def reset(self, seed: int = 0, difficulty: str = "medium", scenario: Scenario | None = None,
              nonce: str = "") -> dict:
        if scenario is None and difficulty.startswith("campus"):
            from .campus import generate_campus
            scenario = generate_campus(seed, difficulty.split("-", 1)[1], self.b)
            scenario.difficulty = difficulty
        self.s = scenario or generate(seed, difficulty, self.b)
        self.secret = hashlib.sha256(f"{self.s.id}:{nonce}:{random.random()}".encode()).hexdigest()
        self.rng = random.Random(self.secret)
        self.node = self.s.start_node
        self.spent = 0.0
        self.done = False
        self.reported = None
        self.searched: set[str] = set()       # rooms searched
        self.seen_views: set[str] = set()
        self.objects: dict[str, dict] = {}    # object id → {"view", "class", "bag": role|None, "room"}
        self.inspected: set[str] = set()
        self.image_refs: dict[str, tuple] = {}
        self.trajectory: list[dict] = [{"t": 0, "tool": "start", "node": self.node, "path": [self.node],
                                        "cost": 0.0, "spent": 0.0}]
        self._populate_objects()
        return {"task": self.s.public_task(), "budget": self.s.budget, "location": self._here(),
                "rooms": self.room_directory(), "perception": self.perception}

    def room_directory(self) -> list[dict]:
        """Public map: every room id, name and floor (the floor plan is not a secret)."""
        return [{"room": n.id, "name": n.room, "floor": n.floor, "type": n.room_type}
                for n in self.b.nodes.values() if self._in_scope(n.id)]

    def _in_scope(self, node_id: str) -> bool:
        scope = self.s.meta.get("scope_floors")
        return not scope or self.b.nodes[node_id].floor in scope

    def step(self, action: dict) -> StepResult:
        if self.done:
            raise RuntimeError("Episode is over; call reset().")
        tool = action.get("tool")
        before = self.node
        handler = getattr(self, f"_t_{tool}", None) if isinstance(tool, str) else None
        if handler is None:
            obs, cost, extra = {"error": f"unknown tool {tool!r}"}, INVALID_COST, {}
        else:
            obs, cost, extra = handler(action)
        self.spent = round(self.spent + cost, 4)
        self.trajectory.append({"t": len(self.trajectory), "tool": tool,
                                "args": {k: v for k, v in action.items() if k not in ("tool", "note")},
                                "note": action.get("note"), "node_before": before, "node": self.node,
                                "cost": cost, "spent": self.spent, "error": obs.get("error"),
                                "summary": _summarize(tool, obs), **extra})
        reward, info = 0.0, {}
        if tool in ("report", "give_up") and "error" not in obs:
            self.done = True
        elif self.spent >= self.s.budget:
            self.done = True
            obs = {**obs, "notice": "budget exhausted — episode over"}
        elif self.s.meta.get("max_steps") and len(self.trajectory) - 1 >= self.s.meta["max_steps"]:
            self.done = True
            obs = {**obs, "notice": "step limit reached — episode over"}
        if self.done:
            reward, info = self.grade()
        else:
            obs = {**obs, "budget_left": round(self.s.budget - self.spent, 2)}
            if self.s.meta.get("max_steps"):
                obs["steps_left"] = self.s.meta["max_steps"] - (len(self.trajectory) - 1)
        return StepResult(obs, reward, self.done, info)

    # --- tools --------------------------------------------------------------------------------
    def _t_read_calendar(self, a):
        return {"calendar": self.s.clues["calendar"]}, COST["read_calendar"], {}

    def _t_read_badge_log(self, a):
        return {"badge_log": self.s.clues["badge_log"]}, COST["read_badge_log"], {}

    def _t_read_chat(self, a):
        return {"chat": self.s.clues["chat"]}, COST["read_chat"], {}

    def _t_go_to(self, a):
        dest = a.get("room")
        if dest not in self.b.nodes or not self._in_scope(dest):
            return {"error": f"unknown room {dest!r}"}, INVALID_COST, {}
        if dest == self.node:
            return {"error": "already here"}, INVALID_COST, {}
        path, cost = self.b.route(self.node, dest)
        self.node = dest
        return {"location": self._here(), "route": [self.b.nodes[p].room for p in path]}, cost, {"path": path}

    def _t_search(self, a):
        views = []
        for v in self.b.nodes[self.node].views:
            self.seen_views.add(v.id)
            entry = {"view": v.id, "description": v.description}
            if self.perception != "vlm":
                entry["objects"] = [{"id": oid, "class": o["class"]} for oid, o in self.objects.items()
                                    if v.id in o["views"]]
            if v.id in self.s.notes:
                entry["note"] = self.s.notes[v.id]
            views.append(entry)
        repeat = self.node in self.searched
        self.searched.add(self.node)
        out = {"room": self.b.nodes[self.node].room, "views": views}
        if self.perception != "oracle":
            # One 2x2 collage of the 4 views instead of 4 images: ~1/4 the image tokens; zoom for detail.
            out["image"] = self._ref(("collage", self.node))
            out["layout"] = "2x2 collage: v0 top-left, v1 top-right, v2 bottom-left, v3 bottom-right"
        return out, COST["search"], {"repeat": repeat}

    def _t_zoom(self, a):
        if self.perception == "oracle":
            return {"error": "zoom needs a photo perception mode"}, INVALID_COST, {}
        view = a.get("view")
        if view not in self.b.views or self.b.views[view].node_id != self.node or self.node not in self.searched:
            return {"error": "zoom on a view of the room you are in, after searching it"}, INVALID_COST, {}
        region = _parse_region(a.get("region"))
        if region is None:
            return {"error": "region must be x0,y0,x1,y1 with 0<=x0<x1<=1, 0<=y0<y1<=1, at least 0.05 wide"}, INVALID_COST, {}
        return {"view": view, "region": list(region), "image": self._ref(("crop", view, region))}, COST["zoom"], {}

    def _t_inspect(self, a):
        oid = a.get("object")
        o = self.objects.get(oid)
        if o is None or self.perception == "vlm":
            return {"error": "unknown object id"}, INVALID_COST, {}
        if self.node not in o["rooms"] or self.node not in self.searched:
            return {"error": "object is not in view — search the room it is in first"}, INVALID_COST, {}
        self.inspected.add(oid)
        out = {"object": oid, "class": o["class"]}
        if o["bag"]:
            if self.perception == "rfdetr":
                # Real pipeline: the model must read the bag's details from the photo itself.
                out["image"] = self._ref(("view", o["view"]))
            else:
                out["details"] = o["description"]
        return out, COST["inspect"], {"inspected_bag": bool(o["bag"])}

    def _t_report(self, a):
        if self.reported is not None:
            return {"error": "already reported"}, INVALID_COST, {}
        if self.perception == "vlm":
            view = a.get("view")
            if view not in self.seen_views:
                return {"error": "you can only report a view you have searched"}, INVALID_COST, {}
            self.reported = {"view": view}
        else:
            oid = a.get("object")
            if oid not in self.inspected:
                return {"error": "you can only report an object you have inspected"}, INVALID_COST, {}
            self.reported = {"object": oid, "view": self.objects[oid]["view"]}
        return {"reported": a.get("object") or a.get("view")}, COST["report"], {}

    def _t_give_up(self, a):
        return {"gave_up": True}, COST["give_up"], {}

    # --- world helpers -----------------------------------------------------------------------
    def _populate_objects(self):
        """Assign random per-episode ids to every object; bags carry hidden roles."""
        items = []
        for view, classes in self.scene_objects.items():
            items += [{"view": view, "class": c, "bag": None} for c in classes]
        bags = [("target", self.s.backpack)] + \
               [("decoy" if d.get("lookalike", True) else "bystander", d) for d in self.s.decoys] + \
               [("bystander", d) for d in self.s.bystanders]
        for role, bag in bags:
            # A placement may be visible from several views (and stops); it is still ONE object.
            items.append({"view": bag["view"], "views": bag.get("views", [bag["view"]]), "class": "backpack",
                          "bag": role, "description": bag["description"], "placement": bag.get("placement")})
        self.rng.shuffle(items)
        for it in items:
            oid = "o" + format(self.rng.getrandbits(24), "06x")
            while oid in self.objects:            # ids must never collide (a collision would drop an object)
                oid = "o" + format(self.rng.getrandbits(24), "06x")
            it.setdefault("views", [it["view"]])
            it["rooms"] = {self.b.views[v].node_id for v in it["views"]}
            it["room"] = self.b.views[it["view"]].node_id
            self.objects[oid] = it

    def _here(self) -> dict:
        n = self.b.nodes[self.node]
        return {"room": n.id, "name": n.room, "floor": n.floor,
                "searched": n.id in self.searched}

    def _ref(self, spec: tuple) -> str:
        """Opaque per-episode image handle — never a file name (file names would leak target/decoy)."""
        ref = "img_" + hashlib.sha256(f"{self.secret}:{spec}".encode()).hexdigest()[:16]
        self.image_refs[ref] = spec
        return ref

    def render_image(self, ref: str) -> Image.Image:
        """Harness-side only: build the image an opaque ref points to."""
        spec = self.image_refs[ref]
        if spec[0] == "view":
            im = _open(self._image_for(spec[1]))
            im.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
            return im
        if spec[0] == "collage":
            tw, th = COLLAGE_TILE
            canvas = Image.new("RGB", (tw * 2, th * 2), "white")
            draw = ImageDraw.Draw(canvas)
            for i, v in enumerate(self.b.nodes[spec[1]].views[:4]):
                tile = ImageOps.fit(_open(self._image_for(v.id)), (tw, th))
                x, y = (i % 2) * tw, (i // 2) * th
                canvas.paste(tile, (x, y))
                draw.rectangle([x, y, x + 26, y + 16], fill="black")
                draw.text((x + 4, y + 2), f"v{i}", fill="white")
            return canvas
        _, view, (x0, y0, x1, y1) = spec
        im = _open(self._image_for(view))
        w, h = im.size
        crop = im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
        crop.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
        return crop

    def _image_for(self, view_id: str) -> Path:
        override = self.s.meta.get("image_overrides", {}).get(view_id)
        if override:
            return DATA_DIR / override
        for role, item in [("target", self.s.backpack)] + [("decoy", d) for d in self.s.decoys]:
            if item["view"] == view_id:
                p = VARIANTS_DIR / f"{view_id}__{role}.jpg"
                if p.exists():
                    return p
        return DATA_DIR / self.b.views[view_id].file

    def vision_ready(self) -> bool:
        """True when every placed bag has a real photo (required for vlm/rfdetr runs)."""
        if self.s.meta.get("suite") == "campus":
            return True          # every campus bag IS a real photo
        need = [("target", self.s.backpack)] + [("decoy", d) for d in self.s.decoys]
        return all((VARIANTS_DIR / f"{it['view']}__{role}.jpg").exists() for role, it in need)

    # --- verifier ------------------------------------------------------------------------------
    def grade(self) -> tuple[float, dict]:
        s, b = self.s, self.b
        target_view = s.backpack["view"]
        rep = self.reported
        if rep is None:
            outcome = "gave_up" if self.trajectory[-1]["tool"] == "give_up" else \
                "budget_exhausted" if self.spent >= s.budget else "turn_limit"
            success = False
        elif "object" in rep:
            role = self.objects[rep["object"]]["bag"]
            success = role == "target"
            outcome = {"target": "success", "decoy": "decoy", "bystander": "bystander_bag"}.get(role, "not_a_bag")
        else:
            target_views = set(s.backpack.get("views", [target_view]))
            lookalike_views = {v for d in s.decoys if d.get("lookalike", True) for v in d.get("views", [d["view"]])}
            other_bag_views = {v for d in s.decoys if not d.get("lookalike", True) for v in d.get("views", [d["view"]])}
            success = rep["view"] in target_views
            outcome = "success" if success else "decoy" if rep["view"] in lookalike_views else \
                "bystander_bag" if rep["view"] in other_bag_views else \
                "right_room_wrong_view" if rep["view"].rsplit("_v", 1)[0] == s.backpack["node"] else "wrong_location"

        steps, process = self._process_labels()
        rubric = self._campus_rubric() if s.meta.get("suite") == "campus" else None
        if rubric is not None:
            scored = [r for r in rubric if r["applies"]]
            process = sum(r["passed"] for r in scored) / len(scored) if scored else 0.0
            reference = s.meta["optimal_plan"]["cost"]
        else:
            # Reference cost for an agent that reasons perfectly but has no luck beyond the clues.
            reference = 0.5 + b.route(s.start_node, s.backpack["node"])[1] + COST["search"] + COST["inspect"]
        efficiency = min(1.0, reference / max(self.spent, 1e-9)) if success else 0.0
        if success:
            reward = 0.5 + 0.25 * efficiency + 0.25 * process
        elif outcome in ("decoy", "bystander_bag", "not_a_bag", "wrong_location", "right_room_wrong_view"):
            reward = -0.25          # confidently wrong is worse than giving up
        else:
            reward = 0.0
        floors = sorted({b.nodes[n].floor for t in self.trajectory for n in t.get("path", [t["node"]])})
        info = {"outcome": outcome, "success": success, "reward": round(reward, 4),
                "efficiency": round(efficiency, 3), "process": round(process, 3),
                "budget": s.budget, "spent": round(self.spent, 2), "reference_cost": round(reference, 2),
                "steps": len(self.trajectory) - 1, "rooms_searched": len(self.searched),
                "floors_visited": floors, "step_labels": steps,
                "target_view": target_view, "reported": rep,
                # RL-facing signals: binary outcome by default; shaped reward is opt-in.
                "reward_outcome": 1.0 if success else 0.0, "reward_shaped": round(reward, 4)}
        if rubric is not None:
            info["rubric"] = rubric
        return round(reward, 4), info

    def _campus_rubric(self) -> list[dict]:
        """Binary checklist, each item checkable from the trajectory + hidden facts."""
        s, tr = self.s, self.trajectory[1:]
        meta = s.meta
        t_of = lambda pred: next((st["t"] for st in tr if pred(st)), None)
        first_move = t_of(lambda st: st["tool"] == "go_to" and not st.get("error"))
        report_t = t_of(lambda st: st["tool"] == "report" and not st.get("error"))
        read_t = {tool: t_of(lambda st, tool=tool: st["tool"] == tool and not st.get("error"))
                  for tool in ("read_calendar", "read_chat", "read_badge_log")}
        before = lambda t, limit: t is not None and (limit is None or t < limit)
        searched_at = {}
        for st in tr:
            if st["tool"] == "search" and not st.get("error"):
                searched_at.setdefault(st["node"], st["t"])
        target_ids = {oid for oid, o in self.objects.items() if o["bag"] == "target"}
        target_views = set(s.backpack.get("views", []))
        verified = any((st["tool"] == "inspect" and st.get("args", {}).get("object") in target_ids) or
                       (st["tool"] == "zoom" and st.get("args", {}).get("view") in target_views)
                       for st in tr if not st.get("error") and (report_t is None or st["t"] < report_t))
        wrong_inspects = sum(1 for st in tr if st["tool"] == "inspect" and not st.get("error")
                             and self.objects.get(st["args"].get("object"), {}).get("bag") not in (None, "target"))
        chain = meta.get("chain_nodes", [])
        chain_ok = all(n in searched_at for n in chain) and \
            [searched_at[n] for n in chain] == sorted(searched_at[n] for n in chain) if chain else True
        traps = set(s.knowable["ruled_out"]) - set(chain)
        trap_hits = sorted(n for n in searched_at if n in traps)
        return [
            {"id": "clues_first", "applies": True, "passed": before(read_t["read_calendar"], first_move)
             and before(read_t["read_chat"], first_move),
             "why": "read the calendar and chat before walking anywhere"},
            {"id": "badge_access", "applies": bool(meta.get("restricted_floors")),
             "passed": before(read_t["read_badge_log"], report_t),
             "why": "checked which floors the owner's badge opens before deciding"},
            {"id": "followed_notes", "applies": bool(chain), "passed": chain_ok,
             "why": "searched every room in the taped-note chain, in order"},
            {"id": "avoided_traps", "applies": True, "passed": not trap_hits,
             "why": "never searched a room the clues ruled out" + (f" (searched {', '.join(trap_hits)})" if trap_hits else "")},
            {"id": "verified_before_report", "applies": report_t is not None, "passed": verified,
             "why": "looked closely at the owner's bag (inspect / zoom) before reporting"},
            {"id": "no_inspect_spam", "applies": True, "passed": wrong_inspects <= 2,
             "why": f"inspected {wrong_inspects} other bag(s) (≤ 2 allowed)"},
        ]

    def _process_labels(self) -> tuple[list[dict], float]:
        """Label each step good / bad / neutral against what the clues justified."""
        k = self.s.knowable
        feasible, clue, ruled_out = set(k["feasible"]), set(k["clue"]), set(k["ruled_out"])
        labels, read, visited, searched = [], set(), set(), set()
        for st in self.trajectory[1:]:
            tool, lab, why = st["tool"], "neutral", ""
            if st.get("error"):
                lab, why = "bad", "invalid call"
            elif tool in ("read_calendar", "read_chat", "read_badge_log"):
                lab, why = ("good", "gathered clues") if tool not in read else ("neutral", "re-read")
                read.add(tool)
            elif tool == "go_to":
                dest = st["node"]
                if not read & {"read_calendar", "read_chat"}:
                    lab, why = "bad", "moved before reading any clues"
                elif dest in visited:
                    lab, why = "bad", "revisited a room"
                elif dest in feasible:
                    lab, why = "good", "room justified by clues"
                elif dest in clue:
                    lab, why = "good", "went to read an on-site clue"
                elif dest in ruled_out:
                    lab, why = "bad", "room ruled out by clues"
                else:
                    lab, why = "bad", "room never mentioned in clues"
                visited.add(dest)
            elif tool == "search":
                if st["node"] in searched:
                    lab, why = "bad", "searched the same room twice"
                elif st["node"] == self.s.start_node and st["node"] not in feasible:
                    lab, why = "neutral", "searched the start"
                else:
                    lab, why = "neutral", ""
                searched.add(st["node"])
            elif tool == "inspect":
                lab, why = ("neutral", "") if st.get("inspected_bag") else ("bad", "inspected a non-bag")
            labels.append({"t": st["t"], "label": lab, "why": why})
        good = sum(l["label"] == "good" for l in labels)
        bad = sum(l["label"] == "bad" for l in labels)
        return labels, (good / (good + bad) if good + bad else 0.0)


def _summarize(tool, obs: dict) -> str:
    """One-line, answer-free summary of an observation for logs and replays."""
    if obs.get("error"):
        return f"error: {obs['error']}"
    if tool == "read_calendar":
        return f"{len(obs['calendar'])} calendar entries"
    if tool == "read_chat":
        return f"{len(obs['chat'])} chat messages"
    if tool == "read_badge_log":
        return f"{len(obs['badge_log'])} badge events"
    if tool == "go_to":
        return "walked: " + " → ".join(obs["route"])
    if tool == "search":
        bags = sum(o["class"] == "backpack" for v in obs["views"] for o in v.get("objects", []))
        notes = [v["note"] for v in obs["views"] if v.get("note")]
        return f"searched {obs['room']}: {bags} bag(s)" + (f"; note: {notes[0]}" if notes else "")
    if tool == "inspect":
        return f"{obs['object']}: {obs.get('details') or obs['class']}"
    if tool == "zoom":
        return f"zoomed into {obs['view']} at {obs['region']}"
    if tool == "report":
        return f"reported {obs['reported']}"
    return tool or ""


def _open(path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def _parse_region(r):
    """Accept [x0,y0,x1,y1] or "x0,y0,x1,y1" (fractions of the view)."""
    try:
        vals = [float(v) for v in (r.split(",") if isinstance(r, str) else r)]
    except (TypeError, ValueError):
        return None
    if len(vals) != 4:
        return None
    x0, y0, x1, y1 = vals
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1) or x1 - x0 < 0.05 or y1 - y0 < 0.05:
        return None
    return (round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3))
