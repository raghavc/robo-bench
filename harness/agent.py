"""Model agent loop with compact state.

Each turn the model gets the system prompt plus ONE user message holding a JSON state block: the task,
budget, location, the clues it has read, what it found in rooms it searched, bags it inspected, and its
own recent notes. The transcript is not replayed, so cost grows with what the agent has learned, not
with turns². Images (vlm mode) are attached only on the turn right after the search that produced them.
"""
from __future__ import annotations

import base64
import io
import json
import time

from env import TOOL_SCHEMAS, OfficeSearchEnv

from .cli_clients import TaintedRunError

SYSTEM = """You are a search agent in an office building. Someone lost their backpack today.
Choose actions to work out where it most likely is and find it. Every action spends budget
(walking costs more between distant rooms and floors; searching a room costs 2; inspecting costs 0.5).
If the budget runs out you fail. You may report exactly once, and only an object you have inspected
(in vlm mode: a view you have searched). Reporting the wrong bag is worse than giving up.
Choose exactly one action per turn. Put a short reason (max 25 words) in the `note` argument."""

MAX_TURNS = 12
NOTE_HISTORY = 8


def tool_specs() -> list[dict]:
    specs = []
    for t in TOOL_SCHEMAS:
        props = {k: {"type": "string", "description": v} for k, v in t["args"].items()}
        props["note"] = {"type": "string", "description": "Short reason for this action (max 25 words)."}
        specs.append({"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": {"type": "object", "properties": props, "required": ["note"]}}})
    return specs


class StateTracker:
    """Builds the compact state the model sees. Only contains things the agent has observed."""

    def __init__(self, env: OfficeSearchEnv, first_obs: dict):
        self.env = env
        self.task = first_obs["task"]
        self.rooms = [[r["room"], r["name"], r["floor"]] for r in first_obs["rooms"]]
        self.clues: dict = {}
        self.searched: dict = {}
        self.inspected: dict = {}
        self.notes: list[str] = []
        self.last: dict = {"action": None, "result": first_obs["location"]}
        self.pending_images: list[str] = []

    def update(self, action: dict, obs: dict):
        tool = action.get("tool")
        if action.get("note"):
            self.notes = (self.notes + [f"{tool}: {action['note']}"])[-NOTE_HISTORY:]
        if "calendar" in obs:
            self.clues["calendar"] = obs["calendar"]
        if "chat" in obs:
            self.clues["chat"] = obs["chat"]
        if "badge_log" in obs:
            self.clues["badge_log"] = obs["badge_log"]
        if tool == "search" and "views" in obs:
            self.searched[self.env.node] = {"name": obs["room"], "views": obs["views"]}
            self.pending_images = [obs["image"]] if obs.get("image") else []
        if tool == "zoom" and obs.get("image"):
            self.pending_images = [obs["image"]]
        if tool == "inspect" and "object" in obs:
            self.inspected[obs["object"]] = obs.get("details") or (
                f"{obs.get('class')} (see attached photo)" if obs.get("image") else obs.get("class"))
            if obs.get("image"):
                self.pending_images = [obs["image"]]
        short = {k: v for k, v in obs.items() if k not in ("calendar", "chat", "badge_log", "views")}
        self.last = {"action": {k: v for k, v in action.items() if k != "note"}, "result": short}

    def message(self) -> dict:
        state = {
            "task": self.task, "perception": self.env.perception,
            "budget_left": round(self.env.s.budget - self.env.spent, 2),
            "location": {"room": self.env.node, "name": self.env.b.nodes[self.env.node].room,
                         "floor": self.env.b.nodes[self.env.node].floor},
            "rooms": self.rooms, "clues_read": self.clues, "searched": self.searched,
            "inspected": self.inspected, "your_recent_notes": self.notes, "last": self.last,
        }
        text = f"<state>{json.dumps(state, separators=(',', ':'))}</state>\nChoose the next action."
        if not self.pending_images:
            return {"role": "user", "content": text}
        parts = [{"type": "text", "text": text}]
        for ref in self.pending_images:
            parts.append({"type": "text", "text": f"image {ref}:"})
            parts.append({"type": "image_url", "image_url": {"url": _data_url(self.env.render_image(ref))}})
        self.pending_images = []
        return {"role": "user", "content": parts}


def _data_url(im) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def run_episode(env: OfficeSearchEnv, obs: dict, client, prices: dict, spend_guard) -> tuple[dict, dict]:
    """Play one episode. Returns (grade info, usage). spend_guard(est_usd) → False aborts."""
    tracker = StateTracker(env, obs)
    tools = tool_specs()
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "turns": 0, "wall_s": 0.0,
             "invalid_calls": 0, "aborted": None, "disqualified": None, "api_equivalent_usd": 0.0}
    info: dict = {}
    t0 = time.time()
    while not env.done and usage["turns"] < MAX_TURNS:
        if not spend_guard(usage["cost_usd"]):
            usage["aborted"] = "spend cap reached"
            break
        msgs = [{"role": "system", "content": SYSTEM}, tracker.message()]
        try:
            c = client.complete(msgs, tools)
        except TaintedRunError as e:
            # The CLI's model reached for its own shell/file tools: the episode doesn't count.
            usage["disqualified"] = str(e)
            break
        usage["turns"] += 1
        usage["api_equivalent_usd"] += getattr(c, "api_equivalent_usd", None) or 0.0
        usage["input_tokens"] += c.input_tokens
        usage["output_tokens"] += c.output_tokens
        cost = c.cost_usd if c.cost_usd is not None else \
            (c.input_tokens * prices.get("input_per_m", 0) + c.output_tokens * prices.get("output_per_m", 0)) / 1e6
        usage["cost_usd"] += cost
        action = {"tool": c.tool, **{k: v for k, v in c.args.items() if isinstance(v, (str, int))}}
        if c.tool is None:
            usage["invalid_calls"] += 1
        r = env.step(action)
        tracker.update(action, r.observation)
        if r.done:
            info = r.info
    if not env.done:
        info = env.step({"tool": "give_up", "note": "turn limit or spend cap"}).info
    if usage["disqualified"]:
        info = {**info, "outcome": "disqualified", "success": False, "reward": 0.0, "efficiency": 0.0}
    usage["wall_s"] = round(time.time() - t0, 2)
    usage["cost_usd"] = round(usage["cost_usd"], 6)
    return info, usage
