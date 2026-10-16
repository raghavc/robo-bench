"""Model clients. Everything speaks the OpenAI-compatible chat-completions format with tool calls,
so one router key (e.g. OpenRouter) covers GPT, Claude, Gemini and Kimi. `MockClient` runs the full
pipeline offline with simulated token usage, for testing and cost estimates."""
from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass

import requests


@dataclass
class Completion:
    tool: str | None
    args: dict
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None      # provider-reported, when available
    latency_s: float


class OpenAICompatibleClient:
    def __init__(self, model: str, base_url: str, api_key_env: str, max_output_tokens: int = 2048,
                 extra: dict | None = None):
        self.model, self.base_url, self.max_output_tokens = model, base_url.rstrip("/"), max_output_tokens
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(f"Missing {api_key_env} — add it to .env (see .env.example)")
        self.extra = extra or {}

    def complete(self, messages: list[dict], tools: list[dict]) -> Completion:
        body = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "required",
                "max_tokens": self.max_output_tokens, "usage": {"include": True}, **self.extra}
        t0 = time.time()
        for attempt in range(5):
            r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=180,
                              headers={"Authorization": f"Bearer {self.api_key}"})
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            break
        else:
            if r.status_code == 429:     # e.g. the daily cap on free models: stop cleanly, resume later
                from .cli_clients import UsageLimitError
                raise UsageLimitError(f"{self.model}: {r.text[:300]}")
            r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        tool, args = None, {}
        if calls:
            tool = calls[0]["function"]["name"]
            try:
                args = json.loads(calls[0]["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                tool, args = None, {}
        u = data.get("usage", {})
        return Completion(tool, args, msg.get("content") or "", u.get("prompt_tokens", 0),
                          u.get("completion_tokens", 0), u.get("cost"), time.time() - t0)


class RoboflowWorkflowClient:
    """A Roboflow Workflows VLM block (e.g. Qwen 3.8 VL 27B running natively on Roboflow) as the model.
    The block has no tool calling, so it answers with one JSON action in text (same format as the CLI
    adapters). It also needs an image every call: text-only turns send a small blank placeholder.
    Billed in Roboflow credits; the block reports no token counts, so tokens are estimated (≈4 chars/token)."""

    URL = "https://serverless.roboflow.com/workflows/run"

    def __init__(self, cfg: dict):
        self.api_key = os.environ.get(cfg.get("api_key_env", "ROBOFLOW_API_KEY"))
        if not self.api_key:
            raise RuntimeError("Missing ROBOFLOW_API_KEY — add it to .env (see .env.example)")
        step = {"type": cfg["block"], "name": "vlm", "images": "$inputs.image", "task_type": "unconstrained",
                "prompt": "$inputs.prompt", "max_tokens": cfg.get("max_output_tokens", 4096)}
        if cfg.get("backend") == "native":
            step.update(backend="native", model_version=cfg["model"],
                        enable_thinking=cfg.get("enable_thinking", False))
        else:
            step.update(backend="openrouter", openrouter_model_version=cfg["model"])
        self.spec = {"version": "1.0",
                     "inputs": [{"type": "WorkflowImage", "name": "image"},
                                {"type": "WorkflowParameter", "name": "prompt"}],
                     "steps": [step],
                     "outputs": [{"type": "JsonField", "name": "out", "selector": "$steps.vlm.*"}]}

    def complete(self, messages: list[dict], tools: list[dict]) -> Completion:
        from .cli_clients import _flatten, _parse_action
        system, user, images = _flatten(messages)
        if images:
            note = "The attached image is the photo referenced in the state."
            image = base64.b64encode(images[0]).decode()
        else:
            note = "The attached image is a blank placeholder: ignore it."
            image = _BLANK_JPEG
        prompt = f"{system}\n\n{note}\n\n{user}"
        body = {"api_key": self.api_key, "specification": self.spec,
                "inputs": {"image": {"type": "base64", "value": image}, "prompt": prompt}}
        t0 = time.time()
        for attempt in range(3):
            try:
                r = requests.post(self.URL, json=body, timeout=120)
            except requests.RequestException as e:
                if attempt == 2:
                    raise RuntimeError(f"Roboflow request failed: {e}") from e
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            if r.status_code != 200:
                raise RuntimeError(f"Roboflow {r.status_code}: {r.text[:300]}")
            break
        out = r.json()["outputs"][0]["out"]
        text = out.get("output") or ""
        tool, args = _parse_action(text)
        tin = out.get("input_tokens") or len(prompt) // 4 + (800 if images else 0)
        tout = out.get("output_tokens") or (len(text) + len(out.get("thinking") or "")) // 4
        return Completion(tool, args, text, tin, tout, 0.0, time.time() - t0)


def _blank_jpeg() -> str:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


_BLANK_JPEG = _blank_jpeg()


class MockClient:
    """Deterministic stand-in for a model: reads the JSON state block and plays a reasonable policy.
    Token counts are estimated from message size (≈4 chars/token; images ≈ 800 tokens)."""

    def __init__(self, *_, **__):
        pass

    def complete(self, messages: list[dict], tools: list[dict]) -> Completion:
        user = messages[-1]["content"]
        text_parts = [p["text"] for p in user if p["type"] == "text"] if isinstance(user, list) else [user]
        n_images = sum(p["type"] == "image_url" for p in user) if isinstance(user, list) else 0
        state = json.loads(re.search(r"<state>(.*)</state>", text_parts[0], re.S).group(1))
        tool, args = self._policy(state)
        chars = sum(len(json.dumps(m)) for m in messages[:-1]) + sum(len(t) for t in text_parts)
        return Completion(tool, {**args, "note": f"mock: {tool}"}, "", chars // 4 + 800 * n_images, 120, None, 0.0)

    def _policy(self, st: dict):
        """Calendar-following stand-in. Not a real model — it only exercises the pipeline end to end."""
        clues = st["clues_read"]
        if "calendar" not in clues:
            return "read_calendar", {}
        if "chat" not in clues:
            return "read_chat", {}
        here = st["location"]["room"]
        searched = st["searched"]
        m = re.search(r"It is a (.+?)\. Find", st["task"])
        want = m.group(1) if m else None
        owner_cap = owner_keys = None
        for c in sorted(clues["chat"], key=lambda c: c["time"]):
            t = c["text"]
            if "cap" in t:
                owner_cap = "clipped to the bag" in t and "nothing's clipped" not in t
            if "keys" in t:
                owner_keys = ("on the bag" in t or "hanging off it" in t) and "NOT on the bag" not in t

        def ok(d):
            if want:
                return d == want
            cap, keys = "cap strapped on" in d, "keychain" in d and "no keychain" not in d
            return (owner_cap is None or cap == owner_cap) and (owner_keys is None or keys == owner_keys)

        if here in searched:
            for view in searched[here]["views"]:
                for o in view.get("objects", []):
                    if o["class"] == "backpack":
                        d = st["inspected"].get(o["id"])
                        if d is None:
                            return "inspect", {"object": o["id"]}
                        if ok(d):
                            return "report", {"object": o["id"]}
        elif st["location"]["room"] not in ("F1_N01", "F4_N01") or searched:
            return "search", {}
        rooms = [(r[0], r[1].lower()) for r in st["rooms"]]
        cancelled = {m.group(1) for c in clues["chat"] if (m := re.search(r"(?:heads up, )?(.+?) is cancelled", c["text"]))}
        for e in clues["calendar"]:
            if e["title"] in cancelled:
                continue
            for rid, name in rooms:
                if (name == e["room"].lower() or name.startswith(e["room"].lower())) and rid not in searched:
                    return "go_to", {"room": rid}
        for rid, _ in rooms:
            if rid not in searched and rid != here:
                return "go_to", {"room": rid}
        return "give_up", {}


def make_client(cfg: dict):
    if cfg.get("provider") == "mock":
        return MockClient()
    if cfg.get("provider") == "roboflow_workflow":
        return RoboflowWorkflowClient(cfg)
    if cfg.get("provider", "").endswith("_cli"):
        from .cli_clients import CLI_CLIENTS
        return CLI_CLIENTS[cfg["provider"]](cfg)
    return OpenAICompatibleClient(cfg["model"], cfg.get("base_url", "https://openrouter.ai/api/v1"),
                                  cfg.get("api_key_env", "OPENROUTER_API_KEY"),
                                  cfg.get("max_output_tokens", 2048), cfg.get("extra"))
