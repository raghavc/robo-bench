"""Subscription adapters: drive Claude Code, Codex CLI and Gemini CLI as plain model proxies.

The eval loop stays in harness/agent.py: every turn, the CLI gets the same system prompt + state as the
API models and must answer with ONE JSON action. Nothing persists between turns.

Sandboxing (so a coding agent can't read the answer off disk):
  * every call runs in a fresh empty temp directory, outside the repo
  * Claude Code: all built-in tools disabled (--tools ""), our system prompt replaces its default
  * Codex / Gemini: their tools cannot be fully disabled, so we watch their event logs; any attempt to
    run a command or touch a file marks the turn TAINTED and the episode is disqualified by the runner
Usage limits raise UsageLimitError so the runner can stop cleanly and resume later.
"""
from __future__ import annotations

import base64
import json
import pathlib
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .clients import Completion

TIMEOUT_S = 300
LIMIT_PATTERNS = re.compile(r"usage limit|rate limit|quota|too many requests|429|limit reached|resets? (at|in)", re.I)
ACTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["tool", "args", "note"],
    "properties": {
        "tool": {"type": "string", "enum": ["read_calendar", "read_badge_log", "read_chat", "go_to", "search",
                                            "inspect", "zoom", "report", "give_up"]},
        "args": {"type": "object", "additionalProperties": False,
                 "required": ["room", "object", "view", "region"],
                 "properties": {"room": {"type": "string"}, "object": {"type": "string"}, "view": {"type": "string"},
                                "region": {"type": "string"}}},
        "note": {"type": "string"},
    },
}
ANSWER_FORMAT = """
You are being called through a command-line tool, but this is NOT a coding task: do not run commands,
read files or use any tools of your own. Reply with exactly one JSON object and nothing else:
{"tool": "<one of: read_calendar, read_badge_log, read_chat, go_to, search, inspect, zoom, report, give_up>",
 "args": {"room": "<room id or empty>", "object": "<object id or empty>", "view": "<view id or empty>",
          "region": "<x0,y0,x1,y1 for zoom, or empty>"},
 "note": "<short reason, max 25 words>"}"""


class UsageLimitError(RuntimeError):
    pass


class TaintedRunError(RuntimeError):
    """The CLI's model tried to use its own tools (shell / file access)."""


def _flatten(messages: list[dict]) -> tuple[str, str, list[bytes]]:
    """System prompt, user text, and any attached images (JPEG bytes, in order)."""
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = messages[-1]["content"]
    images: list[bytes] = []
    if isinstance(user, list):
        texts = []
        for p in user:
            if p["type"] == "text":
                texts.append(p["text"])
            elif p["type"] == "image_url":
                images.append(base64.b64decode(p["image_url"]["url"].split(",", 1)[1]))
                texts.append(f"[image {len(images)} attached]")
        user = "\n".join(texts)
    return system + "\n" + ANSWER_FORMAT, user, images


def _parse_action(text: str) -> tuple[str | None, dict]:
    """Pull the first JSON object with a "tool" key out of the model's reply."""
    for m in re.finditer(r"\{.*\}", text, re.S):
        chunk = m.group(0)
        for end in range(len(chunk), 0, -1):          # tolerate trailing prose after the object
            if chunk[end - 1] != "}":
                continue
            try:
                obj = json.loads(chunk[:end])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "tool" in obj:
                args = {k: v for k, v in (obj.get("args") or {}).items() if v not in ("", None)}
                return obj["tool"], {**args, "note": obj.get("note", "")}
    return None, {}


class _CLIClient:
    binary = ""
    supports_images = False

    def __init__(self, cfg: dict):
        if not shutil.which(self.binary):
            raise RuntimeError(f"`{self.binary}` not found on PATH — install it and log in first")
        self.cfg = cfg

    def _run(self, cmd: list[str], cwd: Path, stdin: str | None = None) -> subprocess.CompletedProcess:
        try:
            p = subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True, text=True, timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"{self.binary} timed out after {TIMEOUT_S}s") from e
        blob = (p.stdout or "") + (p.stderr or "")
        if p.returncode != 0 and LIMIT_PATTERNS.search(blob):
            raise UsageLimitError(f"{self.binary}: {blob.strip()[-300:]}")
        if p.returncode != 0:
            raise RuntimeError(f"{self.binary} exited {p.returncode}: {blob.strip()[-500:]}")
        return p

    def complete(self, messages: list[dict], tools: list[dict]) -> Completion:
        system, user, images = _flatten(messages)
        if images and not self.supports_images:
            raise NotImplementedError(f"{self.binary} adapter is text-only: use --perception oracle")
        sandbox = Path(tempfile.mkdtemp(prefix="lostfound-"))
        t0 = time.time()
        try:
            text, tin, tout, extra_cost = self._call(system, user, sandbox, images)
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)
        tool, args = _parse_action(text)
        # Billed to a subscription, but reported at public list price so models are comparable on cost.
        cost = extra_cost if extra_cost is not None else self._list_price(tin, tout, getattr(self, "_cached", 0))
        c = Completion(tool, args, text, tin, tout, cost, time.time() - t0)
        c.api_equivalent_usd = cost
        return c

    def _list_price(self, tin: int, tout: int, cached: int = 0) -> float:
        cin = self.cfg.get("input_per_m") or 0.0
        cc = self.cfg.get("cached_input_per_m", cin) or 0.0
        cout = self.cfg.get("output_per_m") or 0.0
        return ((tin - cached) * cin + cached * cc + tout * cout) / 1e6


class ClaudeCodeClient(_CLIClient):
    """`claude -p` with every built-in tool disabled and our system prompt replacing its default."""
    binary = "claude"
    supports_images = True

    def _call(self, system, user, sandbox, images=()):
        cmd = ["claude", "-p", "--tools", "", "--system-prompt", system,
               "--no-session-persistence", "--strict-mcp-config", "--setting-sources", ""]
        if self.cfg.get("model"):
            cmd += ["--model", self.cfg["model"]]
        if self.cfg.get("effort"):
            cmd += ["--effort", self.cfg["effort"]]
        if images:
            # Images go in as content blocks of one stream-json user message.
            content = [{"type": "text", "text": user}] + [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": base64.b64encode(im).decode()}} for im in images]
            msg = {"type": "user", "message": {"role": "user", "content": content}}
            p = self._run(cmd + ["--input-format", "stream-json", "--output-format", "stream-json", "--verbose"],
                          sandbox, stdin=json.dumps(msg) + "\n")
            out = next((json.loads(l) for l in reversed(p.stdout.splitlines())
                        if l.strip().startswith("{") and json.loads(l).get("type") == "result"), {})
        else:
            p = self._run(cmd + ["--output-format", "json"], sandbox, stdin=user)
            out = json.loads(p.stdout)
        if out.get("is_error") and LIMIT_PATTERNS.search(str(out.get("result", ""))):
            raise UsageLimitError(str(out.get("result"))[:300])
        u = out.get("usage", {})
        tin = u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
        return out.get("result", ""), tin, u.get("output_tokens", 0), out.get("total_cost_usd")


class CodexClient(_CLIClient):
    """`codex exec` in a read-only sandbox, schema-constrained answer, event log checked for tool use."""
    binary = "codex"
    supports_images = True

    def _call(self, system, user, sandbox, images=()):
        schema = sandbox / "action.schema.json"
        schema.write_text(json.dumps(ACTION_SCHEMA))
        last = sandbox / "last.txt"
        # --ignore-user-config + memories off: no saved memories, MCP servers or rules from ~/.codex leak in
        # (with memories on, ~54k tokens of notes from the user's other sessions were added to every call).
        cmd = ["codex", "exec", "--json", "--skip-git-repo-check", "--ephemeral", "--sandbox", "read-only",
               "--ignore-user-config", "--ignore-rules", "-c", "features.memories=false",
               "-C", str(sandbox), "--output-schema", str(schema), "-o", str(last)]
        if self.cfg.get("model"):
            cmd += ["-m", self.cfg["model"]]
        if self.cfg.get("effort"):
            cmd += ["-c", f'model_reasoning_effort="{self.cfg["effort"]}"']
        for i, im in enumerate(images, 1):
            path = sandbox / f"image_{i}.jpg"
            path.write_bytes(im)
            cmd += ["-i", str(path)]
        cmd += ["-"]
        p = self._run(cmd, sandbox, stdin=f"{system}\n\n{user}")
        tin = tout = cached = 0
        for line in p.stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = ev.get("item") or {}
            if item.get("type") in ("command_execution", "file_change", "mcp_tool_call", "web_search"):
                raise TaintedRunError(f"codex attempted {item.get('type')}: {str(item)[:200]}")
            if ev.get("type") == "error" and LIMIT_PATTERNS.search(str(ev)):
                raise UsageLimitError(str(ev)[:300])
            u = ev.get("usage") or {}
            tin += u.get("input_tokens", 0)                 # OpenAI: input_tokens already INCLUDES cached
            cached += u.get("cached_input_tokens", 0)
            tout += u.get("output_tokens", 0)
        text = last.read_text() if last.exists() else p.stdout
        self._cached = cached
        return text, tin, tout, None


class GeminiClient(_CLIClient):
    """`gemini -p` in plan (read-only) mode, JSON output; any tool call in the stats taints the turn."""
    binary = "gemini"

    def _call(self, system, user, sandbox, images=()):
        cmd = ["gemini", "-o", "json", "--approval-mode", "plan", "--skip-trust", "-p", f"{system}\n\n{user}"]
        if self.cfg.get("model"):
            cmd += ["-m", self.cfg["model"]]
        p = self._run(cmd, sandbox)
        start = p.stdout.find("{")
        out = json.loads(p.stdout[start:]) if start >= 0 else {}
        if out.get("error"):
            msg = str(out["error"])
            raise (UsageLimitError if LIMIT_PATTERNS.search(msg) else RuntimeError)(msg[:300])
        stats = out.get("stats") or {}
        calls = (stats.get("tools") or {}).get("totalCalls", 0)
        if calls:
            raise TaintedRunError(f"gemini made {calls} tool call(s)")
        tin = tout = 0
        for m in (stats.get("models") or {}).values():
            t = m.get("tokens") or {}
            tin += t.get("prompt", 0) or t.get("input", 0)
            tout += t.get("candidates", 0) + t.get("thoughts", 0)
        return out.get("response", ""), tin, tout, None


class AntigravityClient(_CLIClient):
    """`agy -p` (Google Antigravity CLI) in read-only plan mode with the terminal sandbox on, schema-enforced answer.

    We read the event stream rather than the turn count: schema enforcement makes agy add a hidden second turn
    in which the model re-emits its answer through an internal `finish` tool. That is not tool use. Any OTHER
    tool step (files, shell, browser, …) taints the episode.

    Images: agy's stream input is text-only, so photos are written into the sandbox and @-mentioned; the agent
    opens them with its `view_file` tool. That one tool is allowed only on exactly those image paths."""
    binary = "agy"
    supports_images = True
    ALLOWED_TOOLS = {"finish"}

    def _call(self, system, user, sandbox, images=()):
        allowed_files = set()
        for i, im in enumerate(images, 1):
            path = (sandbox / f"image_{i}.jpg").resolve()
            path.write_bytes(im)
            allowed_files.add(str(path))
        if allowed_files:
            mentions = " ".join(f"@{f}" for f in sorted(allowed_files))
            user = (f"{user}\n\nAttached photos: {mentions}\nLook at them with view_file on exactly these paths. "
                    "Do not run commands, list folders, process the images, or open any other file: any other "
                    "tool use ends the episode with a score of 0. Then answer with your action.")
        cmd = ["agy", "-p", f"{system}\n\n{user}", "--output-format", "stream-json", "--mode", "plan", "--sandbox",
               "--json-schema", json.dumps(ACTION_SCHEMA)]
        if self.cfg.get("model"):
            cmd += ["--model", self.cfg["model"]]
        if self.cfg.get("effort"):
            cmd += ["--effort", self.cfg["effort"]]
        p = self._run(cmd, sandbox)
        tin = tout = 0
        result = None
        for line in p.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = ev.get("step_update") or {}
            if step.get("step_type") == "tool" and step.get("tool_name") not in self.ALLOWED_TOOLS:
                params = (step.get("tool_info") or {}).get("parameters") or {}
                target = params.get("AbsolutePath")
                viewing_photo = step.get("tool_name") == "view_file" and target and \
                    str(pathlib.Path(target).resolve()) in allowed_files
                if not viewing_photo:
                    raise TaintedRunError(f"agy used its own tool: {step.get('tool_name')} {str(params)[:120]}")
            u = step.get("usage") or {}
            tin += u.get("input_tokens", 0) + u.get("cache_read_tokens", 0)
            tout += u.get("output_tokens", 0) + u.get("thinking_tokens", 0)
            if ev.get("event") == "result":
                result = ev.get("result") or {}
        if result and result.get("status") == "ERROR":
            # agy sometimes flags the hidden `finish` turn as an error after the model already answered;
            # keep the answer if it starts with a well-formed action.
            resp = (result.get("response") or "").strip()
            try:
                obj, _ = json.JSONDecoder().raw_decode(resp)    # first JSON object, even if pretty-printed
                if isinstance(obj, dict) and "tool" in obj:
                    return json.dumps(obj), tin, tout, None
            except json.JSONDecodeError:
                pass
        if result is None or result.get("status") != "SUCCESS":
            msg = str(result or p.stdout[-300:])[:300]
            raise (UsageLimitError if LIMIT_PATTERNS.search(msg) else RuntimeError)(f"agy: {msg}")
        return result.get("response", ""), tin, tout, None


CLI_CLIENTS = {"claude_cli": ClaudeCodeClient, "codex_cli": CodexClient, "gemini_cli": GeminiClient,
               "agy_cli": AntigravityClient}
