"""List-price cost for runs billed to a subscription (CLI harnesses report tokens, not dollars)."""
CODEX_CACHED_SHARE = 12672 / 22654   # measured on `codex exec`, Oct 2026: cached share of its fixed prompt


def list_price(row: dict, model: dict) -> float | None:
    """Cost of one recorded episode at the model's public per-token price, or None if it can't be priced.
    Rows written before list pricing existed have cost_usd 0; older Codex rows also double-counted cached
    input (input_tokens already includes it), so their split is estimated."""
    u = row["usage"]
    if u.get("cost_usd") or "input_per_m" not in model:
        return u.get("cost_usd") or None
    if u.get("api_equivalent_usd"):              # Claude Code reports its own API-price total
        return u["api_equivalent_usd"]
    cin, cout = model["input_per_m"], model["output_per_m"]
    cc = model.get("cached_input_per_m", cin)
    if model.get("provider") == "codex_cli":
        inp = u["input_tokens"] / (1 + CODEX_CACHED_SHARE)
        cached = inp * CODEX_CACHED_SHARE
        return ((inp - cached) * cin + cached * cc + u["output_tokens"] * cout) / 1e6
    return (u["input_tokens"] * cin + u["output_tokens"] * cout) / 1e6
