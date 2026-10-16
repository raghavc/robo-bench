"""Campus suite checks: every task has exactly one answer, is solvable within the caps, and resists gaming.

    python3 tests/test_campus.py   (or pytest)
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env import Building, OfficeSearchEnv  # noqa: E402
from env.baselines import (EXPLOIT_AGENTS, CampusAppearanceAgent, CampusOraclePlan,  # noqa: E402
                           CampusRandom)
from env.campus import AREA, generate_campus, load_placements, verify_campus  # noqa: E402

B = Building()
DIFFS = ("campus-hard", "campus-extreme")


def _run(cls, diff, seeds=range(60), perception="oracle"):
    env = OfficeSearchEnv(B, perception)
    out = []
    for s in seeds:
        obs = env.reset(s, diff, nonce=cls.__name__)
        out.append((cls(s) if cls is CampusRandom else cls()).run(env, obs))
    return out


mean = lambda xs, k: sum(x[k] for x in xs) / len(xs)


def test_exactly_one_answer_every_task():
    for d in ("hard", "extreme"):
        for seed in range(300):
            s = generate_campus(seed, d, B)
            assert verify_campus(s), (d, seed)
            present = load_placements("P1" in s.meta["placements_present"])
            lookalikes = [p for p in present if p.variant == s.meta["variant"] and p.id != s.backpack["placement"]]
            assert all(p.id in s.meta["eliminations"] for p in lookalikes), (d, seed)


def test_cross_floor_lookalikes_are_on_the_restricted_list():
    for seed in range(200):
        s = generate_campus(seed, "extreme", B)
        t_floor = int(s.backpack["node"][1])
        for p in load_placements("P1" in s.meta["placements_present"]):
            if p.variant == s.meta["variant"] and p.floor != t_floor:
                assert p.floor in s.meta["restricted_floors"]
                assert AREA[p.floor] in s.clues["badge_log"]["access"]["no_access"]
        assert all(int(e.node[1]) not in s.meta["restricted_floors"] for e in s.timeline if e.happened)


def test_every_task_is_solvable_within_caps():
    for d in DIFFS:
        res = _run(CampusOraclePlan, d, range(100))
        assert mean(res, "success") == 1.0
        assert max(r["steps"] for r in res) <= 10


def test_appearance_alone_fails_on_extreme():
    assert mean(_run(CampusAppearanceAgent, "campus-extreme", range(100)), "success") <= 0.05


def test_guessing_and_shortcuts_fail():
    for d in DIFFS:
        assert mean(_run(CampusRandom, d), "success") < 0.15
        assert mean(_run(EXPLOIT_AGENTS["exploit_blind"], d), "success") == 0
        assert mean(_run(EXPLOIT_AGENTS["exploit_uninspected"], d), "success") == 0
        assert mean(_run(EXPLOIT_AGENTS["exploit_trust_guess"], d), "success") < 0.15
        assert mean(_run(EXPLOIT_AGENTS["exploit_inspect_all"], d), "success") < 0.15


def test_step_cap_is_enforced():
    env = OfficeSearchEnv(B)
    env.reset(0, "campus-extreme")
    for _ in range(30):
        if env.done:
            break
        env.step({"tool": "read_calendar"})
    assert env.done and len(env.trajectory) - 1 == 12 and env.grade()[1]["outcome"] == "turn_limit"


def test_scope_is_berkeley_only():
    env = OfficeSearchEnv(B)
    obs = env.reset(1, "campus-hard")
    assert all(r["floor"] in (3, 4, 5, 6) for r in obs["rooms"])
    assert "error" in env.step({"tool": "go_to", "room": "F1_N03"}).observation


def test_no_answer_leaks_in_observations():
    # Public fields ("rooms" directory, "views" in search results) are fine; these are hidden-only.
    hidden_keys = re.compile(r'"(placement|lookalike|eliminated_by|variant|bag|optimal_plan|solution)"|\bP\d{1,2}\b|'
                             r'hat_keychain|variants/|\.jpg|restricted_floor|early_sighting|attribution|passed_through')
    for perception in ("oracle", "vlm"):
        env = OfficeSearchEnv(B, perception)
        for seed in range(15):
            obs = env.reset(seed, "campus-extreme")
            texts = [json.dumps(obs)]
            for a in env.s.meta["optimal_plan"]["actions"][:-2] + [{"tool": "read_badge_log"}]:
                if env.done:
                    break
                texts.append(json.dumps(env.step(a).observation))
            blob = "\n".join(texts)
            assert not hidden_keys.search(blob), (perception, seed, hidden_keys.search(blob).group(0))


def test_photo_mode_report_accepts_any_view_of_the_bag():
    env = OfficeSearchEnv(B, "vlm")
    for seed in range(200):
        env.reset(seed, "campus-extreme")
        if len(env.s.backpack["views"]) > 1:
            break
    views = env.s.backpack["views"]
    node = views[-1].rsplit("_v", 1)[0]
    if node != env.node:
        env.step({"tool": "go_to", "room": node})
    env.step({"tool": "search"})
    r = env.step({"tool": "report", "view": views[-1]})
    assert r.info["success"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS  {n}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {n}  {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
