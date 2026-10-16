"""Red-team suite for the verifier. Run: python3 tests/test_verifier.py  (or pytest)

Each test encodes one way an agent might try to score without solving the task.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env import Building, OfficeSearchEnv, generate  # noqa: E402
from env.baselines import EXPLOIT_AGENTS, ClueAgent  # noqa: E402

B = Building()
SEEDS = range(60)


def _run(cls, difficulty, seeds=SEEDS, perception="oracle"):
    env = OfficeSearchEnv(B, perception)
    out = []
    for s in seeds:
        obs = env.reset(s, difficulty, nonce=cls.__name__)
        out.append(cls().run(env, obs))
    return out


def _mean(xs, key):
    return sum(x[key] for x in xs) / len(xs)


def test_blind_reports_never_score():
    for d in ("easy", "medium", "hard"):
        assert _mean(_run(EXPLOIT_AGENTS["exploit_blind"], d), "success") == 0


def test_reporting_uninspected_bags_is_rejected():
    for d in ("medium", "hard"):
        res = _run(EXPLOIT_AGENTS["exploit_uninspected"], d)
        assert _mean(res, "success") == 0
        assert all(r["reported"] is None for r in res)


def test_skipping_the_close_look_loses_to_decoys():
    honest = _run(ClueAgent, "hard")
    lazy = _run(EXPLOIT_AGENTS["exploit_first_bag"], "hard")
    assert _mean(lazy, "success") < _mean(honest, "success") - 0.2
    assert _mean(lazy, "reward") < 0.25


def test_trusting_the_owners_guess_is_penalised():
    honest = _run(ClueAgent, "hard")
    gullible = _run(EXPLOIT_AGENTS["exploit_trust_guess"], "hard")
    assert _mean(gullible, "reward") < _mean(honest, "reward") - 0.3
    assert _mean(gullible, "process") < 0.5


def test_brute_force_cannot_buy_the_answer():
    honest = _run(ClueAgent, "hard")
    spam = _run(EXPLOIT_AGENTS["exploit_inspect_all"], "hard")
    assert _mean(spam, "reward") < _mean(honest, "reward") - 0.4


def test_one_report_per_episode():
    env = OfficeSearchEnv(B)
    env.reset(1, "easy")
    env.step({"tool": "report", "object": "nope"})          # invalid → not counted, costs budget
    assert not env.done and env.spent > 0


def test_object_ids_are_fresh_every_episode():
    env = OfficeSearchEnv(B)
    env.reset(3, "hard")
    ids_a = {oid for oid, o in env.objects.items() if o["bag"] == "target"}
    env.reset(3, "hard")
    ids_b = {oid for oid, o in env.objects.items() if o["bag"] == "target"}
    assert ids_a != ids_b


def test_observations_do_not_leak_the_answer():
    """Walk a full honest episode and scan every observation for hidden fields."""
    env = OfficeSearchEnv(B)
    for seed in range(20):
        obs0 = env.reset(seed, "hard")
        s = env.s
        hidden = [s.backpack["spot"], "target", "decoy", "bystander", "variants", ".jpg", "\"seed\"", s.backpack["view"] + "__"]
        texts = [json.dumps(obs0)]
        agent = ClueAgent()
        orig = env.step

        def spy(action, _orig=orig):
            r = _orig(action)
            texts.append(json.dumps(r.observation))
            return r
        env.step = spy
        agent.run(env, obs0)
        env.step = orig
        blob = "\n".join(texts).lower()
        for h in hidden:
            if h == s.backpack["spot"] and h.lower() in " ".join(v.description.lower() for v in B.views.values()):
                continue   # spot names that also appear in public view descriptions are fine
            assert h.lower() not in blob, (seed, h)


def test_budget_is_a_hard_stop():
    env = OfficeSearchEnv(B)
    env.reset(0, "hard")
    while not env.done:
        env.step({"tool": "search"})
    assert env.spent >= env.s.budget
    assert env.grade()[1]["outcome"] == "budget_exhausted"


def test_vlm_mode_requires_searched_view_and_hides_files():
    env = OfficeSearchEnv(B, "vlm")
    env.reset(2, "medium")
    r = env.step({"tool": "report", "view": env.s.backpack["view"]})   # never searched there
    assert "error" in r.observation and not env.done
    r = env.step({"tool": "search"})
    ref = r.observation["image"]                        # one 2x2 collage, opaque handle
    assert ref.startswith("img_") and "jpg" not in ref and env.render_image(ref).size == (576, 768)


def test_zoom_is_scoped_costed_and_opaque():
    env = OfficeSearchEnv(B, "vlm")
    env.reset(2, "medium")
    view = env.b.nodes[env.node].views[0].id
    r = env.step({"tool": "zoom", "view": view, "region": "0.2,0.2,0.6,0.6"})
    assert "error" in r.observation                       # must search the room first
    env.step({"tool": "search"})
    spent = env.spent
    for bad in ("0.5,0.5,0.4,0.9", "0,0,1", "a,b,c,d", "0.1,0.1,0.12,0.9"):
        assert "error" in env.step({"tool": "zoom", "view": view, "region": bad}).observation
    other = next(v for v in env.b.views if not v.startswith(env.node))
    assert "error" in env.step({"tool": "zoom", "view": other, "region": "0,0,1,1"}).observation
    r = env.step({"tool": "zoom", "view": view, "region": "0.25,0.4,0.75,0.9"})
    assert r.observation["image"].startswith("img_") and env.spent > spent
    assert max(env.render_image(r.observation["image"]).size) <= 768
    oracle = OfficeSearchEnv(B)
    oracle.reset(2, "medium")
    oracle.step({"tool": "search"})
    assert "error" in oracle.step({"tool": "zoom", "view": view, "region": "0,0,1,1"}).observation


def test_every_object_keeps_a_unique_id():
    env = OfficeSearchEnv(B)
    for seed in range(30):
        env.reset(seed, "campus-extreme")
        assert sum(o["bag"] == "target" for o in env.objects.values()) == 1
        n_items = sum(len(c) for c in env.scene_objects.values()) + 1 + len(env.s.decoys) + len(env.s.bystanders)
        assert len(env.objects) == n_items


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
