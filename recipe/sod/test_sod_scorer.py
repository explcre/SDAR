#!/usr/bin/env python
"""CPU unit tests for the SOD benchmark reward path in SDAR.

Verifies recipe/sod/reward.py (strict-box math_dapo, the SAME scoring SOD uses) grades
boxed AIME answers correctly. Also documents that verl's DEFAULT dispatcher would mis-score
(it uses math_dapo with strict_box_verify=False = Minerva 'Answer:' matching), which is why
the SOD benchmark must use the custom reward wrapper.
Run: python recipe/sod/test_sod_scorer.py   (needs the sdar env; no GPU).
"""
from recipe.sod.reward import compute_score


def _score(data_source, solution, gt):
    r = compute_score(data_source=data_source, solution_str=solution, ground_truth=gt)
    return r["score"] if isinstance(r, dict) else r


def main() -> None:
    # 1) correct boxed answer scores higher than wrong, for both AIME sources
    for ds in ("aime2024", "aime2025"):
        good = _score(ds, "We reason... therefore the answer is \\boxed{33}.", "33")
        bad = _score(ds, "I think it is \\boxed{99}.", "33")
        print(f"{ds}: correct={good}  wrong={bad}")
        assert good > bad, f"{ds}: correct boxed answer must outscore wrong one"
        assert good > 0, f"{ds}: correct boxed answer must score > 0"

    # 2) routing: aime* must reach math_dapo (not crash / not default-0 everything)
    assert _score("aime2024", "\\boxed{0}", "0") > 0, "exact boxed match must score"

    # 3) no boxed answer -> not counted correct (the SOD loop-failure mode)
    noboxed = _score("aime2024", "let me write code, let me write code ...", "33")
    print(f"no-boxed -> {noboxed}")
    assert noboxed == 0 or noboxed < 1, "missing boxed answer must not be graded correct"

    print("OK: SOD reward path (aime* -> math_dapo) routes + grades correctly")


if __name__ == "__main__":
    main()
