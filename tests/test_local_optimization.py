"""Unit tests for the deterministic parts of the local LLM benchmark."""

from evaluation.run_local_optimization import profiles, score_answer


def test_empty_answer_gets_no_quality_points() -> None:
    score = score_answer("", [["disk"], ["iowait"]])

    assert score["score"] == 0.0
    assert score["within_1200_chars"] is False


def test_quality_rubric_scores_facts_structure_and_concision() -> None:
    answer = "Диагноз: проблема DNS.\nПроверки: dig api.internal"
    score = score_answer(answer, [["dns"], ["dig"]])

    assert score == {
        "score": 10.0,
        "matched_fact_groups": 2,
        "total_fact_groups": 2,
        "has_required_structure": True,
        "within_1200_chars": True,
    }


def test_optimized_profile_has_bounded_deterministic_generation() -> None:
    baseline, optimized = profiles("qwen2.5:3b", "llm-lab-sre:latest")

    assert baseline.options["num_predict"] == 512
    assert optimized.options["temperature"] == 0.1
    assert optimized.options["seed"] == 42
    assert optimized.options["num_predict"] == 192
    assert optimized.options["num_ctx"] == 4096
    assert optimized.system is not None
    assert "remediation" in optimized.system
