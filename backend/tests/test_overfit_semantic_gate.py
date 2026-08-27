from app.training.overfit_sanity import cer, semantic_gate_failures


def test_space_only_prediction_fails_semantic_gate() -> None:
    failures = semantic_gate_failures(
        "i am hungry",
        " ",
        space_ratio=1.0,
        unique_meaningful=0,
    )
    assert failures
    assert any("space collapse" in item for item in failures)
    assert any("meaningful token diversity" in item for item in failures)


def test_correct_prediction_passes_semantic_gate() -> None:
    failures = semantic_gate_failures(
        "hello",
        "hello",
        space_ratio=0.1,
        unique_meaningful=4,
    )
    assert failures == []


def test_empty_prediction_has_full_character_error_rate() -> None:
    assert cer("", "hello") == 1.0
