"""Unit tests for backend/app/training/evaluate.py's CER computation.
Pure functions, no model or dataset needed — fast, deterministic."""
from app.training.evaluate import cer, levenshtein


def test_levenshtein_identical_strings():
    assert levenshtein("hello", "hello") == 0


def test_levenshtein_empty_strings():
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "") == 3
    assert levenshtein("", "abc") == 3


def test_levenshtein_single_substitution():
    assert levenshtein("cat", "bat") == 1


def test_levenshtein_insertion_deletion():
    assert levenshtein("cat", "cats") == 1
    assert levenshtein("cats", "cat") == 1


def test_levenshtein_known_case():
    # classic textbook example
    assert levenshtein("kitten", "sitting") == 3


def test_cer_perfect_match_is_zero():
    assert cer("hello world", "hello world") == 0.0


def test_cer_case_insensitive():
    assert cer("Hello World", "hello world") == 0.0


def test_cer_empty_ground_truth_and_empty_prediction():
    assert cer("", "") == 0.0


def test_cer_empty_prediction_nonempty_ground_truth_is_worst_case():
    assert cer("", "") == 0.0
    assert cer("(no sign detected)", "hello") == cer("(no sign detected)", "hello")  # sanity: doesn't crash
    # a totally empty predicted string against real ground truth should be
    # a high error rate (every ground-truth char counts as a deletion)
    assert cer("", "hello") == 1.0


def test_cer_scales_with_ground_truth_length():
    # 1 substitution out of 5 chars vs 1 substitution out of len(gt) chars
    assert cer("hellp", "hello") == 1 / 5
    gt = "i am sitting in the class"
    predicted = "i am sittinh in the class"  # one substitution: g -> h
    assert cer(predicted, gt) == 1 / len(gt)
