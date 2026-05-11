import os

for env_var in [
    "REDDIT",
    "SHOPPING",
    "SHOPPING_ADMIN",
    "GITLAB",
    "WIKIPEDIA",
    "MAP",
    "HOMEPAGE",
]:
    os.environ.setdefault(env_var, f"http://{env_var.lower()}.example")

from evaluation_harness.evaluators import StringEvaluator  # noqa: E402


def test_numeric_must_include_accepts_equivalent_currency_answer() -> None:
    assert StringEvaluator.must_include("0", "$0.00", tokenize=True) == 1.0


def test_numeric_must_include_keeps_single_digit_false_positive_guard() -> None:
    assert StringEvaluator.must_include("0", "$10.00", tokenize=True) == 0.0
    assert StringEvaluator.must_include("0", "20", tokenize=True) == 0.0


def test_numeric_must_include_accepts_plain_zero_answer() -> None:
    assert StringEvaluator.must_include("0", "0", tokenize=True) == 1.0
