import numpy as np

from uaquant import analysis, prompts
from uaquant.data import Item

ITEM = Item("x/test/1", "x", "STEM", "B", {"en": ("Two plus two?", ("3", "4", "5", "6")),
                                           "uk": ("Два плюс два?", ("3", "4", "5", "6"))})


def test_question_lists_options_with_latin_letters():
    text = prompts.question(ITEM, "uk")
    assert text.splitlines() == ["Два плюс два?", "A. 3", "B. 4", "C. 5", "D. 6"]


def test_letter_logprobs_merge_spaced_tokens_and_ignore_others():
    top = [{"token": " B", "logprob": -0.1}, {"token": "B", "logprob": -2.0}, {"token": "The", "logprob": -0.5},
           {"token": "A", "logprob": -3.0}, {"token": "AB", "logprob": -1.0}]
    assert prompts.letter_logprobs(top) == {"B": -0.1, "A": -3.0}


def test_choice_and_distribution():
    row = {"letters": {"A": -3.0, "B": -0.1}}
    assert analysis.choice(row) == "B"
    p = analysis.distribution(row)
    assert np.isclose(p.sum(), 1) and p[2] == 0 and p[1] > p[0]
    assert analysis.choice({"letters": {}}) is None
    assert np.allclose(analysis.distribution({"letters": {}}), 0.25)


def test_holm_is_monotone():
    adjusted = analysis.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] == 0.03 and adjusted["c"] == 0.06 and adjusted["b"] == 0.06
