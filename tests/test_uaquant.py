import json

import numpy as np
import pytest

from uaquant import analysis, experiment, ollama, prompts
from uaquant.data import Item

ITEM = Item(
    "x/test/1",
    "x",
    "STEM",
    "B",
    {"en": ("Two plus two?", ("3", "4", "5", "6")), "uk": ("Два плюс два?", ("3", "4", "5", "6"))},
)


def test_question_lists_options_with_latin_letters():
    text = prompts.question(ITEM, "uk")
    assert text.splitlines() == ["Два плюс два?", "A. 3", "B. 4", "C. 5", "D. 6"]


def test_letter_logprobs_merge_spaced_tokens_and_ignore_others():
    top = [
        {"token": " B", "logprob": -0.1},
        {"token": "B", "logprob": -2.0},
        {"token": "The", "logprob": -0.5},
        {"token": "A", "logprob": -3.0},
        {"token": "AB", "logprob": -1.0},
    ]
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


def test_a_resumed_run_refuses_a_different_model_file(tmp_path, monkeypatch):
    files = {"m:q4": {"digest": "a", "size": 1, "details": {}}}
    monkeypatch.setattr(ollama, "installed", lambda: files)
    monkeypatch.setattr(ollama, "version", lambda: "0.34.3")
    path = tmp_path / "m.json"
    experiment.check_identity(path, "m:q4")
    experiment.check_identity(path, "m:q4")
    files["m:q4"] = {"digest": "b", "size": 1, "details": {}}
    with pytest.raises(RuntimeError):
        experiment.check_identity(path, "m:q4")


def test_aborted_generations_are_recorded_separately(tmp_path, monkeypatch):
    def fake_stream(tag, messages, tokens):
        if "Два" in messages[-1]["content"]:
            return {"error": "prediction aborted", "ttft": None, "wall": 1.0, "attempts": 1}
        return {
            "ttft": 0.1,
            "wall": 1.0,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 1e8,
            "eval_count": 128,
            "eval_duration": 2e9,
            "load_duration": 1e9,
            "attempts": 1,
        }

    monkeypatch.setattr(ollama, "stream", fake_stream)
    monkeypatch.setattr(ollama, "loaded", lambda: [{"name": "m", "size": 5, "size_vram": 5}])
    monkeypatch.setattr(ollama, "unload", lambda name: None)
    path = tmp_path / "bench.jsonl"
    experiment.bench("m", [ITEM, ITEM], path)
    kinds = [json.loads(line)["kind"] for line in path.read_text().splitlines()]
    assert kinds.count("aborted") == 2 and kinds.count("warm") == 2
