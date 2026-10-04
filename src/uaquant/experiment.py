"""Runs every model at every quantisation level: answers to the sampled questions, then speed and memory."""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from . import ollama, prompts
from .data import LANGS, Item

FORMATS = ("fp16", "q8_0", "q6_K", "q5_K_M", "q4_K_M", "q3_K_M")


@dataclass(frozen=True)
class Config:
    family: str
    label: str
    prefix: str
    formats: tuple[str, ...]

    def tags(self) -> list[tuple[str, str]]:
        return [(fmt, f"{self.prefix}-{fmt}") for fmt in self.formats]


CONFIGS = (
    Config("qwen2.5-3b", "Qwen2.5 3B", "qwen2.5:3b-instruct", FORMATS),
    Config("llama3.2-3b", "Llama 3.2 3B", "llama3.2:3b-instruct", FORMATS),
    Config("gemma3-4b", "Gemma 3 4B", "gemma3:4b-it", ("fp16", "q8_0", "q4_K_M")),
)
BENCH_ITEMS = 30
BENCH_TOKENS = 128


def done(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {(r["id"], r["lang"]) for r in map(json.loads, f)}


def answer_all(tag: str, items: list[Item], path: Path) -> None:
    finished = done(path)
    with path.open("a") as f:
        for item in items:
            for lang in LANGS:
                if (item.id, lang) in finished:
                    continue
                start = time.perf_counter()
                reply = ollama.answer(tag, prompts.messages(item, lang))
                first = reply["logprobs"][0]
                row = {
                    "id": item.id,
                    "lang": lang,
                    "token": first["token"],
                    "letters": prompts.letter_logprobs(first["top_logprobs"]),
                    "prompt_tokens": reply["prompt_eval_count"],
                    "seconds": round(time.perf_counter() - start, 4),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()


def bench(tag: str, items: list[Item], path: Path) -> None:
    """Cold load with a warm-up question, then each benchmark question once per language (no prompt is repeated)."""
    if path.exists():
        return
    for m in ollama.loaded():
        ollama.unload(m["name"])
    rows = []
    cold = ollama.stream(tag, prompts.messages(items[-1], "en", "explain"), 1)
    rows.append({"kind": "cold", "load": cold["load_duration"] / 1e9, "wall": cold["wall"]})
    memory = next(m for m in ollama.loaded() if m["name"] == tag)
    rows.append({"kind": "memory", "size": memory["size"], "size_vram": memory["size_vram"]})
    for item in items[:BENCH_ITEMS]:
        for lang in LANGS:
            r = ollama.stream(tag, prompts.messages(item, lang, "explain"), BENCH_TOKENS)
            rows.append(
                {
                    "kind": "warm",
                    "id": item.id,
                    "lang": lang,
                    "ttft": r["ttft"],
                    "wall": r["wall"],
                    "prompt_tokens": r["prompt_eval_count"],
                    "prompt_seconds": r["prompt_eval_duration"] / 1e9,
                    "eval_tokens": r["eval_count"],
                    "eval_seconds": r["eval_duration"] / 1e9,
                }
            )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def run(configs: list[Config], items: list[Item], bench_items: list[Item], results: Path, keep: set[str]) -> None:
    (results / "answers").mkdir(parents=True, exist_ok=True)
    (results / "bench").mkdir(parents=True, exist_ok=True)
    for config in configs:
        for fmt, tag in config.tags():
            name = f"{config.family}-{fmt}"
            if tag not in ollama.installed():
                ollama.pull(tag)
            meta = results / "models" / f"{name}.json"
            meta.parent.mkdir(exist_ok=True)
            info = ollama.installed()[tag]
            meta.write_text(
                json.dumps(
                    {
                        "tag": tag,
                        "digest": info["digest"],
                        "size": info["size"],
                        "details": info["details"],
                        "ollama": ollama.version(),
                    },
                    indent=1,
                )
                + "\n"
            )
            answer_all(tag, items, results / "answers" / f"{name}.jsonl")
            bench(tag, bench_items, results / "bench" / f"{name}.jsonl")
            ollama.unload(tag)
            if tag not in keep:
                ollama.remove(tag)
