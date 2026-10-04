"""Paired English and Ukrainian Global-MMLU questions."""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

URL = "https://huggingface.co/datasets/CohereLabs/Global-MMLU/resolve/{revision}/{lang}/test-00000-of-00001.parquet"
REVISION = "0e619dbeb34206cd48705a1a0ea7fb21cae09993"
LANGS = ("en", "uk")
LETTERS = "ABCD"
OPTIONS = ("option_a", "option_b", "option_c", "option_d")


@dataclass(frozen=True)
class Item:
    id: str
    subject: str
    category: str
    answer: str
    text: dict[str, tuple[str, tuple[str, ...]]]


def download(root: Path) -> None:
    for lang in LANGS:
        dest = root / "gmmlu" / f"{lang}.parquet"
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(URL.format(revision=REVISION, lang=lang), timeout=300)
        response.raise_for_status()
        dest.write_bytes(response.content)


def load(root: Path) -> list[Item]:
    frames = {lang: pd.read_parquet(root / "gmmlu" / f"{lang}.parquet").set_index("sample_id") for lang in LANGS}
    items = []
    for sid, en in frames["en"].iterrows():
        uk = frames["uk"].loc[sid]
        if en["answer"] != uk["answer"]:
            raise ValueError(f"answer mismatch for {sid}")
        text = {
            lang: (str(row["question"]), tuple(str(row[c]) for c in OPTIONS)) for lang, row in (("en", en), ("uk", uk))
        }
        items.append(Item(sid, en["subject"], en["subject_category"], en["answer"], text))
    return items


def stratified(items: list[Item], per_category: int, seed: int) -> list[Item]:
    """A fixed number of questions per subject category; repeated questions are kept once."""
    seen, unique = set(), []
    for it in items:
        key = it.text["en"]
        if key not in seen:
            seen.add(key)
            unique.append(it)
    rng = random.Random(seed)
    sample = []
    for category in sorted({it.category for it in unique}):
        pool = [it for it in unique if it.category == category]
        sample += rng.sample(pool, per_category)
    return sorted(sample, key=lambda it: it.id)


def write_manifest(path: Path, sample: list[Item], seed: int, per_category: int) -> None:
    manifest = {"revision": REVISION, "seed": seed, "per_category": per_category, "items": [it.id for it in sample]}
    path.write_text(json.dumps(manifest, indent=1) + "\n")


def from_manifest(path: Path, items: list[Item]) -> list[Item]:
    ids = json.loads(path.read_text())["items"]
    index = {it.id: it for it in items}
    return [index[i] for i in ids]
