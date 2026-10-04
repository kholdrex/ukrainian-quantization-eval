"""Accuracy, language-specific accuracy change against F16, answer changes, speed and memory; tables and figures."""

import json
from pathlib import Path

import numpy as np

from .data import LANGS, LETTERS, Item
from .experiment import CONFIGS

SEED = 2026
RESAMPLES = 10000


def load(path: Path) -> dict[tuple[str, str], dict]:
    with path.open() as f:
        return {(r["id"], r["lang"]): r for r in map(json.loads, f)}


def choice(row: dict) -> str | None:
    letters = row["letters"]
    return max(letters, key=letters.get) if letters else None


def distribution(row: dict) -> np.ndarray:
    logp = np.array([row["letters"].get(x, -np.inf) for x in LETTERS])
    if np.all(np.isinf(logp)):
        return np.full(4, 0.25)
    p = np.exp(logp - logp.max())
    return p / p.sum()


def interval(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def holm(pvalues: dict) -> dict:
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def accuracy_tables(items: list[Item], results: Path) -> tuple[dict, dict]:
    rng = np.random.default_rng(SEED)
    boots = rng.integers(0, len(items), (RESAMPLES, len(items)))
    out, pvalues = {}, {}
    for config in CONFIGS:
        rows = {fmt: load(results / "answers" / f"{config.family}-{fmt}.jsonl") for fmt in config.formats}
        correct = {
            (fmt, lang): np.array([choice(rows[fmt][(it.id, lang)]) == it.answer for it in items], dtype=float)
            for fmt in config.formats
            for lang in LANGS
        }
        fam = {"label": config.label, "formats": {}}
        for fmt in config.formats:
            entry = {}
            for lang in LANGS:
                v = correct[(fmt, lang)]
                failures = np.mean([rows[fmt][(it.id, lang)]["token"].strip() not in LETTERS for it in items])
                entry[lang] = {
                    "accuracy": float(v.mean()),
                    "ci": interval(v[boots].mean(axis=1)),
                    "format_failures": float(failures),
                }
            if fmt != "fp16":
                for lang in LANGS:
                    ref = rows["fp16"]
                    changed = [choice(rows[fmt][(it.id, lang)]) != choice(ref[(it.id, lang)]) for it in items]
                    kl = [
                        float(np.sum(p * np.log(np.clip(p, 1e-12, 1) / np.clip(q, 1e-12, 1))))
                        for p, q in (
                            (distribution(ref[(it.id, lang)]), distribution(rows[fmt][(it.id, lang)])) for it in items
                        )
                    ]
                    delta = correct[(fmt, lang)] - correct[("fp16", lang)]
                    entry[lang].update(
                        {
                            "change": float(delta.mean()),
                            "change_ci": interval(delta[boots].mean(axis=1)),
                            "answers_changed": float(np.mean(changed)),
                            "correct_to_wrong": int((delta < 0).sum()),
                            "wrong_to_correct": int((delta > 0).sum()),
                            "kl": float(np.mean(kl)),
                        }
                    )
                d = (correct[(fmt, "uk")] - correct[("fp16", "uk")]) - (correct[(fmt, "en")] - correct[("fp16", "en")])
                dist = d[boots].mean(axis=1)
                p = min(1.0, 2 * min((dist <= 0).mean(), (dist >= 0).mean()))
                entry["language_difference"] = {"value": float(d.mean()), "ci": interval(dist), "p": float(p)}
                pvalues[(config.family, fmt)] = p
            fam["formats"][fmt] = entry
        out[config.family] = fam
    for (family, fmt), p in holm(pvalues).items():
        out[family]["formats"][fmt]["language_difference"]["p_holm"] = p
    by_category = {}
    for config in CONFIGS:
        for fmt in config.formats:
            rows = load(results / "answers" / f"{config.family}-{fmt}.jsonl")
            for cat in sorted({it.category for it in items}):
                sel = [it for it in items if it.category == cat]
                by_category[f"{config.family}|{fmt}|{cat}"] = {
                    lang: float(np.mean([choice(rows[(it.id, lang)]) == it.answer for it in sel])) for lang in LANGS
                }
    return out, by_category


CONFIDENCE_BINS = (0.0, 0.9, 0.99, 1.0 + 1e-9)


def changes_by_confidence(items: list[Item], results: Path) -> dict:
    """Exploratory: share of answers that differ from F16, by the F16 probability of the chosen letter."""
    out = {}
    for config in CONFIGS:
        ref = load(results / "answers" / f"{config.family}-fp16.jsonl")
        for fmt in config.formats[1:]:
            rows = load(results / "answers" / f"{config.family}-{fmt}.jsonl")
            for lang in LANGS:
                conf = np.array([distribution(ref[(it.id, lang)]).max() for it in items])
                changed = np.array([choice(ref[(it.id, lang)]) != choice(rows[(it.id, lang)]) for it in items])
                cells = []
                for lo, hi in zip(CONFIDENCE_BINS, CONFIDENCE_BINS[1:], strict=False):
                    sel = (conf >= lo) & (conf < hi)
                    cells.append(
                        {
                            "from": lo,
                            "to": min(hi, 1.0),
                            "share": float(sel.mean()),
                            "changed": float(changed[sel].mean()) if sel.any() else None,
                        }
                    )
                out[f"{config.family}|{fmt}|{lang}"] = cells
    return out


def speed_tables(results: Path) -> dict:
    out = {}
    for config in CONFIGS:
        for fmt in config.formats:
            name = f"{config.family}-{fmt}"
            with (results / "bench" / f"{name}.jsonl").open() as f:
                rows = [json.loads(line) for line in f]
            meta = json.loads((results / "models" / f"{name}.json").read_text())
            warm = [r for r in rows if r["kind"] == "warm"]
            aborted = [r for r in rows if r["kind"] == "aborted"]
            entry = {
                "file_gb": meta["size"] / 1e9,
                "memory_gb": next(r for r in rows if r["kind"] == "memory")["size"] / 1e9,
                "load_s": next(r for r in rows if r["kind"] == "cold")["load"],
            }
            for lang in LANGS:
                w = [r for r in warm if r["lang"] == lang]
                entry[lang] = {
                    "ttft_s": float(np.median([r["ttft"] for r in w])),
                    "prompt_tps": float(np.median([r["prompt_tokens"] / r["prompt_seconds"] for r in w])),
                    "decode_tps": float(np.median([r["eval_tokens"] / r["eval_seconds"] for r in w])),
                    "prompt_tokens": float(np.mean([r["prompt_tokens"] for r in w])),
                    "aborted": sum(r["lang"] == lang for r in aborted),
                }
            out[name] = entry
    return out


def token_ratio(items: list[Item], results: Path) -> dict:
    out = {}
    for config in CONFIGS:
        rows = load(results / "answers" / f"{config.family}-fp16.jsonl")
        en = np.array([rows[(it.id, "en")]["prompt_tokens"] for it in items], dtype=float)
        uk = np.array([rows[(it.id, "uk")]["prompt_tokens"] for it in items], dtype=float)
        out[config.family] = {"en": float(en.mean()), "uk": float(uk.mean()), "ratio": float((uk / en).mean())}
    return out


def label(fmt: str) -> str:
    return "F16" if fmt == "fp16" else fmt.upper()


def figures(acc: dict, path: Path) -> None:
    """Accuracy and the share of answers that differ from F16, by precision level, for both languages."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"en": ("#000000", "o", "-", "English"), "uk": ("#666666", "s", "--", "Ukrainian")}
    fig, axes = plt.subplots(2, len(CONFIGS), figsize=(11, 6.2), sharex="col")
    for col, config in enumerate(CONFIGS):
        fmts = list(config.formats)
        x = np.arange(len(fmts))
        formats = acc[config.family]["formats"]
        for lang, (colour, marker, line, name) in styles.items():
            values = np.array([100 * formats[f][lang]["accuracy"] for f in fmts])
            lo = values - np.array([100 * formats[f][lang]["ci"][0] for f in fmts])
            hi = np.array([100 * formats[f][lang]["ci"][1] for f in fmts]) - values
            axes[0][col].errorbar(
                x, values, yerr=[lo, hi], color=colour, marker=marker, linestyle=line, capsize=3, label=name
            )
            changed = [0.0] + [100 * formats[f][lang]["answers_changed"] for f in fmts[1:]]
            axes[1][col].plot(x, changed, color=colour, marker=marker, linestyle=line, label=name)
        axes[0][col].set_title(config.label, fontsize=12)
        axes[1][col].set_xticks(x, [label(f) for f in fmts], fontsize=9)
        axes[1][col].set_xlabel("Precision level", fontsize=11)
        for row in (0, 1):
            axes[row][col].grid(alpha=0.3)
            axes[row][col].tick_params(labelsize=9)
    axes[0][0].set_ylabel("Accuracy, %", fontsize=11)
    axes[1][0].set_ylabel("Answers changed vs F16, %", fontsize=11)
    axes[0][0].legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(path / "accuracy_by_level.png", dpi=300)
    plt.close(fig)


def main(items: list[Item], results: Path, figures_dir: Path) -> None:
    acc, by_category = accuracy_tables(items, results)
    speed = speed_tables(results)
    summary = {
        "items": len(items),
        "accuracy": acc,
        "by_category": by_category,
        "speed": speed,
        "tokens": token_ratio(items, results),
        "changes_by_confidence": changes_by_confidence(items, results),
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    figures_dir.mkdir(exist_ok=True)
    figures(acc, figures_dir)
