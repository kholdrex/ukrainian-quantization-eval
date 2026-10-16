# GGUF quantisation of small language models on Ukrainian and English questions

Code, model outputs and measurements for the paper *Accuracy and speed of quantised small language models in
Ukrainian and English* (O. Kholodniak).

Archived on Zenodo: [doi:10.5281/zenodo.23408167](https://doi.org/10.5281/zenodo.23408167).

Three small open models (Qwen2.5 3B Instruct, Llama 3.2 3B Instruct, Gemma 3 4B IT) answer 1800 paired English and
Ukrainian Global-MMLU questions at the GGUF precision levels published in the Ollama library (F16, Q8_0, Q6_K, Q5_K_M,
Q4_K_M, Q3_K_M; Gemma 3 only F16, Q8_0, Q4_K_M). The answer is the letter with the highest first-token probability.
The primary measure is the language difference of the accuracy change relative to F16; the share of answers that
change, speed (time to first token, prompt and generation rates), memory and load time are measured on an Apple M1 Pro
laptop with 16 GB of memory.

## Data

[Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU), revision `0e619db`, Apache-2.0, `en` and `uk`
test questions: 300 per subject category (seed 2026), listed in `data/sample.json`. The files are downloaded at the
pinned revision on first use.

## Layout

```
src/uaquant/    data loading, prompts, Ollama client, runner (answers and speed test), analysis and figure
data/           sample.json
results/        answers/<model>-<level>.jsonl (first-token letter log-probabilities and prompt tokens per question
                and language), bench/<model>-<level>.jsonl (load, memory and per-prompt timings;
                `.superseded-<time>` is a measurement replaced by `uaquant rebench`), models/ (Ollama tag, digest
                and size of every file; `parent_model` is copied from Ollama and holds a path on the machine that
                built the file), pilot/ (the pilot of 30 question pairs outside the test sample, run with an earlier
                version of the runner before the plan was fixed and kept as run), summary.json
figures/        figure drawn from summary.json (the figure in the paper is redrawn from the same numbers)
tests/          unit tests
PROTOCOL.md     analysis plan fixed before the test runs, with the changes made during them
```

## Usage

Python 3.10+, [uv](https://docs.astral.sh/uv/) and a running [Ollama](https://ollama.com) (the results were obtained
with Ollama 0.34.3).

```bash
uv sync --extra dev                  # the versions pinned in uv.lock
uv run pytest
uv run uaquant analyze               # summary.json and the figure from the stored outputs
uv run uaquant run test              # resumes the runs; pulls each model file, runs it and removes it again
```

`uaquant analyze` needs no model and no Ollama server. `uaquant run test` pulls every model file before it checks
which answers are already stored, so with the complete results it still downloads all 15 files (53 GB in total), and it
stops if the Ollama version or a model digest differs from the ones recorded in `results/models/`. Timings depend on
the machine and on other load; run the speed test on an idle machine. `uaquant run pilot` draws 8 questions per
category and appends to `results/pilot/`, so move the archived pilot aside before a new one.

## License

Code: MIT (`LICENSE`). Data and results (`data/sample.json`, `results/`, `figures/`): CC BY 4.0
(`LICENSES/CC-BY-4.0.txt`). Global-MMLU (Apache-2.0) and the models keep their own licences, see
`THIRD_PARTY_NOTICES.md`.
