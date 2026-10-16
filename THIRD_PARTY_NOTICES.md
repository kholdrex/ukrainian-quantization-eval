# Third-party data and models

`results/answers/*.jsonl` store, for every question, its Global-MMLU id, the language and the model's first-token
probabilities; `data/sample.json` lists the ids of the sampled questions. The question texts are not stored; they are
downloaded from Global-MMLU at the pinned revision.

- Global-MMLU (https://huggingface.co/datasets/CohereLabs/Global-MMLU, revision 0e619db), published by Cohere Labs
  under the Apache License, Version 2.0 (`LICENSES/Apache-2.0.txt`). It is derived from MMLU
  (https://github.com/hendrycks/test), MIT licence, Copyright (c) 2020 Dan Hendrycks (`LICENSES/MIT-MMLU.txt`).

The results contain outputs of the following models (the first token and the log-probabilities of the letters A–D
for each question, and timings); no model weights are distributed. The models were obtained from the Ollama library
and are subject to their own licences:

- Qwen2.5 3B Instruct: Qwen Research License Agreement
  (https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE).
- Llama 3.2 3B Instruct: Llama 3.2 Community License Agreement
  (https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE).
- Gemma 3 4B IT: Gemma Terms of Use (https://ai.google.dev/gemma/terms).

The MIT licence of this repository (`LICENSE`) covers the code; the data and results (`data/sample.json`, `results/`,
`figures/`) are licensed under CC BY 4.0 (`LICENSES/CC-BY-4.0.txt`).
