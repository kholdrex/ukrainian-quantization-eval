# Third-party data

`results/answers/*.jsonl` store, for every question, its Global-MMLU id, the language and the model's first-token
probabilities; `data/sample.json` lists the ids of the sampled questions. The question texts are not stored; they are
downloaded from Global-MMLU at the pinned revision.

- Global-MMLU (https://huggingface.co/datasets/CohereLabs/Global-MMLU, revision 0e619db), published by Cohere Labs
  under the Apache License, Version 2.0 (`LICENSES/Apache-2.0.txt`). It is derived from MMLU
  (https://github.com/hendrycks/test), MIT licence, Copyright (c) 2020 Dan Hendrycks (`LICENSES/MIT-MMLU.txt`).

The MIT licence of this repository covers the code.
