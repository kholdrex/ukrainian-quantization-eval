# Analysis plan

Fixed on 2026-10-04 after a pilot of one model and format (Qwen2.5 3B, Q4_K_M) on 30 Global-MMLU question pairs
(60 prompts) outside the test sample, before any test call.

## Question
For small open models run locally, does lowering the precision of the weights (GGUF quantisation) cost more accuracy
on Ukrainian questions than on the same questions in English, and what does each precision level give in speed and
memory on a laptop?

## Items
Global-MMLU (revision 0e619db), parallel `en`/`uk` test questions: 300 per subject category, seed 2026; repeated
questions kept once. The Ukrainian questions are translations of the English originals.

Correction made during the first test run, before any result was analysed: the plan first said "4 categories, 1200
pairs"; Global-MMLU has 6 subject categories, so the sample has 1800 pairs.

## Models and formats
Ollama builds of the same checkpoints: Qwen2.5 3B Instruct and Llama 3.2 3B Instruct at F16, Q8_0, Q6_K, Q5_K_M,
Q4_K_M and Q3_K_M; Gemma 3 4B IT at F16, Q8_0 and Q4_K_M (the only formats published for it). Greedy decoding,
seed 0, context 2048, everything else at Ollama defaults; one Ollama version for the whole run.

## Outcome
The model generates one token; among the 20 most likely first tokens, the answer is the letter A–D with the highest
log-probability (" A" and "A" merged). A reply whose top token is not a letter is counted as a format failure and
its rate is reported; the letter rule still gives an answer for it. If no letter is among the 20, the answer is
wrong.

## Analyses
1. Accuracy for every model × format × language with 95% bootstrap intervals over questions (10000 resamples).
2. Primary: for each model and quantised format q, the language difference of the accuracy change relative to F16,
   D_q = (A_uk,q − A_uk,F16) − (A_en,q − A_en,F16), with a paired bootstrap interval over questions (both languages
   of a question resampled together) and a two-sided bootstrap p-value; Holm correction over the 12 contrasts (the text first said 13; Gemma 3 has two quantised levels).
3. Answer changes against F16 per language: share of questions whose answer differs, correct→wrong and wrong→correct
   counts; mean KL divergence between the F16 and quantised letter distributions.
4. Speed and memory per format on an Apple M1 Pro (16 GB): model load time, memory reported by Ollama, time to first
   token, prompt processing and generation rates (tokens/s) for 30 questions per language, each prompt once, with a
   generation limited to 128 tokens; number of prompt tokens per question in each language.
5. By subject category (descriptive).

## Not claimed
Results hold for these checkpoints, this runtime and this machine; no claim about other quantisation methods
(GPTQ, AWQ), larger models or free-form generation quality.

## Changes during the runs

- A generation that Ollama stops because of endless token repetition ("token repeat limit reached") ends the stream
  with an error instead of a final chunk; such generations are recorded as aborted in the speed test and excluded
  from the speed medians (this happened only for Qwen2.5 Q3_K_M with Ukrainian prompts).
- The speed of Qwen2.5 Q5_K_M was measured again on an idle machine (`uaquant rebench`), because another
  CPU-intensive job ran during the first measurement; the accuracy runs are not affected.
- Exploratory, added after the main results: the share of changed answers grouped by the F16 probability of the
  chosen letter (below 0.9, 0.9–0.99, at least 0.99).
- Corrections of the wording above, made after the runs: the pilot covered 30 question pairs (60 prompts), not
  "40 questions"; the speed test limits generation to 128 tokens rather than fixing its length; the Holm family has
  12 contrasts.
