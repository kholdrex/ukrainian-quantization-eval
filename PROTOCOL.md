# Analysis plan

Fixed on 2026-10-04 after a pilot of one model and format (Qwen2.5 3B, Q4_K_M) on 40 Global-MMLU questions outside
the test sample, before any test call.

## Question
For small open models run locally, does lowering the precision of the weights (GGUF quantisation) cost more accuracy
on Ukrainian questions than on the same questions in English, and what does each precision level give in speed and
memory on a laptop?

## Items
Global-MMLU (revision 0e619db), parallel `en`/`uk` test questions: 300 per subject category (4 categories, 1200
pairs), seed 2026; repeated questions kept once. The Ukrainian questions are translations of the English originals.

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
   of a question resampled together) and a two-sided bootstrap p-value; Holm correction over the 13 contrasts.
3. Answer changes against F16 per language: share of questions whose answer differs, correct→wrong and wrong→correct
   counts; mean KL divergence between the F16 and quantised letter distributions.
4. Speed and memory per format on an Apple M1 Pro (16 GB): model load time, memory reported by Ollama, time to first
   token, prompt processing and generation rates (tokens/s) for 30 questions per language, each prompt once, with a
   fixed 128-token generation; number of prompt tokens per question in each language.
5. By subject category (descriptive).

## Not claimed
Results hold for these checkpoints, this runtime and this machine; no claim about other quantisation methods
(GPTQ, AWQ), larger models or free-form generation quality.
