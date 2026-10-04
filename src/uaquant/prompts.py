"""Prompts for answering a question with one letter and for the fixed-length generation benchmark."""

from .data import LETTERS, Item

ANSWER = {
    "en": "Answer the following multiple-choice question. Reply with a single letter: A, B, C or D.",
    "uk": "Дайте відповідь на запитання з варіантами відповіді. Відповідайте однією літерою: A, B, C або D.",
}
EXPLAIN = {
    "en": "Answer the following multiple-choice question and explain your reasoning in detail.",
    "uk": "Дайте відповідь на запитання з варіантами відповіді та докладно поясніть своє міркування.",
}


def question(item: Item, lang: str) -> str:
    stem, options = item.text[lang]
    lines = [stem] + [f"{letter}. {option}" for letter, option in zip(LETTERS, options, strict=True)]
    return "\n".join(lines)


def messages(item: Item, lang: str, task: str = "answer") -> list[dict]:
    system = (ANSWER if task == "answer" else EXPLAIN)[lang]
    return [{"role": "system", "content": system}, {"role": "user", "content": question(item, lang)}]


def letter_logprobs(top: list[dict]) -> dict[str, float]:
    """Highest log-probability of each answer letter among the first-token alternatives (' A' and 'A' merged)."""
    out: dict[str, float] = {}
    for alt in top:
        token = alt["token"].strip()
        if token in LETTERS and len(token) == 1:
            out[token] = max(out.get(token, float("-inf")), alt["logprob"])
    return out
