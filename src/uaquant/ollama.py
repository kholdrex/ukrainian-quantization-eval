"""Minimal client for the local Ollama server."""

import json
import time

import requests

HOST = "http://localhost:11434"
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 2048}


def answer(model: str, messages: list[dict], top: int = 20) -> dict:
    """One greedy token with its top alternatives."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "logprobs": True,
        "top_logprobs": top,
        "options": {**OPTIONS, "num_predict": 1},
    }
    response = requests.post(f"{HOST}/api/chat", json=body, timeout=600)
    response.raise_for_status()
    return response.json()


def stream(model: str, messages: list[dict], tokens: int, attempts: int = 3) -> dict:
    """Generate up to `tokens` tokens, recording the wall-clock time to the first token; a stream that breaks off
    is repeated."""
    for attempt in range(attempts):
        try:
            return {**stream_once(model, messages, tokens), "attempts": attempt + 1}
        except (RuntimeError, requests.RequestException):
            if attempt == attempts - 1:
                raise
            time.sleep(5)


def stream_once(model: str, messages: list[dict], tokens: int) -> dict:
    body = {"model": model, "messages": messages, "stream": True, "options": {**OPTIONS, "num_predict": tokens}}
    start = time.perf_counter()
    first = None
    with requests.post(f"{HOST}/api/chat", json=body, stream=True, timeout=600) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            chunk = json.loads(line)
            if "error" in chunk:
                return {"error": chunk["error"], "ttft": first, "wall": time.perf_counter() - start}
            if first is None and chunk.get("message", {}).get("content"):
                first = time.perf_counter() - start
            if chunk.get("done"):
                chunk["ttft"] = first
                chunk["wall"] = time.perf_counter() - start
                return chunk
    raise RuntimeError("stream ended without a final chunk")


def loaded() -> list[dict]:
    response = requests.get(f"{HOST}/api/ps", timeout=60)
    response.raise_for_status()
    return response.json()["models"]


def unload(model: str) -> None:
    requests.post(f"{HOST}/api/generate", json={"model": model, "keep_alive": 0}, timeout=600).raise_for_status()
    while any(m["name"] == model for m in loaded()):
        time.sleep(0.5)


def pull(model: str) -> None:
    response = requests.post(f"{HOST}/api/pull", json={"model": model, "stream": False}, timeout=7200)
    response.raise_for_status()


def remove(model: str) -> None:
    requests.delete(f"{HOST}/api/delete", json={"model": model}, timeout=600).raise_for_status()


def installed() -> dict[str, dict]:
    response = requests.get(f"{HOST}/api/tags", timeout=60)
    response.raise_for_status()
    return {m["name"]: m for m in response.json()["models"]}


def version() -> str:
    return requests.get(f"{HOST}/api/version", timeout=60).json()["version"]
