"""Thin client for the local vLLM OpenAI-compatible endpoint (non-streaming helper; the server streams itself)."""
import requests
from common import VLLM_URL, LLM_NAME, LLM_REP


def chat(messages, max_tokens=220, temperature=0.0, stream=False):
    body = {"model": LLM_NAME, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "repetition_penalty": LLM_REP, "stream": stream}      # temperature is explicit: the model's own
    if stream:                                                    # generation_config would otherwise force 0.6
        return requests.post(f"{VLLM_URL}/chat/completions", json=body, stream=True, timeout=600)
    r = requests.post(f"{VLLM_URL}/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"], j["usage"]
