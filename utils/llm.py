from __future__ import annotations

import requests
from config.settings import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


class LLMClient:
    """OpenAI-compatible client.

    Strict version: no local-template fallback. If there is no API key or the
    provider call fails, it raises an exception so the UI shows the real error.
    """

    def __init__(self, api_key: str = LLM_API_KEY, base_url: str = LLM_BASE_URL, model: str = LLM_MODEL):
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or "").rstrip("/")
        self.model = (model or "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def require_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError("未配置 LLM_API_KEY / DEEPSEEK_API_KEY，已停止执行；本版本不使用本地模板兜底。")

    def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 800) -> str:
        self.require_enabled()
        if not self.base_url:
            raise RuntimeError("LLM_BASE_URL 为空。")
        if not self.model:
            raise RuntimeError("LLM_MODEL 为空。")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code >= 400:
            raise RuntimeError(f"LLM 调用失败：HTTP {resp.status_code}，{resp.text[:800]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"LLM 返回格式异常：{data}") from exc
