from __future__ import annotations

import json
import re

import requests

from config.settings import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


class LLMClient:
    """OpenAI-compatible client. No local template fallback."""

    def __init__(self, api_key: str = LLM_API_KEY, base_url: str = LLM_BASE_URL, model: str = LLM_MODEL):
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or "").rstrip("/")
        self.model = (model or "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def require_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError("未配置 LLM_API_KEY，系统已停止执行（本版本不使用本地模板兜底）。")

    def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 800) -> str:
        self.require_enabled()
        if not self.base_url:
            raise RuntimeError("LLM_BASE_URL 为空。")
        if not self.model:
            raise RuntimeError("LLM_MODEL 为空。")

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        if resp.status_code >= 400:
            raise RuntimeError(f"LLM 调用失败：HTTP {resp.status_code}，{resp.text[:800]}")
        data = resp.json()
        if "error" in data:
            err = data["error"]
            msg = err.get("message", err) if isinstance(err, dict) else err
            raise RuntimeError(f"LLM 调用失败：{msg}")

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("LLM 返回空 choices。")

        message = choices[0].get("message") or {}
        content = self._extract_message_content(message)
        if not content:
            finish = choices[0].get("finish_reason", "unknown")
            raise RuntimeError(f"LLM 未返回可用文本（finish_reason={finish}）。请增大 max_tokens 或更换模型。")
        return content

    @staticmethod
    def _extract_message_content(message: dict) -> str:
        content = (message.get("content") or "").strip()
        if content:
            return content
        reasoning = (message.get("reasoning_content") or "").strip()
        if not reasoning:
            return ""
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", reasoning, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()
        if "{" in reasoning:
            start = reasoning.find("{")
            end = reasoning.rfind("}")
            if end > start:
                return reasoning[start : end + 1].strip()
        return ""

    @staticmethod
    def extract_json(content: str) -> dict:
        text = content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                raise ValueError("LLM 返回内容不是合法 JSON") from None
            return json.loads(m.group(0))
