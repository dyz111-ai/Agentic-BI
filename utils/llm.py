from __future__ import annotations

import json
import os
import re

import requests


def _get_llm_config():
    """Get LLM configuration from environment variables."""
    from pathlib import Path
    base_dir = Path(__file__).resolve().parents[1]
    env_file = base_dir / ".env"
    
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())
    
    return {
        "api_key": os.getenv("LLM_API_KEY", ""),
        "base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        "model": os.getenv("LLM_MODEL", "deepseek-chat"),
    }


_llm_config = _get_llm_config()


class LLMClient:
    """OpenAI-compatible client. No local template fallback."""

    def __init__(
        self, 
        api_key: str | None = None, 
        base_url: str | None = None, 
        model: str | None = None
    ):
        config = _llm_config
        self.api_key = (api_key or config["api_key"] or "").strip()
        self.base_url = (base_url or config["base_url"] or "").rstrip("/")
        self.model = (model or config["model"] or "").strip()

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
        token_budgets = []
        for t in (max_tokens, max(max_tokens * 2, 2048), 8192):
            if t not in token_budgets:
                token_budgets.append(t)

        last_error = "LLM 未返回可用文本"
        for budget in token_budgets:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": budget,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code >= 400:
                last_error = f"LLM 调用失败：HTTP {resp.status_code}，{resp.text[:800]}"
                continue
            data = resp.json()
            if "error" in data:
                err = data["error"]
                msg = err.get("message", err) if isinstance(err, dict) else err
                last_error = f"LLM 调用失败：{msg}"
                continue

            choices = data.get("choices") or []
            if not choices:
                last_error = "LLM 返回空 choices。"
                continue

            message = choices[0].get("message") or {}
            content = self._extract_message_content(message)
            if content:
                return content
            finish = choices[0].get("finish_reason", "unknown")
            last_error = f"LLM 未返回可用文本（finish_reason={finish}，max_tokens={budget}）。"

        raise RuntimeError(f"{last_error} 请增大 max_tokens 或更换模型。")

    @staticmethod
    def _extract_message_content(message: dict) -> str:
        content = (message.get("content") or "").strip()
        if content:
            return content
        reasoning = (message.get("reasoning_content") or "").strip()
        if not reasoning:
            return ""
        fenced = re.search(r"```(?:json)?\s*(\{.*)\s*```?", reasoning, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()
        if "{" in reasoning:
            start = reasoning.find("{")
            fragment = reasoning[start:].strip()
            if fragment:
                return fragment
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
