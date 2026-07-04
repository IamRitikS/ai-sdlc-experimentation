"""Central config loaded from environment / .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

load_dotenv()

OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./output")
MAX_ITER: int = int(os.getenv("MAX_ITER", "3"))
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
CHECKPOINT_DB: str = os.getenv("CHECKPOINT_DB", "./sdlc_checkpoints.db")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


ZLLAMA_BASE_URL: str = os.getenv("ZLLAMA_BASE_URL", "https://zllama.corp.zscaler.com/v1/")
ZLLAMA_API_KEY: str = os.getenv("ZLLAMA_API_KEY", "")


def get_llm(agent_prefix: str, *, temperature: float = 0.2):
    """Return a LangChain chat model for the given agent, configured via env vars.

    Reads  <AGENT_PREFIX>_PROVIDER  and  <AGENT_PREFIX>_MODEL  from env.
    Supported providers: openai, anthropic, zllama
    Falls back to zllama if not set.
    """
    prefix = agent_prefix.upper()
    provider = os.getenv(f"{prefix}_PROVIDER", "zllama").lower()
    model = os.getenv(f"{prefix}_MODEL", "gpt-4o")

    if provider == "anthropic":
        return ChatAnthropic(model=model, temperature=temperature)

    if provider == "zllama":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=ZLLAMA_BASE_URL,
            api_key=ZLLAMA_API_KEY,
        )

    return ChatOpenAI(model=model, temperature=temperature)
