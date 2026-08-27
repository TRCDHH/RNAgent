"""LLM 工厂（DeepSeek OpenAI 兼容接口）。

配置见 app/config.py（环境变量 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL）。
"""

from langchain_openai import ChatOpenAI

from app import config


def llm_configured() -> bool:
    return bool(config.LLM_API_KEY)


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        model=config.LLM_MODEL,
        temperature=0.7,
        timeout=120,
    )
