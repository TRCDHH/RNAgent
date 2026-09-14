"""LLM 工厂（DeepSeek OpenAI 兼容接口）。

配置见 app/config.py（环境变量 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL）。
"""

from langchain_openai import ChatOpenAI

from app import config


def llm_configured() -> bool:
    return bool(config.LLM_API_KEY)


def get_llm(temperature: float = 0.7, model: str = None) -> ChatOpenAI:
    """创建 LLM 实例。

    temperature/model 可覆盖：预处理 ReAct 决策层需要 temperature=0（可复现）并可能单独
    指定更强的模型；报告与聊天助手沿用默认值，行为不变。
    """
    return ChatOpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        model=model or config.LLM_MODEL,
        temperature=temperature,
        timeout=120,
    )
