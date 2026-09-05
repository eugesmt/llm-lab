"""Single source of truth for available models.

The frontend never hardcodes models — it reads this catalog via GET /api/models.
Capabilities below reflect provider documentation as of 2026-09.

Notes:
* DeepSeek V4 models: thinking on by default; `reasoning_effort` effective
  values are low/high/max (requested medium/xhigh map to high server-side).
* OpenAI: reasoning_effort support differs per model (gpt-6-astra has no
  `none`; the gpt-5.6 family does).
"""

from app.schemas.model import ModelInfo

MODEL_CATALOG: list[ModelInfo] = [
    ModelInfo(
        id="deepseek-v4-flash",
        provider="deepseek",
        display_name="DeepSeek V4 Flash",
        thinking_supported=True,
        reasoning_effort=["low", "high", "max"],
    ),
    ModelInfo(
        id="deepseek-v4-pro",
        provider="deepseek",
        display_name="DeepSeek V4 Pro",
        thinking_supported=True,
        reasoning_effort=["low", "high", "max"],
    ),
    ModelInfo(
        id="gpt-6-astra",
        provider="openai",
        display_name="GPT-6 Astra",
        thinking_supported=True,
        reasoning_effort=["low", "medium", "high", "xhigh", "max"],
    ),
    ModelInfo(
        id="gpt-5.6-sol",
        provider="openai",
        display_name="GPT-5.6 Sol",
        thinking_supported=True,
        reasoning_effort=["none", "low", "medium", "high", "xhigh", "max"],
    ),
    ModelInfo(
        id="gpt-5.6-terra",
        provider="openai",
        display_name="GPT-5.6 Terra",
        thinking_supported=True,
        reasoning_effort=["none", "low", "medium", "high", "xhigh", "max"],
    ),
    ModelInfo(
        id="gpt-5.6-luna",
        provider="openai",
        display_name="GPT-5.6 Luna",
        thinking_supported=True,
        reasoning_effort=["none", "low", "medium", "high", "xhigh", "max"],
    ),
]
