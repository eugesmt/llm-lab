"""Single source of truth for available models.

The frontend never hardcodes models — it reads this catalog via GET /api/models.
Capabilities below reflect provider documentation as of 2026-09.

DeepSeek V4 notes:
* thinking is enabled by default; `reasoning_effort` effective values are
  low/high/max (requested medium/xhigh map to high server-side);
* the special value "none" disables thinking entirely — the adapter sends
  thinking: {"type": "disabled"} instead of the effort parameter.
"""

from app.schemas.model import ModelInfo

MODEL_CATALOG: list[ModelInfo] = [
    ModelInfo(
        id="qwen3:4b",
        provider="ollama",
        display_name="Qwen3 4B (local Ollama)",
        thinking_supported=False,
        reasoning_effort=[],
    ),
    ModelInfo(
        id="deepseek-v4-flash",
        provider="deepseek",
        display_name="DeepSeek V4 Flash",
        thinking_supported=True,
        reasoning_effort=["none", "low", "high", "max"],
    ),
    ModelInfo(
        id="deepseek-v4-pro",
        provider="deepseek",
        display_name="DeepSeek V4 Pro",
        thinking_supported=True,
        reasoning_effort=["none", "low", "high", "max"],
    ),
    # Free OpenRouter models (rate-limited instead of priced). Full list:
    # https://openrouter.ai/models?variant=free  (add as "<vendor>/<model>:free").
    ModelInfo(
        id="inclusionai/ling-3.0-flash-sante:free",
        provider="openrouter",
        display_name="Ling 3.0 Flash Sante (free)",
        thinking_supported=True,
        reasoning_effort=[],
    ),
    ModelInfo(
        id="liquid/lfm-2.5-2.6b:free",
        provider="openrouter",
        display_name="LFM 2.5 2.6B (free)",
        thinking_supported=False,
        reasoning_effort=[],
    ),
    ModelInfo(
        id="google/gemma-4-31b-it:free",
        provider="openrouter",
        display_name="Gemma 4 31B (free)",
        thinking_supported=False,
        reasoning_effort=[],
    ),
    ModelInfo(
        id="google/gemma-4-26b-a4b-it:free",
        provider="openrouter",
        display_name="Gemma 4 26B A4B IT (free)",
        thinking_supported=True,
        reasoning_effort=[],
    ),
    ModelInfo(
        id="minimax/minimax-m3:free",
        provider="openrouter",
        display_name="MiniMax M3 (free)",
        thinking_supported=True,
        reasoning_effort=[],
    ),
    ModelInfo(
        id="nvidia/nemotron-3-ultra-550b-a55b:free",
        provider="openrouter",
        display_name="Nemotron 3 Ultra 550B (free)",
        thinking_supported=True,
        reasoning_effort=[],
    ),
    ModelInfo(
        id="z-ai/glm-5.2:free",
        provider="openrouter",
        display_name="GLM 5.2 (free)",
        thinking_supported=True,
        reasoning_effort=[],
    ),
]
