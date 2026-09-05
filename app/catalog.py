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
    # Free OpenRouter model (rate-limited instead of priced). Add more
    # OpenRouter models here as "<vendor>/<model>:free" when needed.
    ModelInfo(
        id="inclusionai/ling-3.0-flash-sante:free",
        provider="openrouter",
        display_name="Ling 3.0 Flash Sante (free)",
        thinking_supported=True,
        reasoning_effort=[],
    ),
]
