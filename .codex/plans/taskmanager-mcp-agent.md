Goal
Connect llm-lab to the Task Manager Streamable HTTP MCP endpoint and let the selected LLM use its OS-information tools during chat.

Root cause / context
llm-lab currently sends a single user message to an OpenAI-compatible provider and accepts only text completions. It has no MCP client, provider-neutral tool schemas, or model/tool execution loop.

Files / symbols
- app/mcp/client.py: MCP client protocol and Streamable HTTP adapter.
- app/schemas/llm.py: provider-neutral messages, tools, tool calls, and completion turns.
- app/providers/{base,_openai_compat,deepseek,openai,openrouter,mock}.py: tool-aware completion interface and SDK mapping.
- app/services/llm.py: bounded model -> MCP tool -> model loop.
- app/{config,main}.py and app/api/chat.py, app/schemas/chat.py: configuration and request wiring.
- frontend/{index.html,app.js}: opt-in MCP control.
- tests/: MCP client, agent loop, API, and provider mappings.
- pyproject.toml, uv.lock, README.md, .env.example: dependency and usage documentation.

Changes
- Add the official Python MCP SDK and a Streamable HTTP client wrapper.
- Keep ordinary chat behavior unchanged; enable Task Manager tools with an explicit chat flag.
- Expose MCP tool JSON Schemas to providers, execute only advertised tool names, return MCP content to the model, and cap the number of rounds.
- Preserve the provider-SDK boundary: OpenAI SDK objects remain under app/providers and MCP SDK objects under app/mcp.
- Report MCP connectivity/protocol/tool failures as normalized application errors without leaking internals.

Tests
- Unit-test tool-aware provider request/response mapping.
- Unit-test successful multi-turn MCP flow, ordinary-chat compatibility, rejected unknown tools, errors, and loop limit.
- Test the HTTP API flag with fakes; no real API keys or network.
- Run pytest and ruff for the repository.

Risks
- Provider-specific differences in tool-call message payloads, especially DeepSeek reasoning content.
- MCP endpoint unavailable or returning non-text content.
- Unbounded/repeated model tool calls; enforce a strict round limit and advertised-tool allowlist.

Do not touch
- Benchmark semantics.
- Model catalog/pricing data.
- Secret files or API keys.
