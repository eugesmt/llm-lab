# Task Manager MCP integration status

Status: implementation and scoped validation complete.

Implemented:

- Official Python MCP SDK 1.x client over Streamable HTTP.
- Opt-in `use_taskmanager_mcp` chat flag and UI checkbox.
- Provider-neutral tool/message schemas and OpenAI-compatible tool-call mapping.
- Bounded LLM -> MCP -> LLM loop with advertised-tool allowlist.
- Aggregated token usage and `mcp_tools_used` response evidence.
- Configuration, documentation, unit tests, and an opt-in live MCP contract test.

Validated:

- Live connection to `http://127.0.0.1:8090/mcp`.
- Tool discovery returned all three expected OS tools.
- `get_system_overview` returned current Task Manager data.
- MCP agent and provider adapter unit tests pass.

Known environment issue:

- The repository's existing synchronous FastAPI `TestClient` tests hang because Starlette 1.6 selects installed `httpx2` 2.12. This also reproduces on the pre-existing `test_chat_success`; it is not specific to MCP routing. MCP service/API wiring is tested directly without network, and the real MCP transport contract passes.
