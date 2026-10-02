/* LLM Lab frontend logic. Models always come from GET /api/models. */

"use strict";

const EFFORT_ORDER = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];
const CONVERSATION_STORAGE_KEY = "llm-lab.conversation-id";
const dash = "—";

let models = [];
let conversationId = null;
let taskState = {};
let sending = false;

function $(id) { return document.getElementById(id); }

async function api(path, options) {
  const response = await fetch(path, options);
  let body = null;
  try { body = await response.json(); } catch { /* non-JSON error body */ }
  if (!response.ok) {
    const message = body && body.error && body.error.message
      ? body.error.message
      : body && typeof body.detail === "string"
        ? body.detail
      : `HTTP ${response.status}`;
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return body;
}

function setStatus(message, isError = false) {
  const status = $("chat-status");
  status.textContent = message || "";
  status.classList.toggle("error", isError);
  status.hidden = !message;
}

/* ---------- tabs ---------- */

function showTab(tab) {
  const isChat = tab === "chat";
  $("panel-chat").hidden = !isChat;
  $("panel-benchmark").hidden = isChat;
  $("tab-chat").classList.toggle("active", isChat);
  $("tab-benchmark").classList.toggle("active", !isChat);
}

/* ---------- model catalog ---------- */

function selectedModel() {
  return models.find((model) => model.id === $("chat-model").value) || null;
}

function populateEffortSelect(select, efforts) {
  select.replaceChildren();
  const sorted = [...efforts].sort(
    (a, b) => EFFORT_ORDER.indexOf(a) - EFFORT_ORDER.indexOf(b),
  );
  for (const effort of sorted) {
    const option = document.createElement("option");
    option.value = effort;
    option.textContent = effort;
    select.appendChild(option);
  }
  const label = select.closest("label");
  if (label) label.hidden = sorted.length === 0;
  select.disabled = sorted.length === 0;
}

function onChatModelChange() {
  const model = selectedModel();
  const badge = $("chat-provider");
  if (!model) {
    badge.hidden = true;
    return;
  }
  badge.textContent = `provider: ${model.provider}`;
  badge.hidden = false;
  populateEffortSelect($("chat-effort"), model.reasoning_effort);
}

function checkedBenchModels() {
  return models.filter((model) => {
    const checkbox = $(`bench-model-${model.id}`);
    return checkbox && checkbox.checked;
  });
}

function onBenchSelectionChange() {
  const checked = checkedBenchModels();
  let common = null;
  for (const model of checked) {
    const efforts = new Set(model.reasoning_effort);
    common = common === null
      ? efforts
      : new Set([...common].filter((effort) => efforts.has(effort)));
  }
  populateEffortSelect($("bench-effort"), common ? [...common] : []);
}

function renderCatalog() {
  const chatSelect = $("chat-model");
  chatSelect.replaceChildren();
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.display_name;
    chatSelect.appendChild(option);
  }

  const benchBox = $("bench-models");
  benchBox.replaceChildren();
  for (const model of models) {
    const item = document.createElement("div");
    item.className = "item";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = `bench-model-${model.id}`;
    checkbox.value = model.id;
    checkbox.addEventListener("change", onBenchSelectionChange);

    const label = document.createElement("label");
    label.htmlFor = checkbox.id;
    label.textContent = model.display_name;

    const tag = document.createElement("span");
    tag.className = "provider-tag";
    tag.textContent = `(${model.provider})`;

    item.append(checkbox, label, tag);
    benchBox.appendChild(item);
  }

  onChatModelChange();
  onBenchSelectionChange();
}

async function loadModels() {
  try {
    models = await api("/api/models");
    renderCatalog();
  } catch (err) {
    setStatus(`Не удалось загрузить список моделей: ${err.message}`, true);
  }
}

/* ---------- task memory ---------- */

function normalizeTaskState(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function memoryItems(value) {
  if (Array.isArray(value)) return value.map(String).filter((item) => item.trim());
  if (value && typeof value === "object") {
    return Object.entries(value).map(([key, item]) => `${key}: ${item}`).filter((item) => item.trim());
  }
  if (value === null || value === undefined || value === "") return [];
  return [String(value)];
}

function renderMemoryList(element, values) {
  element.replaceChildren();
  const items = memoryItems(values);
  if (items.length === 0) {
    const empty = document.createElement("li");
    empty.textContent = dash;
    element.appendChild(empty);
    return;
  }
  for (const value of items) {
    const item = document.createElement("li");
    item.textContent = value;
    element.appendChild(item);
  }
}

function renderTaskMemory(value) {
  taskState = normalizeTaskState(value);
  $("memory-goal").textContent = taskState.goal || taskState.objective || dash;
  renderMemoryList($("memory-constraints"), taskState.constraints);
  renderMemoryList($("memory-terms"), taskState.terms);
  renderMemoryList($("memory-clarifications"), taskState.clarifications || taskState["уточнения"]);
}

/* ---------- conversation normalization and rendering ---------- */

function normalizeConversationPayload(payload) {
  const nested = payload && payload.conversation && typeof payload.conversation === "object"
    ? payload.conversation
    : {};
  const id = payload?.conversation_id || nested.id || payload?.id || null;
  const messages = Array.isArray(payload?.messages)
    ? payload.messages
    : Array.isArray(nested.messages) ? nested.messages : [];
  const state = payload?.task_state || nested.task_state || {};
  return { id, messages, taskState: normalizeTaskState(state) };
}

function normalizeStoredMessage(raw) {
  const message = raw && typeof raw === "object" ? raw : {};
  const role = message.role === "assistant" ? "assistant" : "user";
  const response = message.response || message.response_json || message.llm_response || message.chat_response
    || message.metadata?.response || (role === "assistant" ? message : null);
  const content = role === "assistant"
    ? (message.content ?? message.answer ?? response?.answer ?? "")
    : (message.content ?? message.prompt ?? message.text ?? "");
  return { role, content: String(content), response: role === "assistant" ? response : null };
}

function appendLabelValue(parent, label, value) {
  const line = document.createElement("div");
  const strong = document.createElement("strong");
  strong.textContent = `${label}: `;
  line.appendChild(strong);
  const text = document.createElement("span");
  text.textContent = value;
  line.appendChild(text);
  parent.appendChild(line);
}

function renderSources(parent, response) {
  const section = document.createElement("section");
  section.className = "message-artifact";
  const title = document.createElement("strong");
  title.textContent = "Источники RAG";
  section.appendChild(title);

  const list = document.createElement("ul");
  const sources = Array.isArray(response?.rag_sources) ? response.rag_sources : [];
  if (sources.length === 0) {
    const item = document.createElement("li");
    item.textContent = response?.rag_mode === "off" ? "Не использовались (режим без RAG)" : "Релевантные источники не найдены";
    list.appendChild(item);
  } else {
    for (const source of sources) {
      const item = document.createElement("li");
      const sectionName = source.section ? ` · ${source.section}` : "";
      const score = Number.isFinite(Number(source.similarity))
        ? ` · similarity ${Number(source.similarity).toFixed(4)}` : "";
      const rerank = source.rerank_score === null || source.rerank_score === undefined
        ? "" : ` · итоговый score ${Number(source.rerank_score).toFixed(4)}`;
      item.textContent = `${source.source || dash}${sectionName} · ${source.chunk_id || dash}${score}${rerank}`;
      list.appendChild(item);
    }
  }
  section.appendChild(list);
  parent.appendChild(section);
}

function renderCitations(parent, response) {
  const section = document.createElement("section");
  section.className = "message-artifact";
  const title = document.createElement("strong");
  title.textContent = "Цитаты";
  section.appendChild(title);

  const list = document.createElement("ol");
  const citations = Array.isArray(response?.citations) ? response.citations : [];
  if (citations.length === 0) {
    const item = document.createElement("li");
    item.textContent = "Цитаты отсутствуют";
    list.appendChild(item);
  } else {
    for (const citation of citations) {
      const item = document.createElement("li");
      const metadata = document.createElement("div");
      metadata.textContent = `${citation.source || dash}${citation.section ? ` · ${citation.section}` : ""} · ${citation.chunk_id || dash}`;
      const quote = document.createElement("blockquote");
      quote.textContent = citation.quote || "Цитата отсутствует";
      item.append(metadata, quote);
      list.appendChild(item);
    }
  }
  section.appendChild(list);
  parent.appendChild(section);
}

function renderAssistantDetails(parent, response) {
  const details = document.createElement("div");
  details.className = "message-details";
  const isRag = response?.rag_mode && response.rag_mode !== "off";
  const grounding = isRag
    ? (response.grounded === true ? "Подтверждён найденными фрагментами" : "Не подтверждён")
    : "Не используется (режим без RAG)";
  const state = response?.unknown === true
    ? "unknown"
    : response?.grounded === true ? "grounded" : "не определено";
  appendLabelValue(details, "Состояние", state);
  appendLabelValue(details, "Статус подтверждения", grounding);
  appendLabelValue(details, "Режим", response?.rag_mode || "off");
  if (response?.unknown === true || response?.unknown_reason) {
    appendLabelValue(details, "Почему нет ответа", response.unknown_reason || "Релевантного контекста недостаточно");
  }
  renderSources(details, response);
  renderCitations(details, response);

  const artifacts = Array.isArray(response?.artifacts) ? response.artifacts : [];
  if (artifacts.length > 0) {
    const files = document.createElement("section");
    files.className = "message-artifact";
    const title = document.createElement("strong");
    title.textContent = "Файлы";
    files.appendChild(title);
    for (const artifact of artifacts) {
      try {
        const url = new URL(artifact.download_url);
        if (url.protocol !== "http:" && url.protocol !== "https:") continue;
        const link = document.createElement("a");
        link.href = url.href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = artifact.filename || artifact.report_id || url.href;
        files.appendChild(link);
      } catch { /* ignore malformed artifact links */ }
    }
    if (files.childElementCount > 1) details.appendChild(files);
  }

  const trace = Array.isArray(response?.mcp_trace) ? response.mcp_trace : [];
  if (trace.length > 0) {
    const traceSection = document.createElement("section");
    traceSection.className = "message-artifact";
    const title = document.createElement("strong");
    title.textContent = "MCP trace";
    traceSection.appendChild(title);
    const list = document.createElement("ol");
    for (const event of trace) {
      const item = document.createElement("li");
      item.textContent = [event.server, event.tool, event.alias, event.status]
        .filter((value) => value !== null && value !== undefined && value !== "")
        .map(String).join(" · ") || dash;
      list.appendChild(item);
    }
    traceSection.appendChild(list);
    details.appendChild(traceSection);
  }
  parent.appendChild(details);
}

function renderMessage(message) {
  const normalized = normalizeStoredMessage(message);
  const card = document.createElement("article");
  card.className = `chat-message ${normalized.role}`;

  const heading = document.createElement("div");
  heading.className = "message-role";
  heading.textContent = normalized.role === "assistant" ? "Ассистент" : "Вы";
  card.appendChild(heading);

  const content = document.createElement("pre");
  content.className = "message-content";
  content.textContent = normalized.content;
  card.appendChild(content);
  if (normalized.role === "assistant") renderAssistantDetails(card, normalized.response || {});
  return card;
}

function renderMessages(messages) {
  const container = $("chat-messages");
  container.replaceChildren();
  const normalized = Array.isArray(messages) ? messages.map(normalizeStoredMessage) : [];
  if (normalized.length === 0) {
    const empty = document.createElement("p");
    empty.className = "chat-empty";
    empty.textContent = "Начните диалог — задайте вопрос по документации.";
    container.appendChild(empty);
    return;
  }
  for (const message of normalized) container.appendChild(renderMessage(message));
  container.scrollTop = container.scrollHeight;
}

function appendConversationMessage(role, content, response = null) {
  const container = $("chat-messages");
  const empty = container.querySelector(".chat-empty");
  if (empty) empty.remove();
  container.appendChild(renderMessage({ role, content, response }));
  container.scrollTop = container.scrollHeight;
}

function renderConversationId() {
  $("chat-conversation-id").textContent = conversationId
    ? `Диалог: ${conversationId}` : "Новый диалог";
}

function setConversationId(id) {
  conversationId = id ? String(id) : null;
  if (conversationId) localStorage.setItem(CONVERSATION_STORAGE_KEY, conversationId);
  else localStorage.removeItem(CONVERSATION_STORAGE_KEY);
  renderConversationId();
}

function applyConversationPayload(payload) {
  const normalized = normalizeConversationPayload(payload);
  if (normalized.id) setConversationId(normalized.id);
  renderMessages(normalized.messages);
  renderTaskMemory(normalized.taskState);
  return normalized;
}

async function createConversation() {
  const data = await api("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  const normalized = applyConversationPayload(data);
  if (!normalized.id) throw new Error("API не вернул conversation_id");
  return normalized;
}

async function restoreConversation() {
  const savedId = localStorage.getItem(CONVERSATION_STORAGE_KEY);
  if (!savedId) {
    renderConversationId();
    renderTaskMemory({});
    return;
  }
  try {
    const data = await api(`/api/conversations/${encodeURIComponent(savedId)}`);
    const normalized = applyConversationPayload(data);
    if (!normalized.id) setConversationId(savedId);
  } catch (err) {
    if (err.status === 404) {
      setConversationId(null);
      renderMessages([]);
      renderTaskMemory({});
      setStatus("Старый диалог не найден. Начните новый.");
    } else {
      setStatus(`Не удалось восстановить диалог: ${err.message}`, true);
    }
  }
}

async function startNewConversation() {
  if (sending) return;
  const button = $("chat-new-conversation");
  button.disabled = true;
  setStatus("Создаём новый диалог…");
  try {
    await createConversation();
    renderMessages([]);
    renderTaskMemory({});
    $("chat-result").hidden = true;
    setStatus("Новый диалог создан.");
  } catch (err) {
    setStatus(`Не удалось создать новый диалог: ${err.message}`, true);
  } finally {
    button.disabled = false;
  }
}

/* ---------- latest response metrics ---------- */

function setCell(id, value) {
  $(id).textContent = value === null || value === undefined || value === "" ? dash : String(value);
}

function renderLatestResponse(data) {
  $("chat-answer").textContent = data.answer || dash;
  const usage = data.usage || {};
  setCell("m-provider", data.provider);
  setCell("m-model", data.model);
  setCell("m-latency", Number.isFinite(Number(data.latency_ms)) ? `${Math.round(data.latency_ms)} ms` : null);
  setCell("m-prompt-tokens", usage.prompt_tokens);
  setCell("m-completion-tokens", usage.completion_tokens);
  setCell("m-total-tokens", usage.total_tokens);
  setCell("m-reasoning-tokens", usage.reasoning_tokens);
  setCell("m-finish", data.finish_reason);
  setCell("m-mcp-tools", (data.mcp_tools_used || []).join(", "));
  $("chat-result").hidden = false;
}

/* ---------- chat ---------- */

function chatPayload(model, prompt) {
  const payload = { model: model.id, prompt, conversation_id: conversationId };
  payload.rag_mode = $("chat-rag-mode")?.value || "off";
  payload.mcp_servers = [];
  const taskmanagerCheckbox = $("chat-use-taskmanager-mcp");
  const obsidianCheckbox = $("chat-use-obsidian-mcp");
  if (taskmanagerCheckbox?.checked) payload.mcp_servers.push("taskmanager");
  if (obsidianCheckbox?.checked) payload.mcp_servers.push("obsidian");
  if ($("chat-effort").value && !$("chat-effort").disabled) {
    payload.reasoning_effort = $("chat-effort").value;
  }
  return payload;
}

async function sendChat() {
  if (sending) return;
  const model = selectedModel();
  const prompt = $("chat-prompt").value.trim();
  if (!model || !prompt) {
    setStatus("Выберите модель и введите сообщение.", true);
    return;
  }

  sending = true;
  $("chat-send").disabled = true;
  $("chat-new-conversation").disabled = true;
  setStatus("Обрабатываем вопрос и ищем контекст в документации…");
  try {
    if (!conversationId) await createConversation();
    appendConversationMessage("user", prompt);
    $("chat-prompt").value = "";

    const data = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(chatPayload(model, prompt)),
    });
    if (data.conversation_id) setConversationId(data.conversation_id);
    if (data.task_state) renderTaskMemory(data.task_state);
    appendConversationMessage("assistant", data.answer || dash, data);
    renderLatestResponse(data);
    setStatus("");
  } catch (err) {
    setStatus(`Ошибка: ${err.message}`, true);
  } finally {
    sending = false;
    $("chat-send").disabled = false;
    $("chat-new-conversation").disabled = false;
  }
}

/* ---------- benchmark ---------- */

function fmtCost(cost) {
  if (!cost || cost.total_cost === null || cost.total_cost === undefined) return dash;
  return `$${cost.total_cost.toFixed(6)}`;
}

function benchCell(row, value, cls) {
  const td = document.createElement("td");
  if (cls) td.className = cls;
  td.textContent = value;
  row.appendChild(td);
  return td;
}

async function runBenchmark() {
  const checked = checkedBenchModels();
  const prompt = $("bench-prompt").value.trim();
  const repeats = Number($("bench-repeats").value) || 1;
  const status = $("bench-status");

  if (checked.length === 0 || !prompt) {
    status.textContent = "Выберите хотя бы одну модель и введите prompt.";
    status.classList.add("error");
    status.hidden = false;
    return;
  }

  $("bench-run").disabled = true;
  status.textContent = "Benchmark выполняется (модели идут последовательно)…";
  status.classList.remove("error");
  status.hidden = false;
  $("bench-result").hidden = true;

  const payload = { prompt, models: checked.map((model) => model.id), repeats };
  if ($("bench-effort").value && !$("bench-effort").disabled) {
    payload.reasoning_effort = $("bench-effort").value;
  }

  try {
    const data = await api("/api/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const tbody = $("bench-body");
    tbody.replaceChildren();
    for (const result of data.results) {
      const row = document.createElement("tr");
      benchCell(row, result.model);
      benchCell(row, result.provider);
      if (result.error) {
        for (let i = 0; i < 7; i += 1) benchCell(row, dash, "error");
        benchCell(row, result.error, "error");
      } else {
        const usage = result.usage || {};
        benchCell(row, `${Math.round(result.latency_ms)}`);
        benchCell(row, usage.prompt_tokens ?? dash);
        benchCell(row, usage.completion_tokens ?? dash);
        benchCell(row, usage.reasoning_tokens ?? dash);
        benchCell(row, usage.total_tokens ?? dash);
        benchCell(row, fmtCost(result.cost));
        benchCell(row, result.finish_reason ?? dash);
        const td = document.createElement("td");
        const details = document.createElement("details");
        details.className = "answer-details";
        const summary = document.createElement("summary");
        summary.textContent = "показать";
        const pre = document.createElement("pre");
        pre.textContent = result.answer;
        details.append(summary, pre);
        td.appendChild(details);
        row.appendChild(td);
      }
      tbody.appendChild(row);
    }
    status.hidden = true;
    $("bench-result").hidden = false;
  } catch (err) {
    status.textContent = `Ошибка: ${err.message}`;
    status.classList.add("error");
    status.hidden = false;
  } finally {
    $("bench-run").disabled = false;
  }
}

/* ---------- init ---------- */

$("tab-chat").addEventListener("click", () => showTab("chat"));
$("tab-benchmark").addEventListener("click", () => showTab("benchmark"));
$("chat-model").addEventListener("change", onChatModelChange);
$("chat-send").addEventListener("click", sendChat);
$("chat-new-conversation").addEventListener("click", startNewConversation);
$("bench-run").addEventListener("click", runBenchmark);
$("chat-prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendChat();
  }
});

renderConversationId();
renderTaskMemory({});
loadModels();
restoreConversation();
