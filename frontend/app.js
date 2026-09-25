/* LLM Lab frontend logic. Models always come from GET /api/models — never hardcoded. */

"use strict";

const EFFORT_ORDER = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];
const dash = "—";

let models = [];

function $(id) { return document.getElementById(id); }

async function api(path, options) {
  const response = await fetch(path, options);
  let body = null;
  try { body = await response.json(); } catch { /* non-JSON error body */ }
  if (!response.ok) {
    const message = body && body.error && body.error.message
      ? body.error.message
      : `HTTP ${response.status}`;
    throw new Error(message);
  }
  return body;
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
  return models.find((m) => m.id === $("chat-model").value) || null;
}

function populateEffortSelect(select, efforts) {
  select.innerHTML = "";
  const sorted = [...efforts].sort(
    (a, b) => EFFORT_ORDER.indexOf(a) - EFFORT_ORDER.indexOf(b)
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
  if (!model) { badge.hidden = true; return; }
  badge.textContent = `provider: ${model.provider}`;
  badge.hidden = false;
  populateEffortSelect($("chat-effort"), model.reasoning_effort);
}

function onBenchSelectionChange() {
  const checked = checkedBenchModels();
  // Benchmark uses one effort for all models: keep only the values every
  // selected model supports (server validates the same way).
  let common = null;
  for (const model of checked) {
    const efforts = new Set(model.reasoning_effort);
    common = common === null ? efforts : new Set([...common].filter((e) => efforts.has(e)));
  }
  populateEffortSelect($("bench-effort"), common ? [...common] : []);
}

function checkedBenchModels() {
  return models.filter((m) => $(`bench-model-${m.id}`) && $(`bench-model-${m.id}`).checked);
}

function renderCatalog() {
  const chatSelect = $("chat-model");
  chatSelect.innerHTML = "";
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.display_name;
    chatSelect.appendChild(option);
  }

  const benchBox = $("bench-models");
  benchBox.innerHTML = "";
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
  const status = $("chat-status");
  try {
    models = await api("/api/models");
    renderCatalog();
  } catch (err) {
    status.textContent = `Не удалось загрузить список моделей: ${err.message}`;
    status.classList.add("error");
    status.hidden = false;
  }
}

/* ---------- chat ---------- */

function setCell(id, value) {
  $(id).textContent = value === null || value === undefined || value === "" ? dash : String(value);
}

function renderArtifacts(artifacts) {
  const container = $("chat-artifacts");
  container.replaceChildren();
  const validArtifacts = (artifacts || []).filter((artifact) => {
    try {
      const url = new URL(artifact.download_url);
      return url.protocol === "http:" || url.protocol === "https:";
    } catch { return false; }
  });
  if (validArtifacts.length === 0) {
    container.hidden = true;
    return;
  }
  const title = document.createElement("strong");
  title.textContent = "Файлы:";
  container.appendChild(title);
  for (const artifact of validArtifacts) {
    const link = document.createElement("a");
    link.href = artifact.download_url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = artifact.filename || artifact.report_id || artifact.download_url;
    container.appendChild(link);
  }
  container.hidden = false;
}

function renderMCPTrace(trace) {
  const container = $("chat-mcp-trace");
  const list = $("chat-mcp-trace-list");
  if (!container || !list) return;
  list.replaceChildren();
  if (!Array.isArray(trace) || trace.length === 0) {
    container.hidden = true;
    return;
  }
  for (const event of trace) {
    const item = document.createElement("li");
    const entry = event && typeof event === "object" ? event : {};
    const details = [entry.server, entry.tool, entry.alias, entry.status]
      .filter((value) => value !== null && value !== undefined && value !== "")
      .map(String);
    item.textContent = details.length > 0 ? details.join(" · ") : dash;
    list.appendChild(item);
  }
  container.hidden = false;
}

async function sendChat() {
  const model = selectedModel();
  const prompt = $("chat-prompt").value.trim();
  const status = $("chat-status");
  const result = $("chat-result");

  if (!model || !prompt) {
    status.textContent = "Выберите модель и введите prompt.";
    status.classList.add("error");
    status.hidden = false;
    return;
  }

  $("chat-send").disabled = true;
  status.textContent = "Ждём ответ…";
  status.classList.remove("error");
  status.hidden = false;
  result.hidden = true;

  try {
    const payload = { model: model.id, prompt };
    payload.mcp_servers = [];
    // Keep cached pages using the former single MCP checkbox functional.
    const taskmanagerCheckbox = $("chat-use-taskmanager-mcp") || $("chat-use-mcp");
    const obsidianCheckbox = $("chat-use-obsidian-mcp");
    if (taskmanagerCheckbox?.checked) payload.mcp_servers.push("taskmanager");
    if (obsidianCheckbox?.checked) payload.mcp_servers.push("obsidian");
    if ($("chat-effort").value && !$("chat-effort").disabled) {
      payload.reasoning_effort = $("chat-effort").value;
    }
    const data = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    $("chat-answer").textContent = data.answer;
    renderArtifacts(data.artifacts);
    renderMCPTrace(data.mcp_trace);
    const usage = data.usage || {};
    setCell("m-provider", data.provider);
    setCell("m-model", data.model);
    setCell("m-latency", `${Math.round(data.latency_ms)} ms`);
    setCell("m-prompt-tokens", usage.prompt_tokens);
    setCell("m-completion-tokens", usage.completion_tokens);
    setCell("m-total-tokens", usage.total_tokens);
    setCell("m-reasoning-tokens", usage.reasoning_tokens);
    setCell("m-finish", data.finish_reason);
    setCell("m-mcp-tools", (data.mcp_tools_used || []).join(", "));
    status.hidden = true;
    result.hidden = false;
  } catch (err) {
    status.textContent = `Ошибка: ${err.message}`;
    status.classList.add("error");
    status.hidden = false;
  } finally {
    $("chat-send").disabled = false;
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

  const payload = {
    prompt,
    models: checked.map((m) => m.id),
    repeats,
  };
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
    tbody.innerHTML = "";
    for (const r of data.results) {
      const row = document.createElement("tr");
      benchCell(row, r.model);
      benchCell(row, r.provider);
      if (r.error) {
        benchCell(row, dash, "error");
        benchCell(row, dash, "error");
        benchCell(row, dash, "error");
        benchCell(row, dash, "error");
        benchCell(row, dash, "error");
        benchCell(row, dash, "error");
        const td = benchCell(row, r.error, "error");
        td.colSpan = 1;
      } else {
        const usage = r.usage || {};
        benchCell(row, `${Math.round(r.latency_ms)}`);
        benchCell(row, usage.prompt_tokens ?? dash);
        benchCell(row, usage.completion_tokens ?? dash);
        benchCell(row, usage.reasoning_tokens ?? dash);
        benchCell(row, usage.total_tokens ?? dash);
        benchCell(row, fmtCost(r.cost));
        benchCell(row, r.finish_reason ?? dash);

        const td = document.createElement("td");
        const details = document.createElement("details");
        details.className = "answer-details";
        const summary = document.createElement("summary");
        summary.textContent = "показать";
        const pre = document.createElement("pre");
        pre.textContent = r.answer;
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
$("bench-run").addEventListener("click", runBenchmark);

loadModels();
