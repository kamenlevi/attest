const $ = (s) => document.querySelector(s);
const api = (path, body) =>
  fetch(path, { method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined }).then((r) => r.json());

let STATE = null;

/* ---- nav ---- */
document.querySelectorAll(".nav-item").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".view").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $("#view-" + b.dataset.view).classList.add("active");
  }));

/* ---- theme ---- */
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  $("#theme-label").textContent = theme === "dark" ? "Dark" : "Light";
}
$("#theme-toggle").addEventListener("click", () => {
  const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  applyTheme(next);
  api("/api/settings", { patch: { theme: next } });
});

/* ---- state load ---- */
async function loadState() {
  STATE = await api("/api/state");
  const c = STATE.config;
  applyTheme(c.theme);
  $("#set-base").value = c.provider.base_url || "";
  $("#set-key").value = "";
  $("#set-key").placeholder = c.provider.api_key_set ? "•••• set — leave blank to keep" : "paste your API key";
  $("#set-gen").value = c.models.generator || "";
  $("#set-judge").value = c.models.judge || "";
  $("#set-vision").value = c.models.vision || "";
  $("#set-embedder").value = c.models.embedder || "local";
  $("#p-lexical").checked = !!c.pipeline.lexical;
  $("#p-expand").checked = !!c.pipeline.expand;
  $("#p-rerank").checked = !!c.pipeline.rerank;
  $("#p-uncited").checked = !!c.pipeline.allow_uncited;
  $("#p-k").value = c.pipeline.k || 8;
  renderLibrary();
}

function renderLibrary() {
  const list = $("#doc-list");
  const docs = STATE.library || [];
  list.innerHTML = docs.length
    ? docs.map((d) => `<li>${escapeHtml(d)}</li>`).join("")
    : `<li class="muted">nothing indexed yet</li>`;
  $("#active-index").textContent = docs.length ? `${docs.length} document${docs.length > 1 ? "s" : ""}` : "no index";
}

/* ---- ask ---- */
$("#ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#q").value.trim();
  if (!q) return;
  $("#q").value = "";
  addUser(q);
  const thinking = addThinking();
  $("#ask-btn").disabled = true;
  try {
    const res = await api("/api/ask", { question: q });
    thinking.remove();
    addAnswer(res);
  } catch (err) {
    thinking.remove();
    addAnswer({ error: "Request failed: " + err });
  }
  $("#ask-btn").disabled = false;
});

function addUser(text) {
  const el = document.createElement("div");
  el.className = "msg user";
  el.textContent = text;
  $("#chat").appendChild(el);
  scrollChat();
}
function addThinking() {
  const el = document.createElement("div");
  el.className = "msg bot";
  el.innerHTML = `<div class="answer-card"><div class="thinking"><span class="dot-pulse"></span> searching your sources…</div></div>`;
  $("#chat").appendChild(el);
  scrollChat();
  return el;
}
function addAnswer(res) {
  const el = document.createElement("div");
  el.className = "msg bot";
  if (res.error) {
    el.innerHTML = `<div class="answer-card"><div class="answer-text abstained">${escapeHtml(res.error)}</div></div>`;
    $("#chat").appendChild(el); scrollChat(); return;
  }
  let badge, text;
  if (res.abstained) {
    badge = `<span class="badge abstain">Not in your sources</span>`;
    text = `<div class="answer-text abstained">The answer isn't in your documents, so I won't guess.</div>`;
  } else if (res.citations && res.citations.length) {
    badge = `<span class="badge grounded">Grounded · cited ${res.citations.map((c) => "[" + c + "]").join(" ")}</span>`;
    text = `<div class="answer-text">${escapeHtml(res.answer)}</div>`;
  } else {
    badge = `<span class="badge abstain">Answered without a citation</span>`;
    text = `<div class="answer-text">${escapeHtml(res.answer)}</div>`;
  }
  const passages = (res.passages || []).map((p) =>
    `<div class="passage"><span class="pid">[${p.id}]</span> <span class="muted">${escapeHtml(p.source)}</span>
       <div class="ptext">${escapeHtml(p.text)}</div></div>`).join("");
  const uncited = res.uncited
    ? `<div class="uncited-note"><b>Unverifiable —</b> from the model's own knowledge, NOT your sources:<br>${escapeHtml(res.uncited)}</div>`
    : "";
  el.innerHTML = `<div class="answer-card">${badge}${text}
    ${passages ? `<details class="sources"><summary>retrieved passages (${res.passages.length})</summary>${passages}</details>` : ""}
    ${uncited}</div>`;
  $("#chat").appendChild(el);
  scrollChat();
}
function scrollChat() { const c = $("#chat"); c.scrollTop = c.scrollHeight; }

/* ---- index ---- */
$("#index-btn").addEventListener("click", async () => {
  const path = $("#doc-path").value.trim();
  if (!path) return;
  const vision = $("#doc-vision").checked;
  $("#index-hint").textContent = vision
    ? "Transcribing pages with the vision model… (one call per page)" : "Embedding…";
  $("#index-btn").disabled = true;
  const res = await api("/api/index", { path, vision });
  $("#index-btn").disabled = false;
  if (res.error) { $("#index-hint").textContent = "⚠ " + res.error; return; }
  $("#index-hint").textContent = res.skipped
    ? "Already indexed — skipped (instant)." : `Indexed ✓ — ${res.chunks} chunks total.`;
  $("#doc-path").value = "";
  STATE.library = (res.sources || []).map((s) => s.split("/").pop());
  renderLibrary();
});

/* ---- settings ---- */
$("#save-settings").addEventListener("click", async () => {
  const patch = {
    provider: { base_url: $("#set-base").value.trim(), api_key: $("#set-key").value.trim() },
    models: { generator: $("#set-gen").value.trim(), judge: $("#set-judge").value.trim(),
      vision: $("#set-vision").value.trim(), embedder: $("#set-embedder").value },
    pipeline: { lexical: $("#p-lexical").checked, expand: $("#p-expand").checked,
      rerank: $("#p-rerank").checked, allow_uncited: $("#p-uncited").checked,
      k: parseInt($("#p-k").value, 10) || 8 },
  };
  STATE = await api("/api/settings", { patch });
  $("#set-key").value = "";
  const f = $("#saved-flash"); f.classList.add("show"); setTimeout(() => f.classList.remove("show"), 1500);
});

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

loadState();
