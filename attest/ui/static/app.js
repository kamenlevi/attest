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
  $("#p-verify").checked = c.pipeline.verify !== false;
  $("#p-uncited").checked = !!c.pipeline.allow_uncited;
  $("#p-k").value = c.pipeline.k || 8;
  $("#cmp-a").value = $("#cmp-a").value || c.models.generator || "";
  $("#cmp-b").value = $("#cmp-b").value || c.models.judge || "";
  renderLibrary();
}

function renderLibrary() {
  const list = $("#doc-list");
  const docs = STATE.library || [];
  list.innerHTML = docs.length
    ? docs.map((d) => {
        const name = typeof d === "string" ? d : d.name;
        const meta = typeof d === "string" ? "" :
          `<span class="doc-meta">${d.chunks} chunks${d.pages ? " · " + d.pages + " pages" : ""}</span>`;
        return `<li>${escapeHtml(name)}${meta}</li>`;
      }).join("")
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

/* The verification ladder → badge. Only "verified" is green: it means the
   citations are real AND an independent judge confirmed they support the answer. */
const BADGES = {
  verified:    { cls: "grounded",  label: "Verified" },
  unverified:  { cls: "cited",     label: "Cited · not verified" },
  unsupported: { cls: "uncited",   label: "Citation doesn't support this" },
  invalid:     { cls: "uncited",   label: "Fabricated citation" },
  uncited:     { cls: "uncited",   label: "No citation" },
  abstained:   { cls: "abstain",   label: "Not in your sources" },
};

function citeLabel(c) {
  return `[${c.id}] ${c.source}${c.page ? " · p. " + c.page : ""}`;
}

function addAnswer(res) {
  const el = document.createElement("div");
  el.className = "msg bot";
  if (res.error) {
    el.innerHTML = `<div class="answer-card"><div class="answer-text abstained">${escapeHtml(res.error)}</div></div>`;
    $("#chat").appendChild(el); scrollChat(); return;
  }
  const v = res.verification || {};
  const b = BADGES[v.status] || (res.abstained ? BADGES.abstained : BADGES.uncited);
  const badge = `<span class="badge ${b.cls}" title="${escapeHtml(v.note || "")}">${b.label}</span>`;
  const text = res.abstained
    ? `<div class="answer-text abstained">The answer isn't in your documents, so I won't guess.</div>`
    : `<div class="answer-text">${escapeHtml(res.answer)}</div>`;
  const cites = (res.cited || []).length
    ? `<div class="cite-row">${res.cited.map((c) =>
        `<span class="cite-pill">${escapeHtml(citeLabel(c))}</span>`).join("")}</div>`
    : "";
  const note = v.note && v.status !== "verified" && v.status !== "abstained"
    ? `<div class="verify-note">${escapeHtml(v.note)}</div>` : "";
  const passages = (res.passages || []).map((p) =>
    `<div class="passage"><span class="pid">[${p.id}]</span> <span class="muted">${escapeHtml(p.source)}${p.page ? " · p. " + p.page : ""}</span>
       <div class="ptext">${escapeHtml(p.text)}</div></div>`).join("");
  const uncited = res.uncited
    ? `<div class="uncited-note"><b>Unverifiable —</b> from the model's own knowledge, NOT your sources:<br>${escapeHtml(res.uncited)}</div>`
    : "";
  el.innerHTML = `<div class="answer-card">${badge}${text}${cites}${note}
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
  await loadState();
});

/* ---- convert ---- */
$("#cv-vision").addEventListener("change", () => {
  $("#cv-pages-row").style.display = $("#cv-vision").checked ? "flex" : "none";
});
$("#cv-btn").addEventListener("click", async () => {
  const path = $("#cv-path").value.trim();
  if (!path) return;
  const vision = $("#cv-vision").checked;
  $("#cv-hint").textContent = vision
    ? "Transcribing pages with the vision model… (one call per page)" : "Extracting & cleaning…";
  $("#cv-btn").disabled = true;
  const res = await api("/api/convert", {
    path, vision,
    pages: vision ? ($("#cv-pages").value.trim() || null) : null,
    out: $("#cv-out").value.trim() || null,
  });
  $("#cv-btn").disabled = false;
  if (res.error) { $("#cv-hint").textContent = "⚠ " + res.error; return; }
  $("#cv-hint").textContent = `Wrote ${res.chars.toLocaleString()} characters → ${res.out}`;
  $("#cv-result").style.display = "block";
  $("#cv-preview").textContent = res.preview || "";
});

/* ---- measure ---- */
const METRIC_DEFS = [
  { key: "bluff_rate", label: "Bluff rate", pct: true, goodLow: true,
    hint: "Of the trap questions (answer NOT in your documents), how often it made something up. 0% is the promise." },
  { key: "answer_coverage", label: "Coverage", pct: true,
    hint: "Of the answerable questions, how many it actually answered (didn't wrongly abstain)." },
  { key: "citation_rate", label: "Citation rate", pct: true,
    hint: "Of the answers given, how many cited a passage." },
  { key: "correctness_rate", label: "Correctness", pct: true,
    hint: "Of the graded answers, how many the judge marked correct." },
];

function fmtMetric(m, def) {
  const val = m[def.key];
  if (val === undefined) return null;
  return (val * 100).toFixed(0) + "%";
}
function metricClass(m, def) {
  const val = m[def.key];
  if (val === undefined) return "";
  const good = def.goodLow ? val <= 0.001 : val >= 0.9;
  const bad = def.goodLow ? val >= 0.2 : val <= 0.5;
  return good ? "good" : bad ? "bad" : "";
}

$("#ev-btn").addEventListener("click", async () => {
  const questions_path = $("#ev-questions").value.trim();
  if (!questions_path) return;
  $("#ev-hint").textContent = "Running every question through the pipeline… (this can take a few minutes)";
  $("#ev-btn").disabled = true;
  const res = await api("/api/eval", { questions_path, judge: $("#ev-judge").checked });
  $("#ev-btn").disabled = false;
  if (res.error) { $("#ev-hint").textContent = "⚠ " + res.error; return; }
  const m = res.metrics;
  $("#ev-hint").textContent = `Done — ${m.n_total} questions (${m.n_answerable} answerable, ${m.n_traps} traps) with ${res.model}.`;
  $("#ev-result").style.display = "block";
  $("#ev-tiles").innerHTML = METRIC_DEFS.map((d) => {
    const v = fmtMetric(m, d);
    if (v === null) return "";
    return `<div class="tile" title="${escapeHtml(d.hint)}">
      <div class="tile-value ${metricClass(m, d)}">${v}</div>
      <div class="tile-label">${d.label}</div></div>`;
  }).join("");
  $("#ev-table").innerHTML =
    `<tr><th></th><th>Question</th><th>Result</th></tr>` +
    (res.rows || []).map((r) => {
      let verdict, cls;
      if (r.abstained) { verdict = r.answerable ? "abstained (missed)" : "abstained ✓"; cls = r.answerable ? "bad" : "good"; }
      else if (!r.answerable) { verdict = "BLUFFED"; cls = "bad"; }
      else if (r.correct === true) { verdict = "correct ✓"; cls = "good"; }
      else if (r.correct === false) { verdict = "incorrect"; cls = "bad"; }
      else { verdict = "answered " + (r.citations.length ? r.citations.map((c) => "[" + c + "]").join(" ") : "(no citation)"); cls = ""; }
      return `<tr><td class="mono">${r.answerable ? "Q" : "trap"}</td>
        <td title="${escapeHtml(r.answer || "")}">${escapeHtml(r.question)}</td>
        <td class="${cls}">${escapeHtml(verdict)}</td></tr>`;
    }).join("");
});

/* ---- compare ---- */
$("#cmp-btn").addEventListener("click", async () => {
  const questions_path = $("#cmp-questions").value.trim();
  const model_a = $("#cmp-a").value.trim();
  const model_b = $("#cmp-b").value.trim();
  if (!questions_path || !model_a || !model_b) {
    $("#cmp-hint").textContent = "⚠ Fill in both models and the questions file."; return;
  }
  $("#cmp-hint").textContent = "Running the full question set through BOTH models… (this can take a while)";
  $("#cmp-btn").disabled = true;
  const res = await api("/api/compare",
    { questions_path, model_a, model_b, judge: $("#cmp-judge").checked });
  $("#cmp-btn").disabled = false;
  if (res.error) { $("#cmp-hint").textContent = "⚠ " + res.error; return; }
  $("#cmp-hint").textContent = "Done.";
  $("#cmp-result").style.display = "block";
  const ma = res.a.metrics, mb = res.b.metrics;
  $("#cmp-table").innerHTML =
    `<tr><th>Trust number</th><th>${escapeHtml(model_a)}</th><th>${escapeHtml(model_b)}</th></tr>` +
    METRIC_DEFS.map((d) => {
      const va = fmtMetric(ma, d), vb = fmtMetric(mb, d);
      if (va === null && vb === null) return "";
      const na = ma[d.key], nb = mb[d.key];
      let winA = "", winB = "";
      if (na !== undefined && nb !== undefined && na !== nb) {
        const aBetter = d.goodLow ? na < nb : na > nb;
        winA = aBetter ? "good" : ""; winB = aBetter ? "" : "good";
      }
      return `<tr><td title="${escapeHtml(d.hint)}">${d.label}</td>
        <td class="${winA}">${va ?? "—"}</td><td class="${winB}">${vb ?? "—"}</td></tr>`;
    }).join("");
});

/* ---- settings ---- */
document.querySelectorAll(".presets .chip").forEach((chip) =>
  chip.addEventListener("click", () => { $("#set-base").value = chip.dataset.base; }));

$("#save-settings").addEventListener("click", async () => {
  const patch = {
    provider: { base_url: $("#set-base").value.trim(), api_key: $("#set-key").value.trim() },
    models: { generator: $("#set-gen").value.trim(), judge: $("#set-judge").value.trim(),
      vision: $("#set-vision").value.trim(), embedder: $("#set-embedder").value },
    pipeline: { lexical: $("#p-lexical").checked, expand: $("#p-expand").checked,
      rerank: $("#p-rerank").checked, verify: $("#p-verify").checked,
      allow_uncited: $("#p-uncited").checked,
      k: parseInt($("#p-k").value, 10) || 8 },
  };
  STATE = await api("/api/settings", { patch });
  $("#set-key").value = "";
  renderLibrary();
  const f = $("#saved-flash"); f.classList.add("show"); setTimeout(() => f.classList.remove("show"), 1500);
});

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

loadState();
