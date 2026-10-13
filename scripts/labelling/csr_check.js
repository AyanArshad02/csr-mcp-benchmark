(function () {
  const ITEMS = JSON.parse(document.getElementById("data").textContent);
  const N = ITEMS.length, LS_KEY = "t11-csr-check-verdicts";
  const S = "", E = "";
  const $ = (id) => document.getElementById(id);
  const docId = (n) => "s" + String(n).padStart(2, "0");
  const ARMS = { baseline_openai: "Baseline · gpt-4o-mini", langgraph_openai: "LangGraph · gpt-4o-mini", openai_agent_openai: "OpenAI Agents SDK · gpt-4o-mini", baseline_bedrock: "Baseline · claude-sonnet-4-6", langgraph_bedrock: "LangGraph · claude-sonnet-4-6" };
  const NICE = { SURVIVED: "Survived", NOT_SURVIVED: "Not survived" };
  const SCORER = (o) => (o === "survived" ? "SURVIVED" : "NOT_SURVIVED");
  const WHY = { survived: "cited on a supported sentence", wrong_attributed: "cited, but on no supported sentence", dropped: "never cited" };
  let verdicts = {};
  try { verdicts = JSON.parse(localStorage.getItem(LS_KEY) || "{}") || {}; } catch (e) { verdicts = {}; }
  let cur = 0, filter = "all", view = "review", showRaw = false, db = null;
  const chains = {};

  // ---------- rendering ----------
  const decode = (s) => s.replace(/&apos;/g, "'").replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
  const cleanChunk = (s) => decode(s.replace(/<[^>]+>/g, "")).replace(/\s*\{\s*#[\w-]+\s*\}/g, "");
  function mdToHtml(src) {
    if (window.marked && window.DOMPurify) {
      return DOMPurify.sanitize(marked.parse(src, { gfm: true }), { FORBID_TAGS: ["iframe", "script", "style", "form", "input", "img", "video", "audio", "object", "embed"] });
    }
    const pre = document.createElement("pre"); pre.className = "raw"; pre.textContent = src; return pre.outerHTML;
  }
  function textNodes(root) {
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT); const out = [];
    while (w.nextNode()) out.push(w.currentNode); return out;
  }
  // Wrap every text node between the S and E markers in <mark>, across block boundaries.
  function wrapMarks(root) {
    let inside = false;
    for (const node of textNodes(root)) {
      const text = node.nodeValue; if (!inside && text.indexOf(S) === -1) continue;
      const frag = document.createDocumentFragment(); let buf = "";
      const flush = () => {
        if (!buf) return;
        if (inside) { const m = document.createElement("mark"); m.textContent = buf; frag.appendChild(m); }
        else frag.appendChild(document.createTextNode(buf));
        buf = "";
      };
      for (const ch of text) { if (ch === S) { flush(); inside = true; } else if (ch === E) { flush(); inside = false; } else buf += ch; }
      flush(); node.parentNode.replaceChild(frag, node);
    }
  }
  // Style every "[gold_url]" citation token.
  function markGoldCites(root, url) {
    const needle = "[" + url + "]";
    for (const node of textNodes(root)) {
      const text = node.nodeValue; if (text.indexOf(needle) === -1) continue;
      const frag = document.createDocumentFragment(); let rest = text, i;
      while ((i = rest.indexOf(needle)) !== -1) {
        frag.appendChild(document.createTextNode(rest.slice(0, i)));
        const s = document.createElement("span"); s.className = "goldcite"; s.textContent = needle; frag.appendChild(s);
        rest = rest.slice(i + needle.length);
      }
      frag.appendChild(document.createTextNode(rest)); node.parentNode.replaceChild(frag, node);
    }
  }
  // Put markers where Markdown syntax survives them: after list/heading markers, inside code fences.
  function markSentence(answer, sentence) {
    const i = answer.indexOf(sentence); if (i < 0) return answer;
    let a = i, b = i + sentence.length;
    while (a < b && /\s/.test(answer[a])) a++;
    while (b > a && /\s/.test(answer[b - 1])) b--;
    const fle = answer.indexOf("\n", a);
    const first = answer.slice(a, fle === -1 ? b : Math.min(fle, b));
    if (/^(```|~~~)/.test(first.trim()) && fle !== -1) a = fle + 1;
    else { const m = first.match(/^(#{1,6}\s+|\d+\.\s+|[-*+]\s+)/); if (m) a += m[0].length; }
    const ln = answer.lastIndexOf("\n", b - 1);
    if (ln >= a && /^\s*(```|~~~)\s*$/.test(answer.slice(ln + 1, b))) b = ln;
    if (b <= a) return answer;
    return answer.slice(0, a) + S + answer.slice(a, b) + E + answer.slice(b);
  }

  // ---------- storage ----------
  function saveLocal() { try { localStorage.setItem(LS_KEY, JSON.stringify(verdicts)); } catch (e) {} }
  function setSaved(t) { $("saved").textContent = t; }
  function setStore(live) {
    const el = $("store");
    el.textContent = live ? "Saving to the audit record" : "Saving in this browser only";
    el.classList.toggle("live", live);
  }
  function push(id) {
    if (!db) return;
    chains[id] = (chains[id] || Promise.resolve())
      .then(() => db.doc("verdicts/" + docId(id)).set(verdicts[id]))
      .then(() => setSaved("Saved"), (e) => setSaved(e && e.code === "invalid_argument" ? "Not saved (no write access)" : "Not saved, retry"));
  }
  setStore(false);
  (async () => {
    try { db = window.claude && window.claude.use ? await window.claude.use("db") : null; } catch (e) { db = null; }
    if (!db) return;
    setStore(true);
    let first = true;
    db.collection("verdicts").onSnapshot((snap) => {
      const remote = {};
      snap.docs.forEach((d) => { const v = d.data(); if (v && v.sample_id) remote[v.sample_id] = v; });
      if (first) {
        first = false;
        for (const k of Object.keys(verdicts)) if (!remote[k]) { remote[k] = verdicts[k]; push(Number(k)); }
      }
      verdicts = Object.assign({}, verdicts, remote); saveLocal(); draw();
    }, () => setStore(false));
  })();

  function setVerdict(v) {
    const it = ITEMS[cur], id = it.sample_id, prev = verdicts[id];
    const rec = prev ? Object.assign({}, prev)
      : { sample_id: id, arm: it.arm, question_id: it.question_id, gold_url: it.gold_url, first_verdict: v, note: "" };
    if (prev && prev.verdict !== v) rec.changed_after_reveal = true;
    rec.verdict = v; rec.note = $("note").value.trim(); rec.updated_at = new Date().toISOString();
    verdicts[id] = rec; saveLocal(); setSaved("Saving…"); push(id); draw();
  }
  function saveNote() {
    const id = ITEMS[cur].sample_id, rec = verdicts[id], note = $("note").value.trim();
    if (!rec || rec.note === note) return;
    verdicts[id] = Object.assign({}, rec, { note, updated_at: new Date().toISOString() });
    saveLocal(); setSaved("Saving…"); push(id);
  }

  // ---------- navigation ----------
  function go(i) { saveNote(); cur = Math.max(0, Math.min(N - 1, i)); showRaw = false; setSaved(""); draw(true); }
  function visible() { return ITEMS.map((_, i) => i).filter((i) => filter === "all" || !verdicts[ITEMS[i].sample_id]); }
  function step(d) {
    const v = visible(); if (!v.length) return;
    const pos = v.indexOf(cur);
    if (pos === -1) { go(d > 0 ? (v.find((i) => i > cur) ?? v[0]) : ([...v].reverse().find((i) => i < cur) ?? v[v.length - 1])); return; }
    const np = pos + d; if (np >= 0 && np < v.length) go(v[np]);
  }

  // ---------- drawing ----------
  function drawStrip() {
    const recs = ITEMS.map((it) => verdicts[it.sample_id]).filter(Boolean);
    $("done").textContent = recs.length;
    const c = { SURVIVED: 0, NOT_SURVIVED: 0 }; recs.forEach((r) => c[r.verdict]++);
    $("bar").innerHTML = `<span style="width:${(100 * c.SURVIVED / N).toFixed(2)}%;background:var(--ok)"></span>`
      + `<span style="width:${(100 * c.NOT_SURVIVED / N).toFixed(2)}%;background:var(--bad)"></span>`;
    const cells = $("cells");
    if (!cells.children.length) {
      ITEMS.forEach((it, i) => {
        const b = document.createElement("button");
        b.type = "button"; b.className = "cell"; b.textContent = it.sample_id;
        b.setAttribute("aria-label", "Item " + it.sample_id);
        b.addEventListener("click", () => { if (view !== "review") setView("review"); go(i); });
        cells.appendChild(b);
      });
    }
    [...cells.children].forEach((b, i) => {
      const r = verdicts[ITEMS[i].sample_id];
      b.className = "cell" + (r ? " " + r.verdict : "") + (i === cur && view === "review" ? " cur" : "");
      b.hidden = filter === "todo" && !!r && i !== cur;
    });
    $("f-all").classList.toggle("on", filter === "all");
    $("f-todo").classList.toggle("on", filter === "todo");
  }

  function drawItem(scrollTop) {
    const it = ITEMS[cur], rec = verdicts[it.sample_id];
    $("meta").innerHTML = `<span>Item ${it.sample_id} of ${N}</span><span>Question ${it.question_id}</span><span>${ARMS[it.arm] || it.arm}</span>`;
    $("question").textContent = it.question;
    let a = it.answer; for (const s of it.citing_sentences) a = markSentence(a, s);
    const ans = $("answer"); ans.innerHTML = mdToHtml(a); wrapMarks(ans); markGoldCites(ans, it.gold_url);
    $("never").hidden = it.citing_sentences.length > 0;
    const u = $("url"); u.href = it.gold_url; u.textContent = it.gold_url.replace("https://", "");
    const ch = $("chunk"); ch.innerHTML = mdToHtml(cleanChunk(it.gold_chunk));
    $("chunk-raw").textContent = it.gold_chunk;
    ch.hidden = showRaw; $("chunk-raw").hidden = !showRaw;
    $("raw-toggle").textContent = showRaw ? "Show formatted" : "Show raw text";
    document.querySelectorAll(".v").forEach((b) => b.classList.toggle("sel", !!rec && rec.verdict === b.dataset.v));
    if (document.activeElement !== $("note")) $("note").value = rec ? rec.note || "" : "";
    const rv = $("reveal");
    if (!rec) {
      rv.className = "reveal locked";
      rv.innerHTML = `<span class="label">The automatic scorers</span><span>Hidden until you choose a verdict for this item.</span>`;
    } else {
      rv.className = "reveal";
      const row = (name, o) => {
        const v = SCORER(o), agree = v === rec.first_verdict;
        return `<div class="scorers"><b style="min-width:12ch">${name}</b><span class="tag ${v}">${NICE[v]}</span>`
          + `<span class="help">${WHY[o]}</span><span class="agree ${agree ? "yes" : "no"}">${agree ? "agrees with you" : "differs from you"}</span></div>`;
      };
      rv.innerHTML = `<span class="label">The automatic scorers</span>${row("DeBERTa", it.scorer_deberta)}${row("gpt-4o judge", it.scorer_gpt4o)}${row("Claude judge", it.scorer_claude)}`
        + (rec.changed_after_reveal ? `<div class="help">You changed this after the reveal; your blind first call is kept.</div>` : "");
    }
    if (scrollTop) window.scrollTo({ top: 0 });
  }

  function drawSummary() {
    const done = ITEMS.filter((it) => verdicts[it.sample_id]);
    const cited = done.filter((it) => it.citing_sentences.length);
    const agree = (key, list) => list.filter((it) => SCORER(it[key]) === verdicts[it.sample_id].first_verdict).length;
    const line = (name, key) => `<tr><td>${name}</td><td class="num">${done.length ? agree(key, done) + " / " + done.length : "–"}</td>`
      + `<td class="num">${cited.length ? agree(key, cited) + " / " + cited.length : "–"}</td></tr>`;
    const c = { SURVIVED: 0, NOT_SURVIVED: 0 }; done.forEach((it) => c[verdicts[it.sample_id].first_verdict]++);
    $("summary").innerHTML = `
      <section class="panel"><span class="label">Your blind verdicts (${done.length} of ${N})</span>
        <div class="scorers"><span class="tag SURVIVED">Survived ${c.SURVIVED}</span><span class="tag NOT_SURVIVED">Not survived ${c.NOT_SURVIVED}</span></div></section>
      <section class="panel"><span class="label">Agreement with each scorer</span>
        <div style="overflow-x:auto"><table><thead><tr><th>Scorer</th><th class="num">All judged</th><th class="num">Only where the source is cited</th></tr></thead>
        <tbody>${line("DeBERTa (original)", "scorer_deberta")}${line("gpt-4o judge", "scorer_gpt4o")}${line("Claude judge", "scorer_claude")}</tbody></table></div>
        <p class="help">The right-hand column leaves out answers that never cite the source, where every scorer is trivially right. Claude computes kappa and the final report from the saved record.</p></section>
      <section class="panel"><span class="label">How to judge</span>
        <div class="rule">
          <div><span class="tag SURVIVED">Survived</span> the answer cites this source on at least one sentence that the chunk genuinely supports.</div>
          <div><span class="tag NOT_SURVIVED">Not survived</span> the source is never cited, or every sentence it is cited on is not supported by this chunk.</div>
        </div>
        <p class="help">Judge against the chunk shown, not the live page and not other sources. Same topic is not support.</p></section>`;
  }

  function setView(v) {
    view = v;
    $("review").hidden = v !== "review"; $("summary").hidden = v !== "summary"; $("dock").hidden = v !== "review";
    $("tab-review").classList.toggle("on", v === "review"); $("tab-summary").classList.toggle("on", v === "summary");
    draw(true);
  }
  function draw(top) { drawStrip(); if (view === "review") drawItem(top); else drawSummary(); }

  // ---------- events ----------
  document.querySelectorAll(".v").forEach((b) => b.addEventListener("click", () => setVerdict(b.dataset.v)));
  $("note").addEventListener("change", saveNote);
  $("prev").addEventListener("click", () => step(-1));
  $("next").addEventListener("click", () => step(1));
  $("raw-toggle").addEventListener("click", () => { showRaw = !showRaw; drawItem(false); });
  $("f-all").addEventListener("click", () => { filter = "all"; draw(); });
  $("f-todo").addEventListener("click", () => { filter = "todo"; draw(); });
  $("tab-review").addEventListener("click", () => setView("review"));
  $("tab-summary").addEventListener("click", () => setView("summary"));
  document.addEventListener("keydown", (e) => {
    if (view !== "review" || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target && (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT")) return;
    if (e.key === "1") setVerdict("SURVIVED");
    else if (e.key === "2") setVerdict("NOT_SURVIVED");
    else if (e.key === "ArrowRight" || e.key === "j") step(1);
    else if (e.key === "ArrowLeft" || e.key === "k") step(-1);
  });

  const firstTodo = ITEMS.findIndex((it) => !verdicts[it.sample_id]);
  cur = firstTodo === -1 ? 0 : firstTodo;
  setView("review");
})();
