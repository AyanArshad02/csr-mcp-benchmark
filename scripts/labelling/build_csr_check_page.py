"""scripts/labelling/build_csr_check_page.py — Build the blind labelling page for Ticket 11 (G4).

Reuses the design of the earlier audit pages (scripts/labelling/audit_template.html) and embeds
results/csr_validation_sample.jsonl. The page saves verdicts to its own database; Claude reads them
back with the ArtifactData tool.

Usage:  python3 scripts/labelling/build_csr_check_page.py OUT.html
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
t = (HERE / "audit_template.html").read_text()
head = t[:t.index('<div class="wrap">')].replace("<title>Ticket 4 Insertion Audit</title>", "<title>CSR Human Check</title>")
head = head.replace("</style>", """
.goldcite { font-family: var(--font-mono); font-size: .8em; background: var(--accent-soft); color: var(--accent); border-radius: 4px; padding: 0 3px; font-weight: 600; }
.never { border: 1px dashed var(--bad); color: var(--bad); background: var(--bad-soft); border-radius: 8px; padding: 10px 12px; font-size: 14px; }
.rule { display: grid; gap: 6px; font-size: 14px; }
.v[data-v="SURVIVED"]:hover, .v[data-v="SURVIVED"].sel { border-color: var(--ok); color: var(--ok); }
.v[data-v="NOT_SURVIVED"]:hover, .v[data-v="NOT_SURVIVED"].sel { border-color: var(--bad); color: var(--bad); }
.v[data-v="SURVIVED"].sel { background: var(--ok-soft); }
.v[data-v="NOT_SURVIVED"].sel { background: var(--bad-soft); }
.cell.SURVIVED { background: var(--ok-soft); border-color: var(--ok); color: var(--ok); }
.cell.NOT_SURVIVED { background: var(--bad-soft); border-color: var(--bad); color: var(--bad); }
.tag.SURVIVED { background: var(--ok-soft); color: var(--ok); }
.tag.NOT_SURVIVED { background: var(--bad-soft); color: var(--bad); }
.scorers { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; }
</style>""")
script_lib = t[t.index('<script src="https://cdnjs'):t.index('<script>\n')]

BODY = """
<div class="wrap">
  <header class="head">
    <div>
      <h1>CSR Human Check</h1>
      <p>Did this gold source survive into the answer, cited on a sentence its chunk genuinely supports? The three automatic scorers' verdicts appear only after you lock in yours.</p>
    </div>
    <div class="head-right">
      <span class="pill" id="store">Connecting…</span>
      <button class="btn" id="tab-review" type="button">Review</button>
      <button class="btn" id="tab-summary" type="button">Summary</button>
    </div>
  </header>
  <section class="strip" aria-label="Progress">
    <div class="strip-top">
      <span><span class="count" id="done">0</span> of 30 judged</span>
      <div class="head-right">
        <button class="btn" id="f-all" type="button">All</button>
        <button class="btn" id="f-todo" type="button">Not judged yet</button>
      </div>
    </div>
    <div class="bar" id="bar"></div>
    <div class="cells" id="cells"></div>
  </section>
  <main id="review" class="wrap" style="padding:0;max-width:none">
    <div class="item-head">
      <div class="meta" id="meta"></div>
      <div class="question" id="question"></div>
    </div>
    <div class="compare">
      <section class="panel">
        <div class="panel-head"><span class="label">Agent's answer</span><span class="help">sentences citing the gold source are highlighted</span></div>
        <div id="never" class="never" hidden>This answer never cites the gold source on the right.</div>
        <div class="md" id="answer"></div>
      </section>
      <section class="panel">
        <div class="panel-head">
          <span class="label">Gold source the agent retrieved</span>
          <button class="toggle" id="raw-toggle" type="button">Show raw text</button>
        </div>
        <a class="url" id="url" target="_blank" rel="noopener"></a>
        <div class="md" id="chunk"></div>
        <pre class="raw" id="chunk-raw" hidden></pre>
      </section>
    </div>
    <section class="reveal locked" id="reveal"></section>
  </main>
  <main id="summary" class="summary" hidden></main>
</div>
<div class="dock" id="dock">
  <div class="dock-in">
    <div class="verdicts" role="group" aria-label="Verdict">
      <button class="v" data-v="SURVIVED" type="button">Survived <kbd>1</kbd></button>
      <button class="v" data-v="NOT_SURVIVED" type="button">Not survived <kbd>2</kbd></button>
    </div>
    <textarea class="note" id="note" placeholder="Note (optional): why, or what made it hard"></textarea>
    <span class="saved" id="saved"></span>
    <div class="nav">
      <button class="btn" id="prev" type="button">← Prev</button>
      <button class="btn" id="next" type="button">Next →</button>
    </div>
  </div>
</div>
"""

JS = (HERE / "csr_check.js").read_text()

items = [json.loads(l) for l in open("results/csr_validation_sample.jsonl")]
keep = ("sample_id", "arm", "question_id", "question", "answer", "gold_url", "gold_chunk",
        "citing_sentences", "scorer_deberta", "scorer_gpt4o", "scorer_claude")
data = json.dumps([{k: i[k] for k in keep} for i in items], ensure_ascii=False).replace("</", "<\\/")
missing = [i["sample_id"] for i in items for s in i["citing_sentences"] if s not in i["answer"]]
page = head + BODY + '<script id="data" type="application/json">' + data + "</script>\n" + script_lib + "<script>\n" + JS + "\n</script>"
Path(sys.argv[1]).write_text(page)
print(f"wrote {sys.argv[1]} ({len(page)} bytes); citing sentences not found in their answer: {missing}")
