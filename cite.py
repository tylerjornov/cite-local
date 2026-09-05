#!/usr/bin/env python3
"""Cite — local folder RAG for Ollama (stdlib only).

    python3 cite.py

Opens a browser UI. Pick folders, ask questions, get answers with citations.
Requires Ollama running with nomic-embed-text + granite4.2:3b (or override).
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError

PORT = int(os.environ.get("CITE_PORT", "8787"))
OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
EMBED_MODEL = os.environ.get("CITE_EMBED", "nomic-embed-text")
GEN_MODEL = os.environ.get("CITE_GEN", "granite4.2:3b")
CHUNK_CHARS = 900
CHUNK_OVERLAP = 120
TOP_K = 6
MAX_FILES = 400
MAX_FILE_BYTES = 1_500_000
MAX_CHUNKS = 2500

TEXT_EXT = {
    ".txt", ".md", ".mdx", ".rst", ".org", ".tex", ".html", ".htm",
    ".csv", ".tsv", ".json", ".xml", ".yml", ".yaml", ".toml", ".ini",
    ".log", ".rtf", ".py", ".js", ".ts", ".tsx", ".jsx", ".c", ".h",
    ".cpp", ".cs", ".java", ".go", ".rs", ".rb", ".php", ".swift",
    ".kt", ".sql", ".sh", ".r", ".bib", ".adoc",
}

INDEX_LOCK = threading.Lock()
INDEX: list[dict] = []  # {path, start, text, vec}


def ollama_post(path: str, body: dict, timeout: int = 180) -> dict:
    req = Request(
        f"{OLLAMA}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def embed(text: str) -> list[float]:
    data = ollama_post("/api/embeddings", {"model": EMBED_MODEL, "prompt": text})
    vec = data.get("embedding")
    if not vec:
        raise RuntimeError("empty embedding")
    return vec


def cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / math.sqrt(na * nb)


def chunk_text(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + CHUNK_CHARS)
        if end < n:
            br = text.rfind("\n\n", i + CHUNK_CHARS // 2, end)
            if br == -1:
                br = text.rfind("\n", i + CHUNK_CHARS // 2, end)
            if br != -1:
                end = br
        piece = text[i:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        i = max(end - CHUNK_OVERLAP, i + 1)
    return chunks


def ollama_ok() -> dict:
    try:
        req = Request(f"{OLLAMA}/api/tags")
        with urlopen(req, timeout=4) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
        names = [m.get("name", "") for m in tags.get("models", [])]
        return {
            "ok": True,
            "models": names,
            "embed": EMBED_MODEL,
            "gen": GEN_MODEL,
            "has_embed": any(EMBED_MODEL in n for n in names),
            "has_gen": any(GEN_MODEL in n for n in names),
            "chunks": len(INDEX),
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "embed": EMBED_MODEL,
            "gen": GEN_MODEL,
            "chunks": len(INDEX),
        }


def generate(question: str, hits: list[dict]) -> str:
    sources = []
    for i, h in enumerate(hits, 1):
        sources.append(f"[{i}] {h['path']}\n{h['text']}")
    ctx = "\n\n".join(sources)
    prompt = (
        "You answer using ONLY the numbered sources. "
        "After each factual sentence, add the source tag like [1]. "
        "If the sources do not contain the answer, say so. "
        "Do not invent citations. Be concise.\n\n"
        f"SOURCES:\n{ctx}\n\nQUESTION: {question}\n\nANSWER:"
    )
    data = ollama_post(
        "/api/generate",
        {
            "model": GEN_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 4096, "num_predict": 512},
        },
        timeout=300,
    )
    return (data.get("response") or "").strip()


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Cite</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg: #14120e; --surface: #1e1b16; --ink: #f0eadc; --muted: #9a9184;
    --line: #322d24; --gold: #c9a45c; --gold-dim: #8a7038; --danger: #c45c4a;
    --ok: #7a9e6a;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; min-height: 100%; background: var(--bg); color: var(--ink);
    font-family: "IBM Plex Sans", system-ui, sans-serif; }
  body { display: grid; grid-template-rows: auto 1fr; }
  header {
    display: flex; align-items: center; justify-content: space-between; gap: 1rem;
    padding: 1rem 1.25rem; border-bottom: 1px solid var(--line);
  }
  h1 { font-family: "Source Serif 4", Georgia, serif; font-size: 1.35rem; font-weight: 600;
    margin: 0; letter-spacing: -0.02em; }
  h1 span { color: var(--gold); }
  .status { font-size: 0.75rem; color: var(--muted); text-align: right; }
  .status strong { color: var(--ink); font-weight: 500; }
  .dot { display: inline-block; width: 0.5rem; height: 0.5rem; border-radius: 50%;
    background: var(--danger); margin-right: 0.35rem; }
  .dot.on { background: var(--ok); }
  main {
    display: grid; grid-template-columns: minmax(240px, 320px) 1fr;
    min-height: 0;
  }
  @media (max-width: 720px) { main { grid-template-columns: 1fr; } }
  aside { border-right: 1px solid var(--line); padding: 1.1rem; overflow: auto; }
  section.work { padding: 1.1rem 1.25rem 2rem; overflow: auto; }
  label.pick {
    display: flex; align-items: center; justify-content: center;
    border: 1px dashed var(--gold-dim); border-radius: 10px; padding: 1.1rem;
    cursor: pointer; color: var(--gold); font-size: 0.9rem; font-weight: 500;
  }
  label.pick:hover { background: #252017; }
  input[type=file] { display: none; }
  .hint { color: var(--muted); font-size: 0.78rem; line-height: 1.45; margin: 0.75rem 0 1rem; }
  .meta { font-size: 0.78rem; color: var(--muted); }
  button {
    font: inherit; cursor: pointer; border: 0; border-radius: 8px;
    background: var(--gold); color: #1a150c; font-weight: 600; padding: 0.55rem 0.9rem;
  }
  button.ghost { background: transparent; color: var(--muted); border: 1px solid var(--line); }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.75rem; }
  textarea {
    width: 100%; min-height: 5.5rem; resize: vertical; background: var(--surface);
    color: var(--ink); border: 1px solid var(--line); border-radius: 10px;
    padding: 0.8rem 0.9rem; font: inherit; line-height: 1.5;
  }
  textarea:focus { outline: 2px solid var(--gold-dim); }
  .answer {
    margin-top: 1.1rem; background: var(--surface); border: 1px solid var(--line);
    border-radius: 12px; padding: 1.1rem 1.2rem;
  }
  .answer h2 { font-family: "Source Serif 4", Georgia, serif; font-size: 1.05rem; margin: 0 0 0.7rem; }
  .answer p, .answer li { line-height: 1.55; font-size: 0.95rem; }
  .cites { margin-top: 1rem; }
  .cite {
    border-top: 1px solid var(--line); padding: 0.7rem 0; font-size: 0.8rem;
  }
  .cite b { color: var(--gold); font-weight: 600; }
  .cite pre {
    white-space: pre-wrap; font-family: inherit; color: var(--muted); margin: 0.35rem 0 0;
  }
  .err { color: var(--danger); font-size: 0.85rem; }
  .bar { height: 4px; background: var(--line); border-radius: 4px; overflow: hidden; margin-top: 0.6rem; }
  .bar > i { display: block; height: 100%; width: 0; background: var(--gold); }
</style>
</head>
<body>
<header>
  <h1>Cite<span>.</span></h1>
  <div class="status" id="status"><span class="dot" id="dot"></span>checking Ollama…</div>
</header>
<main>
  <aside>
    <label class="pick">Add folder
      <input id="dir" type="file" webkitdirectory multiple />
    </label>
    <p class="hint">Safari and Chrome both support folder pick. Text files only (.md, .txt, source, html). Indexed in RAM for this session.</p>
    <p class="meta" id="lib">Library: empty</p>
    <div class="bar" hidden id="barwrap"><i id="bar"></i></div>
    <div class="row">
      <button class="ghost" id="clear" type="button">Clear library</button>
    </div>
    <p class="err" id="sideerr"></p>
  </aside>
  <section class="work">
    <textarea id="q" placeholder="Ask a question about the files you added…"></textarea>
    <div class="row">
      <button id="ask" type="button" disabled>Search, summarize, cite</button>
      <button class="ghost" id="reload" type="button">Recheck Ollama</button>
    </div>
    <p class="err" id="err"></p>
    <div class="answer" id="out" hidden>
      <h2>Answer</h2>
      <div id="body"></div>
      <div class="cites" id="cites"></div>
    </div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
let files = [];

async function status() {
  try {
    const s = await fetch("/api/status").then(r => r.json());
    const d = $("dot");
    if (!s.ok) {
      d.className = "dot";
      $("status").innerHTML = `<span class="dot"></span>Ollama not reachable at ${s.error || "localhost:11434"}`;
      $("ask").disabled = true;
      return;
    }
    const miss = [];
    if (!s.has_embed) miss.push(s.embed);
    if (!s.has_gen) miss.push(s.gen);
    d.className = miss.length ? "dot" : "dot on";
    $("status").innerHTML = miss.length
      ? `<span class="dot"></span>pull missing: ${miss.join(", ")}`
      : `<span class="dot on"></span><strong>${s.gen}</strong> · ${s.embed} · ${s.chunks} chunks`;
    $("ask").disabled = miss.length > 0 || s.chunks === 0;
    $("lib").textContent = `Library: ${s.chunks} chunks`;
  } catch (e) {
    $("status").textContent = "UI server error";
  }
}

function readFile(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(String(r.result || ""));
    r.onerror = () => rej(r.error);
    r.readAsText(file);
  });
}

$("dir").addEventListener("change", async (ev) => {
  $("sideerr").textContent = "";
  const list = [...ev.target.files];
  const payload = [];
  for (const f of list) {
    const name = (f.webkitRelativePath || f.name).toLowerCase();
    if (/\.(png|jpe?g|gif|webp|pdf|zip|docx|pptx|xlsx|mp[34]|mov|dmg|exe|bin)$/.test(name)) continue;
    if (f.size > 1500000) continue;
    try {
      payload.push({ path: f.webkitRelativePath || f.name, text: await readFile(f) });
    } catch (_) {}
  }
  if (!payload.length) {
    $("sideerr").textContent = "No readable text files in that folder.";
    return;
  }
  $("barwrap").hidden = false;
  $("bar").style.width = "8%";
  try {
    const r = await fetch("/api/index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ files: payload }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    $("bar").style.width = "100%";
    await status();
  } catch (e) {
    $("sideerr").textContent = e.message || String(e);
  }
  setTimeout(() => { $("barwrap").hidden = true; $("bar").style.width = "0"; }, 600);
  ev.target.value = "";
});

$("clear").onclick = async () => {
  await fetch("/api/clear", { method: "POST" });
  await status();
  $("out").hidden = true;
};

$("reload").onclick = status;

$("ask").onclick = async () => {
  const q = $("q").value.trim();
  $("err").textContent = "";
  if (!q) return;
  $("ask").disabled = true;
  $("ask").textContent = "Working…";
  try {
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ q }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    $("out").hidden = false;
    $("body").textContent = j.answer;
    $("cites").innerHTML = (j.hits || []).map((h, i) =>
      `<div class="cite"><b>[${i+1}]</b> ${escapeHtml(h.path)} <pre>${escapeHtml(h.text)}</pre></div>`
    ).join("");
  } catch (e) {
    $("err").textContent = e.message || String(e);
  }
  $("ask").disabled = false;
  $("ask").textContent = "Search, summarize, cite";
};

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = String(s);
  return d.innerHTML;
}

status();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _json(self, code: int, obj: dict):
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 40_000_000:
            raise ValueError("payload too large")
        return json.loads(self.rfile.read(n).decode("utf-8") or "{}")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            raw = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if path == "/api/status":
            self._json(200, ollama_ok())
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/clear":
                with INDEX_LOCK:
                    INDEX.clear()
                self._json(200, {"ok": True, "chunks": 0})
                return
            if path == "/api/index":
                body = self._read_json()
                files = body.get("files") or []
                added = 0
                skipped = 0
                with INDEX_LOCK:
                    if len(INDEX) >= MAX_CHUNKS:
                        self._json(400, {"error": "library full — clear it first"})
                        return
                for f in files[:MAX_FILES]:
                    p = str(f.get("path") or "file")
                    ext = Path(p).suffix.lower()
                    if ext and ext not in TEXT_EXT:
                        skipped += 1
                        continue
                    text = str(f.get("text") or "")
                    if len(text.encode("utf-8", "ignore")) > MAX_FILE_BYTES:
                        skipped += 1
                        continue
                    for ch in chunk_text(text):
                        try:
                            vec = embed(ch)
                        except Exception as e:
                            self._json(502, {"error": f"embed failed ({EMBED_MODEL}): {e}"})
                            return
                        with INDEX_LOCK:
                            if len(INDEX) >= MAX_CHUNKS:
                                break
                            INDEX.append({"path": p, "text": ch, "vec": vec})
                            added += 1
                self._json(200, {"ok": True, "chunks": len(INDEX), "added": added, "skipped": skipped})
                return
            if path == "/api/ask":
                body = self._read_json()
                q = (body.get("q") or "").strip()
                if not q:
                    self._json(400, {"error": "empty question"})
                    return
                with INDEX_LOCK:
                    if not INDEX:
                        self._json(400, {"error": "library is empty — add a folder first"})
                        return
                    snapshot = list(INDEX)
                try:
                    qv = embed(q)
                except Exception as e:
                    self._json(502, {"error": f"embed failed: {e}"})
                    return
                scored = sorted(
                    snapshot, key=lambda h: cosine(qv, h["vec"]), reverse=True
                )[:TOP_K]
                hits = [{"path": h["path"], "text": h["text"], "score": cosine(qv, h["vec"])} for h in scored]
                try:
                    answer = generate(q, hits)
                except Exception as e:
                    self._json(502, {"error": f"generate failed ({GEN_MODEL}): {e}"})
                    return
                self._json(200, {"answer": answer, "hits": hits})
                return
            self._json(404, {"error": "not found"})
        except Exception as e:
            self._json(500, {"error": str(e)})


def main():
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"Cite  →  {url}")
    print(f"embed  {EMBED_MODEL}")
    print(f"gen    {GEN_MODEL}")
    print("Ollama must be running. Ctrl+C to stop.")
    threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
        httpd.server_close()


if __name__ == "__main__":
    main()
