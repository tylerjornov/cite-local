#!/usr/bin/env python3
"""Cite — local folder RAG for Ollama (stdlib only).

    python3 cite.py

Opens a browser UI. Pick folders or paste a Mac path, then ask questions.
Uses nomic-embed-text + granite4.2:3b (override with CITE_EMBED / CITE_GEN).
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

PORT = int(os.environ.get("CITE_PORT", "8787"))
OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
EMBED_MODEL = os.environ.get("CITE_EMBED", "nomic-embed-text")
GEN_MODEL = os.environ.get("CITE_GEN", "granite4.2:3b")
CHUNK_CHARS = 900
CHUNK_OVERLAP = 120
TOP_K = 6
MAX_FILES = 2500
MAX_FILE_BYTES = 1_500_000
MAX_CHUNKS = 4000
MAX_BODY = 8_000_000  # per request; the UI sends small batches

TEXT_EXT = {
    ".txt", ".md", ".mdx", ".rst", ".org", ".tex", ".html", ".htm",
    ".csv", ".tsv", ".json", ".xml", ".yml", ".yaml", ".toml", ".ini",
    ".log", ".rtf", ".py", ".js", ".ts", ".tsx", ".jsx", ".c", ".h",
    ".cpp", ".cs", ".java", ".go", ".rs", ".rb", ".php", ".swift",
    ".kt", ".sql", ".sh", ".r", ".bib", ".adoc",
}
SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__",
    ".obsidian", ".trash", "Library", ".cache",
}

INDEX_LOCK = threading.Lock()
INDEX: list[dict] = []


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
    dot = na = nb = 0.0
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


def add_file_text(path: str, text: str) -> tuple[int, str | None]:
    """Embed chunks from one file. Returns (added, error)."""
    added = 0
    if len(text.encode("utf-8", "ignore")) > MAX_FILE_BYTES:
        return 0, None
    for ch in chunk_text(text):
        with INDEX_LOCK:
            if len(INDEX) >= MAX_CHUNKS:
                return added, "full"
        try:
            vec = embed(ch)
        except Exception as e:
            return added, f"embed failed ({EMBED_MODEL}): {e}"
        with INDEX_LOCK:
            if len(INDEX) >= MAX_CHUNKS:
                return added, "full"
            INDEX.append({"path": path, "text": ch, "vec": vec})
            added += 1
    return added, None


def index_payload(files: list) -> dict:
    added = 0
    skipped = 0
    n = 0
    for f in files:
        if n >= MAX_FILES:
            skipped += 1
            continue
        p = str(f.get("path") or "file")
        ext = Path(p).suffix.lower()
        if ext and ext not in TEXT_EXT:
            skipped += 1
            continue
        n += 1
        a, err = add_file_text(p, str(f.get("text") or ""))
        added += a
        if err == "full":
            return {"ok": True, "chunks": len(INDEX), "added": added, "skipped": skipped, "full": True}
        if err:
            return {"error": err, "chunks": len(INDEX), "added": added}
    return {"ok": True, "chunks": len(INDEX), "added": added, "skipped": skipped}


def walk_local(root: Path) -> tuple[list[tuple[str, str]], int]:
    pairs: list[tuple[str, str]] = []
    skipped = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            fp = Path(dirpath) / name
            if fp.suffix.lower() not in TEXT_EXT:
                skipped += 1
                continue
            try:
                if fp.stat().st_size > MAX_FILE_BYTES:
                    skipped += 1
                    continue
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                skipped += 1
                continue
            rel = str(fp)
            pairs.append((rel, text))
            if len(pairs) >= MAX_FILES:
                return pairs, skipped
    return pairs, skipped


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
  input.path {
    width: 100%; margin-top: 0.6rem; background: var(--surface); color: var(--ink);
    border: 1px solid var(--line); border-radius: 8px; padding: 0.5rem 0.65rem; font: inherit;
  }
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
  textarea:focus, input.path:focus { outline: 2px solid var(--gold-dim); }
  .answer {
    margin-top: 1.1rem; background: var(--surface); border: 1px solid var(--line);
    border-radius: 12px; padding: 1.1rem 1.2rem;
  }
  .answer h2 { font-family: "Source Serif 4", Georgia, serif; font-size: 1.05rem; margin: 0 0 0.7rem; }
  .cites { margin-top: 1rem; }
  .cite { border-top: 1px solid var(--line); padding: 0.7rem 0; font-size: 0.8rem; }
  .cite b { color: var(--gold); font-weight: 600; }
  .cite pre { white-space: pre-wrap; font-family: inherit; color: var(--muted); margin: 0.35rem 0 0; }
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
    <input class="path" id="disk" placeholder="/Users/you/Documents/notes"/>
    <div class="row">
      <button class="ghost" id="fromdisk" type="button">Index this path</button>
    </div>
    <p class="hint">Folder pick sends files in small batches. For a huge library, paste the folder path and index from disk (no upload cap).</p>
    <p class="meta" id="lib">Library: empty</p>
    <div class="bar" hidden id="barwrap"><i id="bar"></i></div>
    <p class="meta" id="prog"></p>
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
const BATCH = 6;

async function status() {
  try {
    const s = await fetch("/api/status").then(r => r.json());
    if (!s.ok) {
      $("status").innerHTML = `<span class="dot"></span>Ollama not reachable`;
      $("ask").disabled = true;
      return s;
    }
    const miss = [];
    if (!s.has_embed) miss.push(s.embed);
    if (!s.has_gen) miss.push(s.gen);
    $("status").innerHTML = miss.length
      ? `<span class="dot"></span>pull missing: ${miss.join(", ")}`
      : `<span class="dot on"></span><strong>${s.gen}</strong> · ${s.embed} · ${s.chunks} chunks`;
    $("ask").disabled = miss.length > 0 || s.chunks === 0;
    $("lib").textContent = `Library: ${s.chunks} chunks`;
    return s;
  } catch (e) {
    $("status").textContent = "UI server error";
    return null;
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

async function postIndex(files) {
  const r = await fetch("/api/index", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ files }),
  });
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}

$("dir").addEventListener("change", async (ev) => {
  $("sideerr").textContent = "";
  const list = [...ev.target.files].filter((f) => {
    const name = (f.webkitRelativePath || f.name).toLowerCase();
    if (/\.(png|jpe?g|gif|webp|pdf|zip|docx|pptx|xlsx|mp[34]|mov|dmg|exe|bin)$/.test(name)) return false;
    if (f.size > 1500000) return false;
    return true;
  });
  if (!list.length) {
    $("sideerr").textContent = "No readable text files in that folder.";
    ev.target.value = "";
    return;
  }
  $("barwrap").hidden = false;
  try {
    for (let i = 0; i < list.length; i += BATCH) {
      const slice = list.slice(i, i + BATCH);
      const payload = [];
      for (const f of slice) {
        try {
          payload.push({ path: f.webkitRelativePath || f.name, text: await readFile(f) });
        } catch (_) {}
      }
      if (payload.length) await postIndex(payload);
      const pct = Math.round(((i + slice.length) / list.length) * 100);
      $("bar").style.width = pct + "%";
      $("prog").textContent = `Indexing ${Math.min(i + slice.length, list.length)} / ${list.length} files`;
      await status();
    }
    $("prog").textContent = "Done.";
  } catch (e) {
    $("sideerr").textContent = e.message || String(e);
  }
  setTimeout(() => { $("barwrap").hidden = true; $("bar").style.width = "0"; }, 800);
  ev.target.value = "";
});

$("fromdisk").onclick = async () => {
  const p = $("disk").value.trim();
  $("sideerr").textContent = "";
  if (!p) { $("sideerr").textContent = "Paste a folder path first."; return; }
  $("barwrap").hidden = false;
  $("bar").style.width = "30%";
  $("prog").textContent = "Reading disk… this can take a while";
  try {
    const r = await fetch("/api/index-path", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: p }),
    });
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || r.statusText);
    $("bar").style.width = "100%";
    $("prog").textContent = `Added ${j.added} chunks` + (j.full ? " (hit chunk cap)" : "");
    await status();
  } catch (e) {
    $("sideerr").textContent = e.message || String(e);
  }
  setTimeout(() => { $("barwrap").hidden = true; $("bar").style.width = "0"; }, 800);
};

$("clear").onclick = async () => {
  await fetch("/api/clear", { method: "POST" });
  await status();
  $("out").hidden = true;
  $("prog").textContent = "";
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
        if n > MAX_BODY:
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
                result = index_payload(body.get("files") or [])
                code = 200 if result.get("ok") else 502
                self._json(code, result)
                return
            if path == "/api/index-path":
                body = self._read_json()
                raw_path = str(body.get("path") or "").strip()
                if not raw_path:
                    self._json(400, {"error": "empty path"})
                    return
                root = Path(raw_path).expanduser()
                if not root.is_dir():
                    self._json(400, {"error": f"not a folder: {root}"})
                    return
                pairs, skipped = walk_local(root)
                added = 0
                for rel, text in pairs:
                    a, err = add_file_text(rel, text)
                    added += a
                    if err == "full":
                        self._json(200, {"ok": True, "chunks": len(INDEX), "added": added, "skipped": skipped, "full": True})
                        return
                    if err:
                        self._json(502, {"error": err, "added": added})
                        return
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
                scored = sorted(snapshot, key=lambda h: cosine(qv, h["vec"]), reverse=True)[:TOP_K]
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
