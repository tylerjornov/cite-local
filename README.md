# Cite

Local **search → summarize → cite** over folders on your Mac. One Python 3 file, no pip. Uses the Ollama models you already have:

| Role | Model |
|---|---|
| Embed (index) | `nomic-embed-text` |
| Answer + citations | `granite4.2:3b` |

## Run

Ollama must already be running. Then:

```bash
python3 cite.py
```

A browser tab opens at `http://127.0.0.1:8787`. Pick a folder, wait for indexing, ask a question.

Optional:

```bash
CITE_GEN=qwen3.5:4b python3 cite.py     # different generator
CITE_PORT=9000 python3 cite.py
```

On 8GB RAM: keep **one** generator loaded. Cite defaults to Granite 3B. Close other Ollama chats first.

```bash
ollama stop llama3.2:3b
ollama stop qwen3.5:4b
ollama stop phi4-mini
```

## What it does

1. You choose folder(s) in the browser (`webkitdirectory` — Safari and Chrome).
2. Text files are chunked and embedded with `nomic-embed-text`.
3. A question retrieves the closest chunks (cosine).
4. `granite4.2:3b` answers **only from those chunks** and tags `[1]`, `[2]`, … matching the source quotes below the answer.

Index lives in RAM for the session. Clearing the library or quitting the process drops it. No cloud.

Folder pick used to POST the entire library in one request and died with **payload too large**. Files now go in batches of 6. For a large library, paste the path (e.g. `/Users/you/Documents/notes`) and click **Index this path** — Python reads the disk directly, so nothing is uploaded.

Caps: 2500 files, 4000 chunks, 1.5MB per file. Hidden dirs / `node_modules` / `.git` are skipped.

## Limits (8GB-safe)

- ~400 files, ~2500 chunks, 1.5MB per file
- Context window 4096 on generate

## License

MIT
