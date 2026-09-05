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

Skipped: images, PDF, Office, archives. Convert those to `.md` / `.txt` first.

## Limits (8GB-safe)

- ~400 files, ~2500 chunks, 1.5MB per file
- Context window 4096 on generate

## License

MIT
