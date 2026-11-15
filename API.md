# Multimodal RAG — API Guide for Frontend Developers

This guide explains how to talk to the Multimodal RAG backend so you can build a
chat UI, a search box, or a document viewer on top of it — without knowing
anything about the ML pipeline underneath.

The backend is a **FastAPI** service. It answers questions by searching a
vector database of PDFs (datasheets like the LM317 / LM2596) and generating an
answer with a multimodal LLM. Responses include the answer text **plus** the
supporting evidence: the exact document chunks, their tables, and their figures.

- Interactive docs: `GET /docs` (Swagger UI) — try every endpoint in the browser.
- Base URL: `http://localhost:8000` locally. In production, use the deployed URL
  (e.g. `https://your-app.up.railway.app`). Everything below is relative to it.
- Versioning: every endpoint is under `/v1` except `/` and `/health`, which stay
  unversioned so uptime probes keep a fixed URL. A breaking change would go to
  `/v2` while `/v1` keeps working. The old unprefixed paths (`/query`,
  `/retrieve`, ...) were removed on 2026-11-15 and return 404.

---

## 1. The endpoints at a glance

| Endpoint | Method | Purpose | Returns |
|---|---|---|---|
| `/` | GET | Index / endpoint list | `{name, version, endpoints}` |
| `/health` | GET | Is the service up? | `{status: "ok", vector_store: "loaded"}`, or **503** when the store cannot load |
| `/v1/retrieve` | POST | Get evidence **without** an answer | `RetrieveResponse` |
| `/v1/query` | POST | Get evidence **and** a full answer | `QueryResponse` |
| `/v1/query/stream` | GET | Same as `/v1/query` but token-by-token (SSE) | Server-Sent Events |
| `/v1/images/{response_id}/{index}` | GET | Fetch one figure as an image | raw JPEG bytes |
| `/docs` | GET | Swagger UI (try-it-out docs) | HTML |

There are only **three shapes** you need to understand:

1. **Request** — always the same: `{"query": "...", "top_k": 5, "max_images": 3}` (both optional).
2. **`RetrieveResponse`** — the chunks + everything they contain.
3. **`QueryResponse`** — the same, plus `answer` and `answer_characters`.

### Errors

A failure is a real HTTP error status, never a 200 with an apology in the
answer. Every error body has the same shape: `{"detail": "..."}`.

| Status | Meaning | Example `detail` |
|---|---|---|
| `422` | The body is invalid (FastAPI's own check, `detail` is a list), or `/v1/query/custom` named a provider or model that is not in the catalogue | `"Model gpt-9 not found for provider openai"` |
| `500` | Unexpected server error; the real message is only in the server log | `"Internal server error"` |
| `502` | The model provider failed (bad key, quota, timeout), or it returned no answer text | `"The model provider failed: ..."`, or `"The model used its whole output budget (512 tokens, reasoning included) before writing any answer text. Raise max_tokens and try again."` |
| `503` | The vector store could not be loaded | `"Vector store unavailable: ..."` |

`/health` returns 503 with its usual body (`status: "error"` plus `detail`),
so an uptime probe can rely on the status code alone.

---

## 2. The request

Both `/v1/retrieve` and `/v1/query` take the same JSON body:

```json
{
  "query": "Tell me about the pin configuration of the LM317.",
  "top_k": 5
}
```

| Field | Type | Rules | Meaning |
|---|---|---|---|
| `query` | string | 1–2000 chars | The question you want answered |
| `top_k` | integer | 1–30, default 5 | How many document chunks to fetch. Every fetched chunk's text and tables go to the model, so more chunks = more evidence, but a longer prompt. |
| `max_images` | integer | 0–50, default 3 | Most figures sent to the model, taken in chunk order. 0 sends every image. Images cost far more tokens than text. A text-only model (`supports_vision` false in `config/providers.py`) is sent none, whatever this says. |

Start with the defaults; raise `top_k` if answers miss context, and raise
`max_images` (or set 0) for questions about graphs and diagrams.

---

## 3. The response — what comes back

This is what `POST /v1/query` returns (identical top-level shape for `/v1/retrieve`,
minus the two `answer` fields):

```jsonc
{
  "query": "Tell me about the pin configuration of the LM317.",
  "num_chunks": 2,                 // how many chunks were retrieved
  "total_images": 3,               // sum of image_count across chunks
  "total_tables": 2,               // sum of tables across chunks
  "response_id": "79c107621a50474989f2c3eab59beb7a",  // key for image URLs
  "answer": "The LM317 is an adjustable linear voltage regulator...",  // /v1/query only
  "answer_characters": 1086,       // /v1/query only
  "chunks": [
    {
      "chunk_id": 1,               // 1-based position within this response
      "enhanced_content": "**SEARCHABLE DESCRIPTION: LM317 Voltage Regulator...", // markdown
      "raw_text": "Pin 1 (ADJ): adjustment pin...", // plain OCR text
      "tables_html": ["<table><tr><th>Pin</th><th>Name</th>...</table>"],
      "image_count": 2,
      "images_base64": ["/9j/4AAQSkZJRg==...", "/9j/4AAQSkZJRg==..."],
      "image_urls": ["/v1/images/79c107621a50474989f2c3eab59beb7a/0",
                     "/v1/images/79c107621a50474989f2c3eab59beb7a/1"],
      "has_table": true,
      "has_image": true,
      "source": "lm317_datasheet.pdf",
      "metadata": { }
    }
  ]
}
```

### What each chunk field means for your UI

| Field | Type | How to render it |
|---|---|---|
| `enhanced_content` | string | Markdown text. Render with a markdown parser (it uses `**bold**`, headings, lists). This is the search-friendly summary. |
| `raw_text` | string | Plain OCR text from the PDF. Use for a "raw source" toggle. |
| `tables_html` | string[] | Each entry is a full `<table>...</table>` HTML string. Inject as HTML (sanitise first) or parse into your own table component. |
| `images_base64` | string[] | Base64-encoded JPEGs (note the `data:image/jpeg` prefix is **not** included — add it yourself if needed). |
| `image_urls` | string[] | Relative HTTP paths to the same images. **Prefetch and absolutise these** (see §5). |
| `has_table` / `has_image` | boolean | Quick flags to skip rendering empty sections. |
| `source` | string \| null | Source filename, if the chunk knows it. |

### The two ways to get images (important)

Every figure exists in **two formats**:

1. **Inline base64** — `images_base64[i]`. It is a **base64-encoded JPEG string**,
   without the `data:` prefix. To display:
   ```js
   `<img src="data:image/jpeg;base64,${chunk.images_base64[i]}">`
   ```
   Pros: works forever, no extra request, no server state. Cons: big JSON payloads.

2. **HTTP URL** — `image_urls[i]`, e.g. `/v1/images/79c10.../1`. Fetch it with
   `GET {base}/v1/images/79c10.../1` and you get **raw JPEG bytes**
   (`Content-Type: image/jpeg`). Use it directly in `<img src>`:
   ```js
   `<img src="${base}${chunk.image_urls[i]}">`
   ```
   Pros: tiny JSON, browser-cacheable. Cons: **expires** (see §6).

Both are the **same JPEG**, decoded differently. Pick base64 when you must keep
the data around, URLs when you want clean, lightweight JSON.

---

## 4. Streaming the answer (`/v1/query/stream`)

`GET /v1/query/stream?query=...&top_k=5` is the same as `/v1/query` but the answer
arrives as you type it. It's a standard **Server-Sent Events** stream
(`Content-Type: text/event-stream`), so in the browser you can use
`EventSource` directly — no library needed.

Because it's a GET, put the parameters in the **query string** and URL-encode:
`/v1/query/stream?query=Tell%20me%20about%20the%20pin%20configuration%20of%20the%20LM317.`

Event flow, in order:

| Event | `data` payload | When |
|---|---|---|
| `retrieval` | a full `RetrieveResponse` (the chunks) | immediately, once |
| `token` | `{"delta": "The "}` | many times, one fragment of the answer each |
| `done` | `{"answer": "...", "answer_characters": 1086, "response_id": "...", "num_chunks": 2}` | after the last token |
| `error` | `{"message": "..."}` | only if something fails after the stream started; `message` is the same text an error `detail` would carry |

```js
const es = new EventSource(
  `${BASE}/v1/query/stream?query=${encodeURIComponent("Tell me about the pin configuration of the LM317.")}&top_k=5`
);
es.addEventListener("retrieval", (e) => renderChunks(JSON.parse(e.data)));
es.addEventListener("token", (e) => appendToAnswer(JSON.parse(e.data).delta));
es.addEventListener("done", (e) => { es.close(); /* JSON.parse(e.data).answer is final */ });
es.addEventListener("error", (e) => console.error("stream failed", e));
```

Tip: build your answer by concatenating all `delta`s; the `done` event gives you
the authoritative final string.

The status is 200 as soon as the stream starts, so a failing model provider
shows up as an `error` event, not a 502. Only a store failure, which happens
before the stream starts, is a real `503`. (Planned: `POST /v1/query/stream`
will wait for the first token, so early provider failures become a real 502.)

---

## 5. Working example (plain JS)

```js
const BASE = "http://localhost:8000"; // or your deployed URL

async function ask(query, top_k = 10) {
  const res = await fetch(`${BASE}/v1/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function absolutize(url) {
  return url.startsWith("/") ? `${BASE}${url}` : url;
}

const data = await ask("Tell me about the pin configuration of the LM317.");
console.log(data.answer);                       // the generated answer
for (const chunk of data.chunks) {
  console.log(chunk.enhanced_content);          // markdown evidence
  console.log(chunk.tables_html);               // HTML table strings
  chunk.image_urls.forEach((u) => console.log(absolutize(u))); // absolute <img> URLs
}
```

**Always call `absolutize` on `image_urls`** — they are returned as relative
paths (`/v1/images/...`) and must be prefixed with the base URL before a browser
can load them.

---

## 6. Dependencies & gotchas (read this)

- **`response_id` ties everything together.** Every chunk's `image_urls` is
  `/v1/images/{response_id}/{index}`. The `index` is the figure's **global,
  0-based position across the whole response** — NOT per-chunk. Chunk 2's
  images start where chunk 1's ended. You never compute this yourself: each
  chunk's `image_urls` array is already the correct list for that chunk. Just
  take it as-is.

- **Image URLs expire.** The backend keeps figures in memory for ~30 minutes
  (`response_id` lives in a TTL cache). After that, `GET /v1/images/{id}/{index}`
  returns `404 Not Found`. This is by design (memory management) — if you must
  keep images around longer, use `images_base64` instead, or re-run the query.

- **`response_id` is per-response.** Two different queries → two different ids,
  even for the same underlying document. Don't reuse an id from a previous
  response.

- **Missing / expired images → `404`.** Always handle image load errors in your
  UI (broken-image fallback), because the 30-minute TTL is real.

- **`raw_text` and `enhanced_content` are different.** `enhanced_content` is
  the AI-summarised, markdown version (nice to read). `raw_text` is the raw OCR
  (exact source wording, uglier). Prefer `enhanced_content` for display.

- **`tables_html` is raw HTML.** The strings come straight from the PDF
  parser. If you inject them with `innerHTML`, sanitise them first.

- **Streaming and the chunks arrive separately.** In `/v1/query/stream`, the
  `retrieval` event contains the chunks; the answer tokens follow. If you want
  both in one place, combine the `retrieval` event's data with the `done`
  event's `answer`.

---

## 7. Prerequisites before you point a browser at this

- **CORS:** the API **allows browser access** from a configured allow-list of
  origins (currently `http://localhost:5173`, `http://localhost:3000`,
  `http://127.0.0.1:5173` — your local Vite dev server is covered). For a
  production frontend URL, the backend must add it to
  `CORS_ALLOWED_ORIGINS` (a comma-separated env var), e.g.
  `CORS_ALLOWED_ORIGINS=https://app.example.com,http://localhost:5173`.
  Non-browser clients are unaffected.

- **Health check:** always `GET /health` before hammering the API — the vector
  store loads lazily, so the very first request can take a few seconds.

- **Non-browser clients** (curl, Node, mobile apps) are unaffected by CORS.

---

## 8. Choosing an endpoint

| You want… | Use |
|---|---|
| A ready-to-display answer + evidence | `POST /v1/query` |
| Evidence only (build your own answer elsewhere) | `POST /v1/retrieve` |
| A typewriter/streaming answer effect | `GET /v1/query/stream` |
| The figures of a response | `images_base64` (inline) or `image_urls` → `GET /v1/images/...` |
| Just a health/probe check | `GET /health` |
