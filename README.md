# VideoQna

VideoQna indexes a local video into Qdrant using this flow:

1. Extract full-video subtitles with Whisper.
2. Create a low-resolution proxy video and split shots with TransNetV2.
3. Select one representative keyframe per shot by choosing the frame nearest
   the shot's average visual feature centroid.
4. Send keyframe images plus overlapping subtitles to a Hugging Face Router Qwen multimodal API in small batches.
5. Receive structured scene JSON with summaries, visual/dialogue keywords, and `search_text`.
6. Embed the scene `search_text` retrieval document.
7. Store the vector and JSON metadata in local persistent Qdrant.

## Setup

```bash
cd VideoQna
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

CUDA is used automatically when PyTorch and CTranslate2 can see a CUDA GPU.
`requirements.txt` pins the PyTorch `cu121` wheel so Windows installs do not
accidentally pick a newer incompatible CPU/CUDA build. You can verify the
environment with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

On Windows, faster-whisper through CTranslate2 also needs cuDNN 8 DLLs for
Whisper CUDA execution. If CTranslate2 sees the GPU but `cudnn_ops_infer64_8.dll`
is not on `PATH`, VideoQna tries the installed CUDA Toolkit `bin` directories
before falling back to CPU for Whisper. The optional faster-whisper VAD filter
imports ONNX Runtime, which can be brittle on some Windows CUDA installs, so it
is disabled by default. Add `--whisper-vad` only after ONNX Runtime imports
cleanly in your environment.

Edit `.env` and set `HF_TOKEN`. The same token is used for VLM, LLM, and embedding API calls.
Chat/VLM calls use the Hugging Face Router's OpenAI-compatible API. You can
pin a provider either in the model id or with a provider variable:

```env
HF_VLM_MODEL=Qwen/Qwen3.5-9B:together
# or:
HF_VLM_MODEL=Qwen/Qwen3.5-9B
HF_VLM_PROVIDER=together
```

If Hugging Face Router is slow or busy, transient timeout retries can appear
while indexing. They are safe if the run continues. To wait longer before a
request is treated as timed out, set:

```env
HF_CHAT_TIMEOUT=180
HF_CHAT_MAX_RETRIES=5
HF_CHAT_RETRY_BASE_DELAY=1.0
```

By default, embeddings run locally with Qwen3-Embedding-0.6B. The first run
downloads the model into the Hugging Face cache, then reuses it for indexing and
RAG queries:

```env
EMBEDDING_BACKEND=local
LOCAL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
LOCAL_EMBEDDING_DEVICE=auto
```

For Qwen3 embedding models, query embeddings are formatted with an English
retrieval instruction, while stored scene documents are embedded without an
instruction. This follows the Qwen3 embedding model card guidance for
instruction-aware retrieval.

To use a hosted embedding API instead, set `EMBEDDING_BACKEND=api` and configure
`HF_EMBEDDING_MODEL` plus `HF_EMBEDDING_PROVIDER`.

Changing embedding models changes vector dimensions. Qwen3-Embedding-0.6B writes
1024-dimensional vectors, while Qwen3-Embedding-8B writes 4096-dimensional
vectors, so use a fresh Qdrant collection or clear the old one when switching.

TransNetV2 uses the system `ffmpeg` executable when available. If it is missing,
VideoQna falls back to the `imageio-ffmpeg` binary installed by `requirements.txt`.
You can also install ffmpeg directly:

```bash
conda install -c conda-forge ffmpeg
# or, if you use Homebrew
brew install ffmpeg
```

## Index

```bash
python pipeline.py index --video ../ep_1.mp4 --collection video_qna_qwen06b --max-shots 3
```

Useful options:

```bash
python pipeline.py index \
  --video /path/to/video.mp4 \
  --collection video_qna_qwen06b \
  --whisper-model base \
  --whisper-device cuda \
  --whisper-compute-type int8 \
  --language ko \
  --transnet-threshold 0.5 \
  --transnet-device auto \
  --transnet-weights /path/to/transnetv2-pytorch-weights.pth \
  --proxy-width 320 \
  --keyframe-workers 1 \
  --api-workers 3 \
  --api-mode unified \
  --scene-batch-size 8 \
  --embedding-backend local \
  --local-embedding-model Qwen/Qwen3-Embedding-0.6B \
  --qdrant-batch-size 16 \
  --candidate-stride 0.5
```

By default, Whisper runs on CUDA with `int8` compute type. TransNetV2 keeps the
`auto` device setting and prefers CUDA when PyTorch can see the GPU.
Add `--whisper-vad` if you want faster-whisper's optional silence filtering and
your ONNX Runtime install is stable.

For Japanese videos, pass `--language ja` and consider `--whisper-model small`
or `medium` if the default `base` subtitles look garbled. You can also raise
`--whisper-beam-size` and set `WHISPER_INITIAL_PROMPT` with common names or terms
to bias transcription toward the right spellings.

Character names usually need either clear subtitle mentions or an optional
sidecar glossary. This is not required for every video; it is a per-video hint
when you know the cast or aliases. For a video `video/ep_1.mp4`, VideoQna
automatically uses the first matching UTF-8 file:
`video/ep_1.characters.txt`, `video/ep_1.cast.txt`, or
`video/ep_1.glossary.txt`. Each line can contain a name plus visual clues, for
example `탄지로 / Tanjiro / 炭治郎: 이마 흉터, 체크무늬 겉옷, 검붉은 머리`.

With `--keyframe-workers 1`, keyframes are extracted with a single-pass sampler:
the video is scanned in timestamp order to avoid repeated random seeks. Values
above `1` use parallel per-shot seeking, which can be faster on some SSDs but
may be slower for long compressed videos.

The default `--api-mode unified` sends `--scene-batch-size 8` keyframe images
plus their subtitles to one multimodal Qwen call, then embeds completed scene
`search_text` locally. Inside that single Qwen call, the prompt asks the model
to process each shot in two internal passes: first inspect the keyframe image
only, then combine that visual evidence with subtitles for the final JSON. This
avoids the older two-call `VLM -> LLM` path for each shot. `--api-workers`
controls how many unified scene batches can be in flight.
Local embeddings are shared across workers so the model is loaded once, and the
embedding stage is forced to one worker to avoid loading duplicate CUDA models.
Completed summaries are accumulated up to `LOCAL_EMBEDDING_BATCH_SIZE` before
local embedding inference. Use `--api-mode stage` or `--api-mode shot` only if
you want the older split `VLM -> LLM -> embedding` behavior.

`--qdrant-batch-size` controls how many completed shots are written per Qdrant
upsert batch. Parallel API calls reduce wall-clock waiting time but do not
reduce the number of remote VLM/LLM requests.
Completed per-shot API results are also checkpointed to `api_results.jsonl`
before Qdrant writes, so `--resume-run` can avoid repeating successful remote
calls after an interruption.

After indexing, you can run a video-level entity sweep to infer recurring
character/person candidates from all stored scene summaries and subtitles, then
attach supported names back to matching shots and re-embed only those updated
shots:

```bash
python pipeline.py entities \
  --run-dir data/runs/<run_id> \
  --collection video_qna_qwen06b
```

This is an indexing-time/post-processing step, not a query-time step. It adds
time once after indexing, but normal `/ask` retrieval stays fast. If the API
server is running against local Qdrant, stop it before this command because
local Qdrant storage cannot be opened by two Python processes at once.

The `shot_api` timing includes remote multimodal scene calls, local embedding,
and Qdrant writes. In the older `stage`/`shot` modes it also includes separate
VLM and LLM calls. Local embedding batch duration is logged separately as
`[embedding] local batch_done=... elapsed=...s`; if total `shot_api` is long but
embedding batch logs are short, the remote model calls are the bottleneck.

Indexing writes a timing report to each run directory:

```text
data/runs/<run_id>/timings.json
```

If `--transnet-weights` is omitted, the pipeline checks `TRANSNET_WEIGHTS`,
the current directory, and the `VideoQna/` directory. Otherwise it uses the
default weights loaded by the installed `transnetv2_pytorch` package.

## Stats

```bash
python pipeline.py stats --collection video_qna_qwen06b
```

## RAG API

Run the server from the `VideoQna` directory:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

Open the UI:

```text
http://localhost:8000/
```

Ask a question:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "주인공이 놀라거나 당황하는 장면은?",
    "collection": "video_qna_qwen06b",
    "top_k": 5,
    "dense_top_k": 20,
    "bm25_top_k": 20,
    "dense_workers": 3
  }'
```

The server performs:

1. Qwen LLM query expansion. It is enabled by default for better recall, uses
   `/no_think`, and caches repeated questions in the running server process.
2. Parallel dense retrieval with the configured embedder against Qdrant vectors.
3. Contextual BM25 over stored JSON payload fields. Payloads and the BM25 index
   are cached per collection while the point count is unchanged.
4. RRF fusion of dense and BM25 rankings.
5. Qwen LLM answer generation with timestamped sources.

`retrieval_debug.timings_sec` reports the per-stage latency for `/ask`.
By default, `/ask` uses LLM query expansion plus LLM answer generation. Set
`use_llm_query_expansion` to `false` only when latency matters more than recall.

For fast search-only requests, disable the remote LLM stages:

```json
{
  "use_llm_query_expansion": false,
  "generate_answer": false,
  "top_k": 4,
  "dense_top_k": 20,
  "bm25_top_k": 20
}
```

This returns ranked sources with a local fallback answer. Re-enable
`generate_answer` when you want the slower synthesized Korean answer.

## Stored Payload

Qdrant stores the vector for `summary` only. The payload also keeps:

- `video_path`, `shot_id`, `shot_start_sec`, `shot_end_sec`
- `keyframe_timestamp_sec`, `image_path`
- `frame_description`, `shot_subtitles`
- `summary`, `action`, `context`, `emotion`
- `people`, `objects`, `places`, `visual_keywords`, `dialogue_keywords`
- `search_text` used as the embedding document text
- `vlm_model`, `llm_model`, `embedding_model`, `embedding_backend`
