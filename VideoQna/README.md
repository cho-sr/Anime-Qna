# VideoQna

VideoQna indexes a local video into Qdrant using this flow:

1. Extract full-video subtitles with Whisper.
2. Create a low-resolution proxy video and split shots with TransNetV2.
3. Select one representative keyframe per shot using K-Means with `k=1`.
4. Send only the keyframe image to a Hugging Face Router Qwen VLM API.
5. Send the VLM frame description plus the full shot subtitles to a Qwen LLM API.
6. Embed only the LLM `summary` with a Hugging Face feature-extraction API.
7. Store the vector and JSON metadata in local persistent Qdrant.

## Setup

```bash
cd VideoQna
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `HF_TOKEN`. The same token is used for VLM, LLM, and embedding API calls.
Chat/VLM calls use the Hugging Face Router's OpenAI-compatible API. You can
pin a provider either in the model id or with a provider variable:

```env
HF_VLM_MODEL=Qwen/Qwen3.5-9B:together
# or:
HF_VLM_MODEL=Qwen/Qwen3.5-9B
HF_VLM_PROVIDER=together
```

Embedding calls use Hugging Face feature extraction through `huggingface_hub`.
The default `.env.example` uses an `hf-inference` multilingual embedding model.
If you want to keep Qwen3 embeddings, use a provider/model pair that supports
feature extraction, for example:

```env
HF_EMBEDDING_PROVIDER=scaleway
HF_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
```

TransNetV2 also needs the system `ffmpeg` executable:

```bash
conda install -c conda-forge ffmpeg
# or, if you use Homebrew
brew install ffmpeg
```

## Index

```bash
python pipeline.py index --video ../ep_1.mp4 --collection video_qna --max-shots 3
```

Useful options:

```bash
python pipeline.py index \
  --video /path/to/video.mp4 \
  --collection video_qna \
  --whisper-model base \
  --language ko \
  --transnet-threshold 0.5 \
  --transnet-weights /path/to/transnetv2-pytorch-weights.pth \
  --proxy-width 320 \
  --candidate-stride 0.5
```

Indexing writes a timing report to each run directory:

```text
data/runs/<run_id>/timings.json
```

If `--transnet-weights` is omitted, the pipeline checks `TRANSNET_WEIGHTS`,
the current directory, the `VideoQna/` directory, and finally any weights bundled
inside the installed `transnetv2_pytorch` package.

## Stats

```bash
python pipeline.py stats --collection video_qna
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
    "collection": "video_qna",
    "top_k": 5,
    "dense_top_k": 20,
    "bm25_top_k": 20
  }'
```

The server performs:

1. Qwen LLM query expansion.
2. Dense retrieval with the configured embedding API against Qdrant vectors.
3. Contextual BM25 over stored JSON payload fields.
4. RRF fusion of dense and BM25 rankings.
5. Qwen LLM answer generation with timestamped sources.

## Stored Payload

Qdrant stores the vector for `summary` only. The payload also keeps:

- `video_path`, `shot_id`, `shot_start_sec`, `shot_end_sec`
- `keyframe_timestamp_sec`, `image_path`
- `frame_description`, `shot_subtitles`
- `summary`, `action`, `context`, `emotion`
- `vlm_model`, `llm_model`, `embedding_model`
