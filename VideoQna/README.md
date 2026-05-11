# VideoQna

VideoQna indexes a local video into Qdrant using this flow:

1. Extract full-video subtitles with Whisper.
2. Create a low-resolution proxy video and split shots with TransNetV2.
3. Select one representative keyframe per shot using K-Means with `k=1`.
4. Send only the keyframe image to a Hugging Face Qwen VLM.
5. Send the VLM frame description plus the full shot subtitles to a Qwen LLM.
6. Embed only the LLM `summary` with Qwen3 Embedding.
7. Store the vector and JSON metadata in local persistent Qdrant.

## Setup

```bash
cd VideoQna
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `HF_TOKEN`.

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

If `--transnet-weights` is omitted, the pipeline checks `TRANSNET_WEIGHTS`,
the current directory, the `VideoQna/` directory, and finally any weights bundled
inside the installed `transnetv2_pytorch` package.

## Stats

```bash
python pipeline.py stats --collection video_qna
```

## Stored Payload

Qdrant stores the vector for `summary` only. The payload also keeps:

- `video_path`, `shot_id`, `shot_start_sec`, `shot_end_sec`
- `keyframe_timestamp_sec`, `image_path`
- `frame_description`, `shot_subtitles`
- `summary`, `action`, `context`, `emotion`
- `vlm_model`, `llm_model`, `embedding_model`
