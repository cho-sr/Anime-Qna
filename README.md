# VideoQna

<p align="center">
  <img src="assets/3조판넬%20(841%20x%201189%20mm).png" alt="VideoQna 3조판넬" width="900">
</p>

VideoQna는 긴 영상을 업로드하면 장면 단위로 인덱싱하고, 사용자가 자연어로 질문했을 때 관련 장면과 답변을 찾아주는 비디오 RAG 시스템입니다.

## 주요 기능

- 웹 UI에서 영상 업로드 및 백그라운드 인덱싱
- Whisper 기반 전체 자막 추출
- TransNetV2 기반 shot/scene 분할
- shot별 keyframe 추출 및 768px 다운스케일 이미지 저장
- Hugging Face Router 기반 Qwen multimodal scene summary 생성
- Qwen3 local embedding과 Qdrant 저장
- Dense 검색 + BM25 검색 + RRF fusion 기반 장면 검색
- 검색된 장면을 바탕으로 한국어 답변 생성
- 업로드 완료 후 warm-up으로 첫 질문 지연 감소
- 등장인물 이름/별칭/외형 힌트 파일 지원

## 파이프라인

```text
Video Upload
-> Whisper subtitles
-> TransNet shot detection
-> Keyframe extraction
-> VLM scene summary
-> Local embedding
-> Qdrant
-> RAG answer
```

## 설치

```bash
cd VideoQna
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

`.env`에서 최소한 `HF_TOKEN`을 설정합니다.

```env
HF_TOKEN=your_huggingface_token
HF_VLM_MODEL=Qwen/Qwen3.5-9B:together
HF_LLM_MODEL=Qwen/Qwen3.5-9B:together
EMBEDDING_BACKEND=local
LOCAL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
LOCAL_EMBEDDING_DEVICE=auto
```

CUDA 확인:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

## 서버 실행

반드시 `VideoQna` 폴더에서 실행합니다.

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

브라우저에서 접속:

```text
http://127.0.0.1:8000/
```

## 웹 사용법

1. `Video Upload`에서 영상 파일을 선택합니다.
2. `New Collection`은 비워두면 자동 생성됩니다.
3. `Lang`을 선택합니다.
   - `auto`: Whisper 자동 감지
   - `ja`: 일본어 영상
   - `ko`: 한국어 영상
   - `en`: 영어 영상
4. `업로드 인덱싱`을 누릅니다.
5. 상태가 `done`이 되면 질문할 수 있습니다.

웹 UI에서는 복잡한 튜닝 옵션을 숨기고 안정 기본값을 사용합니다.

```text
api_workers = 15
scene_batch_size = 5
qdrant_batch_size = 16
max_shots = 0
top_k = 4
dense_top_k = 20
bm25_top_k = 20
LLM query expansion = on
LLM answer = on
```

## 인덱싱 결과 위치

```text
data/uploads/<job_id>/                 업로드된 원본 영상
data/index_jobs/<job_id>/              인덱싱 상태와 로그
data/runs/<run_id>/                    subtitles, shots, timings, scene records
data/keyframes/<run_id>/               shot별 keyframe 이미지
data/qdrant_uploads/<collection>/      업로드 영상용 Qdrant 저장소
```

서버는 다음 Qdrant 경로를 자동 탐색합니다.

```text
data/qdrant
data/qdrant*
data/qdrant_uploads/<collection>
```

그래서 collection을 바꿀 때마다 서버를 다시 열 필요가 없습니다.

## Warm-Up

첫 질문이 느린 이유는 로컬 임베딩 모델 로딩, CUDA 초기화, BM25 캐시 생성이 처음 한 번 발생하기 때문입니다. 업로드 인덱싱이 끝나면 서버가 자동으로 해당 collection을 warm-up합니다.

수동 warm-up:

```bash
curl -X POST http://127.0.0.1:8000/warmup/video_qna_ep_3_6e49e5
```

서버를 재시작하면 메모리 캐시가 사라지므로 다시 warm-up이 필요할 수 있습니다.

## 등장인물 이름 힌트

영상만 보고 인물의 실제 이름을 확정하는 것은 어렵습니다. 이름 정확도를 높이려면 영상 파일 옆에 UTF-8 텍스트 파일을 둡니다.

자동 탐색 파일명:

```text
video/ep_3.characters.txt
video/ep_3.cast.txt
video/ep_3.glossary.txt
```

예시:

```text
유키: 검은 머리 여학생, 교복, 주인공
하루: 금발 남학생, 키가 큼
미카: 짧은 갈색 머리, 안경
```

이 힌트는 VLM/LLM scene summary 단계에서 인물명을 보수적으로 붙이는 데 사용됩니다. 자막에 이름이 나오거나 외형 힌트가 맞는 경우에만 이름 연결이 안정적입니다.

## REST API

### Health

```bash
curl http://127.0.0.1:8000/health
```

### Upload

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "video=@video/ep_3.mp4" \
  -F "collection=video_qna_ep3" \
  -F "language=ja"
```

### Job Status

```bash
curl http://127.0.0.1:8000/jobs/<job_id>
```

### Stats

```bash
curl http://127.0.0.1:8000/stats/video_qna_ep3
```

### Ask

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "주인공이 놀라는 장면은?",
    "collection": "video_qna_ep3"
  }'
```

## CLI 인덱싱

웹 UI 대신 직접 CLI로 인덱싱할 수도 있습니다.

```bash
python pipeline.py index \
  --video video/ep_3.mp4 \
  --collection video_qna_ep3 \
  --qdrant-path data/qdrant_ep3 \
  --language ja \
  --api-mode unified \
  --api-workers 15 \
  --scene-batch-size 5 \
  --qdrant-batch-size 16
```

테스트용 일부 shot만 처리:

```bash
python pipeline.py index \
  --video video/ep_3.mp4 \
  --collection test_ep3_20 \
  --max-shots 20 \
  --api-workers 15 \
  --scene-batch-size 5
```

기본 keyframe은 768px 다운스케일 이미지 1장만 저장합니다. 답변 화면에도 이 이미지가 표시됩니다. 원본 해상도 keyframe도 함께 저장하려면 다음 옵션을 추가합니다.

```bash
--save-original-keyframes
```

## Entity Sweep

인덱싱 후 자막과 scene summary를 기반으로 반복 등장 인물 후보를 추론하고 Qdrant record를 보강할 수 있습니다.

```bash
python pipeline.py entities \
  --run-dir data/runs/<run_id> \
  --collection video_qna_ep3 \
  --qdrant-path data/qdrant_ep3
```

로컬 Qdrant는 같은 path를 두 Python 프로세스가 동시에 열 수 없습니다. API 서버가 같은 Qdrant path를 사용 중이면 서버를 끄고 실행합니다.

## 성능 메모

실측 예시:

```text
23.7분 영상 / 약 345 shots
총 인덱싱: 약 10~15분

Whisper: 약 1~2분
TransNet: 약 2분
Keyframes: 약 3~6분
VLM/API: 약 3~5분
```

병목은 보통 `keyframes`와 `shot_api`입니다. 현재 안정 설정은 다음과 같습니다.

```text
api_workers = 15
scene_batch_size = 5
```

`scene_batch_size`를 6 이상으로 올리면 Hugging Face Router에서 `400 Input validation error`가 날 수 있어 기본값 5를 권장합니다.

각 run의 타이밍은 다음 파일에 저장됩니다.

```text
data/runs/<run_id>/timings.json
```

## Troubleshooting

### `Could not import module "api_server"`

`VideoQna` 폴더에서 서버를 실행해야 합니다.

```bash
cd VideoQna
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

### Qdrant collection을 못 찾는 경우

서버는 여러 Qdrant 경로를 자동 탐색합니다. 그래도 안 잡히면 `/health`에서 `qdrant_search_paths`를 확인합니다.

```bash
curl http://127.0.0.1:8000/health
```

### 첫 질문이 느린 경우

첫 질문 전에 warm-up을 실행합니다.

```bash
curl -X POST http://127.0.0.1:8000/warmup/<collection>
```

### HF `400 Input validation error`

한 번에 넣는 이미지/텍스트가 많을 때 발생할 수 있습니다. `scene_batch_size=5`를 사용합니다.

### 로컬 Qdrant lock 오류

같은 Qdrant path를 서버와 CLI가 동시에 열면 lock 오류가 납니다. 서버를 끄거나 다른 `--qdrant-path`를 사용합니다.

## Stored Payload

Qdrant payload에는 다음 정보가 저장됩니다.

- `video_path`, `shot_id`, `shot_start_sec`, `shot_end_sec`
- `keyframe_timestamp_sec`, `image_path`
- `frame_description`, `shot_subtitles`
- `summary`, `action`, `context`, `emotion`
- `people`, `objects`, `places`
- `visual_keywords`, `dialogue_keywords`
- `character_candidates`
- `search_text`
- `vlm_model`, `llm_model`, `embedding_model`, `embedding_backend`
