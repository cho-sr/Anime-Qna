# 🎌 AnimeMind — RAG 기반 애니메이션 Q&A 시스템

> 캡스톤 디자인 프로젝트: 애니메이션 영상에서 자막을 추출하고 벡터 DB에 저장하여 LLM 기반 Q&A를 제공하는 시스템

---

## 📐 시스템 아키텍처

```
영상 파일 (.mp4)
    │
    ▼
[1. 프레임 추출] video_processor.py
    - OpenCV로 N초 간격 프레임 분할
    - 챕터 ID 자동 부여 (N분 단위)
    │
    ▼
[2. 자막 추출] subtitle_extractor.py
    ├─ Whisper STT   (faster-whisper: 음성 → 텍스트)
    ├─ SRT 파싱      (기존 자막 파일 활용)
    └─ OCR           (EasyOCR: 화면 자막 인식)
    │
    ▼
[3. 챕터 분석] vector_store.py → ChapterBuilder
    - 자막을 챕터 단위로 그룹핑
    - GPT-4o-mini로 각 챕터 개요 + 이벤트 목록 생성
    │
    ▼
[4. 벡터 DB 저장] vector_store.py → AnimeVectorStore
    - ChromaDB에 챕터별 임베딩 저장
    - 메타데이터: chapter_id, start/end_time, summary, events
    │
    ▼
[5. Q&A] qa_engine.py → AnimeQAEngine
    - 질문 → 벡터 검색 (Top-K 챕터 검색)
    - 검색된 챕터 컨텍스트 + GPT-4o-mini 답변 생성
    │
    ▼
[웹 UI / API] index.html + api_server.py
```

---

## 📁 파일 구조

```
anime_qa_system/
├── video_processor.py     # 영상 → 프레임 추출
├── subtitle_extractor.py  # 자막 추출 (Whisper / SRT / OCR)
├── vector_store.py        # ChromaDB 저장/검색 + 챕터 분석
├── qa_engine.py           # RAG Q&A 엔진
├── api_server.py          # FastAPI 백엔드
├── pipeline.py            # CLI 실행 스크립트
├── index.html             # 웹 UI
├── requirements.txt       # 의존성
└── .env.example           # 환경변수 예시
```

---

## 🚀 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY 입력
```

### 2. (선택) Tesseract OCR 설치 - OCR 모드 사용 시
```bash
# Ubuntu
sudo apt-get install tesseract-ocr tesseract-ocr-jpn tesseract-ocr-kor

# macOS
brew install tesseract tesseract-lang
```

### 3. 실행 방법

#### A. CLI 사용 (권장 - 개발/테스트)
```bash
# 영상 인덱싱 (Whisper)
python pipeline.py index \
  --video "원피스_ep1.mp4" \
  --title "원피스 EP1" \
  --method whisper \
  --whisper-model base \
  --language ja

# SRT 파일 사용
python pipeline.py index \
  --video "anime.mp4" \
  --title "나루토 EP1" \
  --method srt \
  --srt "naruto_ep1.srt"

# 단발 질문
python pipeline.py ask \
  --title "원피스 EP1" \
  --question "루피가 샹크스를 만나는 장면은 몇 분?"

# 대화형 모드
python pipeline.py chat --title "원피스 EP1"

# 등록 목록 확인
python pipeline.py list
```

#### B. 웹 UI + API 서버
```bash
# API 서버 실행
python api_server.py

# 브라우저에서 index.html 열기 (또는 Live Server 사용)
```

---

## 🔧 주요 설정값

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `frame_interval` | 2.0초 | 프레임 추출 간격 |
| `chapter_duration` | 60.0초 | 챕터 길이 |
| `whisper_model` | base | tiny/base/small/medium/large |
| `top_k` | 3 | 검색 시 참조 챕터 수 |

---

## 💡 성능 팁

- **일본어 애니메이션**: `--language ja` 옵션 추가 시 Whisper 정확도 향상
- **자막 파일 있을 때**: `--method srt`가 가장 빠르고 정확함
- **GPU 있을 때**: `WhisperExtractor(device="cuda")`로 변경 시 10x 빠름
- **chapter_duration**: 장면 전환이 빠른 애니는 30초, 긴 에피소드는 120초 권장

---

## 🧠 기술 스택

| 역할 | 라이브러리 |
|------|-----------|
| 프레임 추출 | OpenCV |
| 음성 인식 | faster-whisper |
| OCR | EasyOCR |
| 임베딩 | OpenAI text-embedding-3-small |
| 벡터 DB | ChromaDB |
| LLM | GPT-4o-mini (LangChain) |
| API | FastAPI |
