"""
pipeline.py
전체 파이프라인 CLI 실행 스크립트

사용법:
  # 영상 인덱싱 (Whisper STT)
  python pipeline.py index --video anime.mp4 --title "원피스 EP1" --method whisper

  # 영상 인덱싱 (SRT 자막 파일 사용)
  python pipeline.py index --video anime.mp4 --title "원피스 EP1" --method srt --srt subtitle.srt

  # Q&A
  python pipeline.py ask --title "원피스 EP1" --question "루피가 처음 기어 2를 쓴 장면은?"

  # 대화형 QA 모드
  python pipeline.py chat --title "원피스 EP1"
"""

import argparse
import sys
from dotenv import load_dotenv

load_dotenv()


def cmd_index(args):
    from video_processor import VideoProcessor
    from subtitle_extractor import SubtitleExtractor
    from vector_store import AnimeVectorStore, ChapterBuilder

    print(f"\n{'='*50}")
    print(f"🎬 인덱싱 시작: {args.title}")
    print(f"{'='*50}")

    # 1. 프레임 추출
    processor = VideoProcessor(
        frame_interval=args.frame_interval,
        chapter_duration=args.chapter_duration,
        save_frames=(args.method == "ocr")  # OCR 모드에서만 저장
    )
    frames = processor.extract_frames(args.video, output_dir=f"frames/{args.title}")

    # 2. 자막 추출
    extractor = SubtitleExtractor(
        method=args.method,
        model_size=getattr(args, "whisper_model", "base"),
        language=getattr(args, "language", None),
        languages=["ko", "ja", "en"]
    )
    subtitle_chunks = extractor.extract(
        video_path=args.video,
        srt_path=getattr(args, "srt", None),
        frames=frames if args.method == "ocr" else None,
        chapter_duration=args.chapter_duration
    )

    if not subtitle_chunks:
        print("❌ 자막을 추출하지 못했습니다. 방법을 확인하세요.")
        sys.exit(1)

    # 3. 챕터 분석
    builder = ChapterBuilder(chapter_duration=args.chapter_duration)
    chapters = builder.build_chapters(subtitle_chunks)
    chapters = builder.process_all_chapters(chapters)

    # 4. 벡터 DB 저장
    store = AnimeVectorStore(db_path=args.db_path)
    store.index_chapters(chapters, anime_title=args.title)

    print(f"\n🎉 인덱싱 완료: {len(chapters)}개 챕터 저장됨")


def cmd_ask(args):
    from vector_store import AnimeVectorStore
    from qa_engine import AnimeQAEngine

    store = AnimeVectorStore(db_path=args.db_path)
    engine = AnimeQAEngine(vector_store=store)

    result = engine.ask(question=args.question, anime_title=args.title)

    print(f"\n{'='*50}")
    print(f"❓ 질문: {result.question}")
    print(f"{'='*50}")
    print(f"💬 답변:\n{result.answer}")
    print(f"\n📍 관련 타임스탬프: {', '.join(result.timestamps)}")


def cmd_chat(args):
    from vector_store import AnimeVectorStore
    from qa_engine import AnimeQAEngine

    store = AnimeVectorStore(db_path=args.db_path)
    engine = AnimeQAEngine(vector_store=store)

    print(f"\n🎌 [{args.title}] 대화형 Q&A 모드 (종료: 'quit' 또는 Ctrl+C)")
    print("="*50)

    history = []
    while True:
        try:
            question = input("\n❓ 질문: ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "종료"):
                break

            result = engine.ask_with_history(question, args.title, history)
            print(f"\n💬 답변:\n{result.answer}")
            print(f"📍 참조: {', '.join(result.timestamps)}")

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": result.answer})

        except KeyboardInterrupt:
            break

    print("\n👋 종료합니다.")


def cmd_list(args):
    from vector_store import AnimeVectorStore
    store = AnimeVectorStore(db_path=args.db_path)
    animes = store.list_animes()
    if animes:
        print("\n📚 등록된 애니메이션:")
        for a in animes:
            print(f"  - {a}")
    else:
        print("등록된 애니메이션이 없습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anime RAG Q&A Pipeline")
    parser.add_argument("--db-path", default="./chroma_db", help="ChromaDB 경로")

    subparsers = parser.add_subparsers(dest="command")

    # index 커맨드
    p_index = subparsers.add_parser("index", help="영상 인덱싱")
    p_index.add_argument("--video", required=True, help="영상 파일 경로")
    p_index.add_argument("--title", required=True, help="애니메이션 제목")
    p_index.add_argument("--method", choices=["whisper", "srt", "ocr"], default="whisper")
    p_index.add_argument("--srt", help="SRT 자막 파일 경로 (method=srt 시)")
    p_index.add_argument("--whisper-model", default="base", help="Whisper 모델 크기")
    p_index.add_argument("--language", default=None, help="언어 코드 (ja/ko/en)")
    p_index.add_argument("--frame-interval", type=float, default=2.0)
    p_index.add_argument("--chapter-duration", type=float, default=60.0)

    # ask 커맨드
    p_ask = subparsers.add_parser("ask", help="단발성 질문")
    p_ask.add_argument("--title", required=True, help="애니메이션 제목")
    p_ask.add_argument("--question", required=True, help="질문")

    # chat 커맨드
    p_chat = subparsers.add_parser("chat", help="대화형 Q&A")
    p_chat.add_argument("--title", required=True, help="애니메이션 제목")

    # list 커맨드
    p_list = subparsers.add_parser("list", help="등록된 애니메이션 목록")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()
