from dotenv import load_dotenv
load_dotenv()   # MUST be before any core/ imports — transcriber.py reads
                # SARVAM_API_KEY at module level, not inside a function.

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractors import extract_action_items, extract_key_decisions, extract_questions
from core.rag import build_rag_chain, ask_question


def run_pipeline(source : str) -> dict:
    print("starting AI Video Assistant")

    # process_input now returns (wav_path, chunks)
    wav_path, chunks = process_input(source)

    # transcribe_all no longer takes a language argument — transcriber.py
    # detects the language of each chunk and routes to Whisper or Sarvam itself.
    transcript = transcribe_all(chunks)

    print(f"raw transcription (first 300 characters ) {transcript[:300]}")

    title = generate_title(transcript)
    summary = summarize(transcript)
    action_item = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    # source is needed so the vector store keeps this video's chunks separate
    # from every other video in the collection.
    rag_chain = build_rag_chain(transcript, source)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()

    result = run_pipeline(source)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")

    rag_chain = result["rag_chain"]

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        if not question:
            continue

        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")