# """" first a just eigulai chilo when i waas just using whisper--- 
# Main script → process_input(source)

# process_input() → download_youtube_audio() → chunk_audio() → returns chunks

# transcribe_all(chunks) → loops diye jabe → transcribe_chunk() → Whisper model → text

# Output → Full transcript printed.

# """
# import whisper
# import os
# import requests
# from pydub import AudioSegment

# # Sarvam's sync STT-translate API rejects audio longer than 30s.
# # We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
# SARVAM_PIECE_SECONDS = 25


# WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")


# SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
# SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
# SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

# _model = None


# def load_model():

#     global _model  

#     if _model is None: 
#         print(f"Loading Whisper model: {WHISPER_MODEL} ...")
#         _model = whisper.load_model(WHISPER_MODEL) 
#         print("Whisper model loaded.")
#     return _model 

# def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:

#     model = load_model()

#     task = "translate" if translate else "transcribe"

#     result = model.transcribe(chunk_path, task=task)

#     return result['text']

# def transcribe_all(chunks: list, translate: bool = False) -> str:

#     full_transcript = ""

#     for i, chunk in enumerate(chunks):
#         print(f"Transcribing chunk {i+1}")
#         text = transcribe_chunk(chunk, translate=translate)

#         full_transcript += text + " "

#     print("Transcription completed")

#     return full_transcript





import whisper
import os
import requests
from pydub import AudioSegment

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

# Languages routed to Sarvam. Bengali covers both Indian and Bangladeshi Bangla —
# LID can only detect the language, not the regional dialect, so both go through
# the same path. Sarvam also supports en-IN, so a chunk that's mostly English with
# a stray bit of Hindi/Bangla still gets routed here rather than mistranscribed.
SARVAM_LANGS = {"hi", "bn"}

_model = None


def load_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _model


def detect_languages_multi(chunk_path: str, window_seconds: int = 30, samples: int = 3) -> set:
    """Sample multiple points across the full chunk (start/middle/end by default)
    instead of just the first 30s, so a language switch partway through a long
    chunk isn't missed. Returns every language detected across the sampled windows."""
    model = load_model()
    audio = AudioSegment.from_wav(chunk_path)
    duration_ms = len(audio)
    window_ms = window_seconds * 1000

    if duration_ms <= window_ms:
        offsets = [0]
    else:
        span = duration_ms - window_ms
        offsets = [int(span * i / (samples - 1)) for i in range(samples)]

    detected = set()
    for offset in offsets:
        window = audio[offset: offset + window_ms]
        tmp_path = f"{chunk_path}_lidcheck.wav"
        window.export(tmp_path, format="wav")
        try:
            raw = whisper.pad_or_trim(whisper.load_audio(tmp_path))
            mel = whisper.log_mel_spectrogram(raw, n_mels=model.dims.n_mels).to(model.device)
            _, probs = model.detect_language(mel)
            detected.add(max(probs, key=probs.get))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return detected


def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()
    result = model.transcribe(chunk_path, task="transcribe")
    return result["text"]


def _send_to_sarvam(piece_path: str) -> str:
    """Send one <=30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}
    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )
    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()
    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """Sarvam sync API only accepts <=30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts."""
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")
    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms
    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")
        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)
    return full_text.strip()


def transcribe_chunk(chunk_path: str) -> str:
    """Auto-detect language across the chunk, then route:
    English only -> free local Whisper. Hindi/Bangla anywhere -> Sarvam."""
    langs = detect_languages_multi(chunk_path)
    print(f"  Detected languages in chunk: {langs}")
    if langs & SARVAM_LANGS:
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list) -> str:
    full_transcript = ""
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        full_transcript += transcribe_chunk(chunk) + " "
    print("Transcription complete.")
    return full_transcript.strip()