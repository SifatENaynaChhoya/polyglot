
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


from utils.audio_processor import process_input
from core.transcriber import transcribe_all
import os

# Check environment and working directory
print("KEY LOADED:", os.getenv("SARVAM_API_KEY"))  # should print your key
print("CWD:", os.getcwd())

# Input source
source = "https://www.youtube.com/watch?v=Zbdrej3S7Sc"

# Process and transcribe — language is now auto-detected per chunk
chunks = process_input(source)
transcript = transcribe_all(chunks)

# Output
print("\n=== TRANSCRIPT ===\n")
print(transcript)



""""
Main script → process_input(source)

process_input() → download_youtube_audio() → chunk_audio() → returns chunks

transcribe_all(chunks) → loops diye jabe → transcribe_chunk() → Whisper model → text

Output → Full transcript printed.

"""