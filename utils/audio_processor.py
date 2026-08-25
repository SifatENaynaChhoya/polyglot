'''
If YouTube URL:
       process_input → download_youtube_audio → chunk_audio → return chunks

If Local File:
       process_input → convert_to_wav → chunk_audio → return chunks'''



import yt_dlp   #downloads vd from youtube, fb etc and extract audio
import os



from pydub import AudioSegment
from pydub.utils import which

AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe   = which("ffprobe")

from pydub import AudioSegment
AudioSegment.converter = r"C:\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
AudioSegment.ffprobe   = r"C:\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffprobe.exe"



DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR,exist_ok = True) #making the directory

def download_youtube_audio(url :str) ->str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s") #os.path.join() -> automatically uses the correct path separator
                                                                  #for your operating system: / on macOS/Linux , \ on Windows
                                                                  #When yt-dlp runs, it replaces: %(title)s → the actual video title (e.g., "Team meeting"),%(ext)s → the file extension (e.g., "wav"). and as DOWNLOAD_DIR = 'downloades' so ->file name becomes: downloades/Team meeting.wav

    #how to behave when downloading and processing media.

    settings = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "ffmpeg_location": r"C:\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe",
        "cookiesfrombrowser": ("firefox",),   # pulls your logged-in YouTube session from Edge
        "sleep_interval_requests": 2,       # small pause between requests — helps with the 429
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }


    with yt_dlp.YoutubeDL(settings) as yt_downloader:
        vd_info = yt_downloader.extract_info(url, download=True) #Returns a dictionary (vd_info) with metadata about the video and saves locally
        audio_filename = yt_downloader.prepare_filename(vd_info).replace(".webm", ".wav").replace(".m4a", ".wav") # Force the file extension to .wav after download/conversion
    return audio_filename

#print(download_youtube_audio("https://www.youtube.com/watch?v=fB2JQXEH_94"))   
#data =download_youtube_audio("https://www.youtube.com/watch?v=o126p1QN_RI&t=118s")

def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path

#print(convert_to_wav(data))
#final_data = convert_to_wav(data)

def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list: #10 min chunk
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 #chunks counts in milisec, so min to sec to milisec

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks

#print(chunk_audio(final_data))


def process_input(source: str, cleanup_intermediate: bool = True) -> tuple[str, list]:
    if source.startswith(("http://", "https://")):
        print("Detected URL. Downloading audio...")
        raw_path = download_youtube_audio(source)
    else:
        print("Detected local file.")
        raw_path = source

    probe = AudioSegment.from_file(raw_path)
    if probe.channels == 1 and probe.frame_rate == 16000:
        print("Already mono 16 kHz — skipping conversion.")
        wav_path = raw_path
    else:
        print("Normalizing to mono 16 kHz WAV...")
        wav_path = convert_to_wav(raw_path)
        if cleanup_intermediate and raw_path != source:
            os.remove(raw_path)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return wav_path, chunks