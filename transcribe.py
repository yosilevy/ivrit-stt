import os
import sys
import ffmpeg
import time
from faster_whisper import WhisperModel
import traceback
import logging
from logging.handlers import RotatingFileHandler

# How long to sleep (in seconds) between scans of all folders
SCAN_SLEEP_SECONDS = 120  # 2 minutes
# Minimum file size (in MB) for an MP4 to be considered for transcription
MIN_FILE_SIZE_MB = 1


def setup_logging(log_dir):
    """Set up rotating logging to a file in the given directory, redirecting print to logging.info."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "transcribe.log")
    handler = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[handler]
    )
    # Redirect print to logging.info
    import builtins
    builtins.print = lambda *args, **kwargs: logging.info(" ".join(str(a) for a in args))


def extract_audio(input_file, output_wav):
    """Extracts mono 16kHz audio from an MP4 file and returns extraction time in seconds."""
    start = time.time()
    (
        ffmpeg
        .input(input_file)
        .output(output_wav, ac=1, ar='16000')
        .overwrite_output()
        .run(quiet=True)
    )
    end = time.time()
    return end - start


def transcribe_file(model, mp4_file):
    """Extracts audio, transcribes it, and returns transcript, info, extraction time, and transcription time."""
    wav_file = mp4_file + ".wav"
    extract_time = extract_audio(mp4_file, wav_file)
    start = time.time()
    segments, info = model.transcribe(wav_file, beam_size=5)
    end = time.time()
    transcript = "\n".join([f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}" for segment in segments])
    os.remove(wav_file)
    return transcript, info, extract_time, end - start


def get_subfolders_sorted_by_date(root_folder):
    """Returns a list of subfolders under root_folder, sorted by ascending modification date."""
    subfolders = [os.path.join(root_folder, d) for d in os.listdir(root_folder)
                  if os.path.isdir(os.path.join(root_folder, d))]
    subfolders.sort(key=lambda x: os.path.getmtime(x))
    return subfolders


def get_unhandled_mp4s(folder):
    """Returns a list of MP4 files in the folder that have not been transcribed (no TXT), are >1MB, and accessible."""
    files = os.listdir(folder)
    mp4s = [f for f in files if f.lower().endswith('.mp4')]
    unhandled = []
    for mp4 in mp4s:
        mp4_path = os.path.join(folder, mp4)
        txt_path = os.path.splitext(mp4_path)[0] + '.txt'
        # Skip if transcript already exists
        if os.path.exists(txt_path):
            continue
        try:
            # Skip if not a file or too small
            if not os.path.isfile(mp4_path):
                continue
            size_mb = os.path.getsize(mp4_path) / (1024 * 1024)
            if size_mb < MIN_FILE_SIZE_MB:
                continue
            # Try opening the file to check accessibility
            with open(mp4_path, 'rb') as f:
                f.read(1)
        except Exception as e:
            print(f"Skipping inaccessible file: {mp4_path} ({e})")
            continue
        unhandled.append(mp4_path)
    return unhandled


def process_folder(model, folder):
    """Processes all unhandled MP4s in a folder, rescanning until all are handled."""
    while True:
        unhandled = get_unhandled_mp4s(folder)
        if not unhandled:
            break
        for mp4_file in unhandled:
            print(f"Transcribing {mp4_file}...")
            file_start = time.time()
            try:
                # Transcribe and save transcript
                transcript, info, extract_time, transcribe_time = transcribe_file(model, mp4_file)
                txt_path = os.path.splitext(mp4_file)[0] + ".txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"Detected language: {info.language} (probability: {info.language_probability})\n")
                    f.write(transcript)
                print(f"Saved transcript to {txt_path}")
                print(f"Audio extraction time for {mp4_file}: {extract_time:.2f} seconds")
                print(f"Transcription time for {mp4_file}: {transcribe_time:.2f} seconds (total file time: {time.time() - file_start:.2f} seconds)")
            except Exception as e:
                print(f"Error processing {mp4_file}: {e}")
                traceback.print_exc()
                continue


def main():
    """Main loop: initializes model, scans all subfolders, processes MP4s, and repeats after sleeping."""
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <root_folder>")
        sys.exit(1)
    root_folder = sys.argv[1]
    log_dir = os.path.join(root_folder, "logs")
    setup_logging(log_dir)
    print("Initializing Whisper model...")
    model_init_start = time.time()
    # Load the Whisper model once and reuse for all files
    model = WhisperModel("ivrit-ai/whisper-large-v3-turbo-ct2", device="cpu", compute_type="int8")
    model_init_end = time.time()
    print(f"Model initialized in {model_init_end - model_init_start:.2f} seconds.")

    while True:
        print(f"Scanning root folder: {root_folder}")
        # Get all subfolders sorted by date
        subfolders = get_subfolders_sorted_by_date(root_folder)
        print(f"Found {len(subfolders)} subfolders.")
        for folder in subfolders:
            print(f"Processing folder: {folder}")
            # Process all MP4s in this folder before moving to the next
            process_folder(model, folder)
        print(f"All folders scanned. Sleeping for {SCAN_SLEEP_SECONDS//60} minutes...")
        time.sleep(SCAN_SLEEP_SECONDS)

if __name__ == "__main__":
    main() 