from faster_whisper import WhisperModel

# Load model (small is enough)
model = WhisperModel("base", compute_type="int8")

def audio_to_text(file_path):
    segments, _ = model.transcribe(file_path)
    
    text = ""
    for segment in segments:
        text += segment.text + " "
    
    return text.strip()