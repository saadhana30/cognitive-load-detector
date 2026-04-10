import whisper

# load once (fast reuse)
model = whisper.load_model("base")  # you can use "small" if needed

def audio_to_text(file_path):
    try:
        result = model.transcribe(file_path)
        return result["text"]
    except:
        return ""