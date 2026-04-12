from faster_whisper import WhisperModel

# ✅ Load lightweight model (IMPORTANT)
model = WhisperModel("tiny", compute_type="int8")  
# tiny = FAST + works on Render

def audio_to_text(file_path):
    try:
        segments, _ = model.transcribe(file_path)

        text = ""
        for segment in segments:
            text += segment.text + " "

        return text.strip()

    except Exception as e:
        print("Whisper error:", e)
        return ""