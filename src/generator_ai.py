from transformers import pipeline

# ❌ DO NOT load at top
summarizer = None
question_generator = None


def load_models():
    global summarizer, question_generator

    if summarizer is None:
        summarizer = pipeline("summarization")

    if question_generator is None:
        question_generator = pipeline("text-generation", model="gpt2")


def generate_summary(text):
    load_models()
    try:
        result = summarizer(text, max_length=60, min_length=20, do_sample=False)
        return result[0]["summary_text"]
    except:
        return "Summary generation failed."


def generate_quiz(text):
    load_models()
    try:
        prompt = f"Generate 3 questions from this text:\n{text}\nQuestions:"
        result = question_generator(prompt, max_length=100, num_return_sequences=1)
        return result[0]["generated_text"]
    except:
        return "Quiz generation failed."