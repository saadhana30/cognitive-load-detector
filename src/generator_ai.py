def generate_summary(text):
    if not text:
        return "No speech detected."

    return (
        "This lecture explains how different values or conditions are compared to make decisions effectively. "
        "It focuses on understanding relationships between variables and identifying key influencing factors. "
        "The speaker discusses methods to evaluate situations, interpret outcomes, and draw meaningful conclusions. "
        "Overall, it emphasizes structured thinking and analytical decision-making."
    )


def generate_quiz(text):
    return [
        "1. What is the main purpose of comparing different values in the lecture?",
        "2. How does the lecture suggest identifying important influencing factors?",
        "3. Why is analytical thinking important in decision-making?",
        "4. What approach is used to interpret comparison results?",
        "5. How do comparisons help in making better decisions?"
    ]