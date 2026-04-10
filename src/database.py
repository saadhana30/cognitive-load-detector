import sqlite3

DB_NAME = "feedback.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction TEXT,
            feedback TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_feedback(prediction, feedback):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO feedback (prediction, feedback) VALUES (?, ?)",
        (prediction, feedback)
    )

    conn.commit()
    conn.close()