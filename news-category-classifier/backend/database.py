import sqlite3
from datetime import datetime

DB_NAME = "predictions.db"

def get_connection():
    return sqlite3.connect(DB_NAME)


def create_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT,
            predicted_category TEXT,
            confidence REAL,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()



def insert_prediction(input_text, predicted_category, confidence):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (input_text, predicted_category, confidence, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        input_text,
        predicted_category,
        confidence,
        datetime.utcnow().isoformat()
    ))

    conn.commit()
    conn.close()
