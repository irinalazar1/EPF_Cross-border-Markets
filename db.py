import sqlite3
import pandas as pd

DB_PATH = "epf_dashboard.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expert_id TEXT NOT NULL,
            forecast_date TEXT NOT NULL,
            timestamp_slot TEXT NOT NULL,
            forecast REAL NOT NULL,
            adjusted REAL NOT NULL,
            flagged INTEGER NOT NULL,
            load_fr REAL,
            confidence INTEGER,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (expert_id) REFERENCES users(username)
        )
    """)
    conn.commit()
    conn.close()

def load_users():
    conn = get_connection()
    cursor = conn.execute("SELECT username, email, password_hash, role FROM users")
    rows = cursor.fetchall()
    conn.close()
    return {u: {"email": e, "password": p, "role": r} for u, e, p, r in rows}

def save_new_user(username, password_hash, email, role):
    conn = get_connection()
    conn.execute(
        "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
        (username, email.strip().lower(), password_hash, role),
    )
    conn.commit()
    conn.close()

def save_feedback(rows_df):
    conn = get_connection()
    for _, row in rows_df.iterrows():
        conn.execute(
            """
            INSERT INTO feedback
                (expert_id, forecast_date, timestamp_slot, forecast, adjusted, flagged, load_fr, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["expert_id"], str(row["forecast_date"]), str(row["timestamp_slot"]),
                float(row["forecast"]), float(row["adjusted"]), int(bool(row["flagged"])),
                float(row["load_fr"]), int(row["confidence"]),
                row["timestamp"],
            ),
        )
    conn.commit()
    conn.close()

def load_feedback():
    conn = get_connection()
    df_feedback = pd.read_sql_query("SELECT * FROM feedback", conn)
    conn.close()
    if not df_feedback.empty:
        df_feedback["forecast_date"] = pd.to_datetime(df_feedback["forecast_date"]).dt.date
        df_feedback["timestamp_slot"] = pd.to_datetime(df_feedback["timestamp_slot"])
        df_feedback["flagged"] = df_feedback["flagged"].astype(bool)
    return df_feedback

def has_submitted(expert_id, forecast_date):
    conn = get_connection()
    cursor = conn.execute(
        "SELECT COUNT(*) FROM feedback WHERE expert_id = ? AND forecast_date = ?",
        (expert_id, str(forecast_date)),
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0