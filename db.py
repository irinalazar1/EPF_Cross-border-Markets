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
    # One-time profile per user -- asked only once (see get_user_profile),
    # since domain experience is a stable trait, not something that changes
    # submission to submission. A moderator variable for the research
    # question: does prior EPF experience change how much the tool helps.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            username TEXT PRIMARY KEY,
            epf_experience TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    """)
    # Reflection survey shown right after a successful submission (the
    # "moment of success" -- see render_submission_survey() in app.py), not
    # a generic anytime feedback box. Linked to (username, forecast_date) so
    # it can be joined against the "feedback" table's MAE evaluation later --
    # correlating self-reported comprehension/relevance against objective
    # forecast-improvement is the actual research question this supports.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS submission_survey (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            forecast_date TEXT NOT NULL,
            usability_rating INTEGER NOT NULL,
            comprehension_rating INTEGER NOT NULL,
            context_relevance_rating INTEGER NOT NULL,
            comment TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    """)
    # One-time gate per user: research-purposes disclaimer (consented) and
    # the step-by-step tutorial (completed_tutorial). Persisted here, not
    # just session state, so it's a genuine one-time acknowledgment --
    # never re-shown on later logins once both are true.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS onboarding_status (
            username TEXT PRIMARY KEY,
            consented INTEGER NOT NULL,
            completed_tutorial INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (username) REFERENCES users(username)
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

def get_user_profile(username):
    """Returns the stored epf_experience string for this user, or None if
    they've never answered it -- used to decide whether to ask the
    experience question again (skip if already answered once)."""
    conn = get_connection()
    cursor = conn.execute("SELECT epf_experience FROM user_profile WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def save_user_profile(username, epf_experience, timestamp):
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO user_profile (username, epf_experience, timestamp) VALUES (?, ?, ?)",
        (username, epf_experience, timestamp),
    )
    conn.commit()
    conn.close()

def load_all_user_profiles():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM user_profile", conn)
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def save_submission_survey(username, forecast_date, usability_rating, comprehension_rating,
                            context_relevance_rating, comment, timestamp):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO submission_survey
            (username, forecast_date, usability_rating, comprehension_rating, context_relevance_rating, comment, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (username, str(forecast_date), int(usability_rating), int(comprehension_rating),
         int(context_relevance_rating), comment, timestamp),
    )
    conn.commit()
    conn.close()

def load_submission_survey():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM submission_survey ORDER BY timestamp DESC", conn)
    conn.close()
    if not df.empty:
        df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def get_onboarding_status(username):
    """Returns (consented, completed_tutorial) as booleans, or (False, False)
    if this user has never started onboarding."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT consented, completed_tutorial FROM onboarding_status WHERE username = ?", (username,)
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return False, False
    return bool(row[0]), bool(row[1])

def save_onboarding_status(username, consented, completed_tutorial, timestamp):
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO onboarding_status (username, consented, completed_tutorial, timestamp) VALUES (?, ?, ?, ?)",
        (username, int(consented), int(completed_tutorial), timestamp),
    )
    conn.commit()
    conn.close()