import streamlit as st
import datetime as dt
import os
import io
import re
from datetime import date, datetime, timedelta, timezone

import bcrypt
import holidays
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dotenv import load_dotenv

from db import init_db, load_users, save_new_user, save_feedback, load_feedback, has_submitted

load_dotenv()

# Run the Streamlit Dashboard using 'streamlit run app.py'

st.set_page_config(page_title='EPF Expert Review', layout='wide')
st.title('EPF Expert Review')

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = "margaridamascarenhas"
GITHUB_REPO = "DAM_Forecast_V4"
GITHUB_BRANCH = "main"

DNN_FILE = "DNN_forecasts_10AM.csv"
QR_FILE = "QR_forecasts_10AM.csv"
BE_DATA_FILE = "Data_BE_UTC.csv"

STEPS_PER_DAY = 96  # 15-minute resolution: 24h * 4

EXPERT_ROLES = ["expert"]  # roles selectable at self-registration


# --------------------------------------------------------------------------
# GITHUB DATA LOADING (live, cached with a TTL so daily 10AM/14h updates get picked up)
# --------------------------------------------------------------------------

def fetch_csv_from_github(owner, repo, branch, path, fname, token, usecols=None):
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}/{fname}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.get(raw_url, headers=headers, timeout=30)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), usecols=usecols)


@st.cache_data(ttl=1800)  # 30 min -- catches both the 10AM and 14h daily refreshes without hammering GitHub
def get_dnn_df():
    # Only 3 columns exist in this file already -- nothing to trim.
    df = fetch_csv_from_github(GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, "Forecast", DNN_FILE, GITHUB_TOKEN)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["date_only"] = df["DateTime"].dt.date
    return df.sort_values("DateTime").reset_index(drop=True)


@st.cache_data(ttl=1800)
def get_qr_df():
    # The real file has 41 columns (3 calibration windows x 13 quantiles each);
    # only the expanding-window bands we actually plot are fetched/parsed.
    cols = ["DateTime", "Imputed", "QR_expanding_q0.01", "QR_expanding_q0.1",
            "QR_expanding_q0.9", "QR_expanding_q0.99"]
    df = fetch_csv_from_github(GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, "Forecast", QR_FILE, GITHUB_TOKEN,
                                usecols=cols)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["date_only"] = df["DateTime"].dt.date
    return df.sort_values("DateTime").reset_index(drop=True)


@st.cache_data(ttl=1800)
def get_be_df():
    # Already a lean 9-column file -- nothing to trim.
    df = fetch_csv_from_github(GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, "datasets", BE_DATA_FILE, GITHUB_TOKEN)
    df["Date"] = pd.to_datetime(df["Date"])
    df["date_only"] = df["Date"].dt.date
    return df.sort_values("Date").reset_index(drop=True)


init_db()


# --------------------------------------------------------------------------
# SHARED HELPER FUNCTIONS
# --------------------------------------------------------------------------

def get_available_dates(dnn_df):
    """Dates with a complete 96-slot DNN forecast."""
    counts = dnn_df.groupby("date_only").size()
    return sorted(counts[counts == STEPS_PER_DAY].index)


def dnn_forecast(forecast_date, dnn_df):
    """Returns (values_array, timestamps) for a 96-slot day, or (None, None)."""
    day_rows = dnn_df[dnn_df["date_only"] == forecast_date].sort_values("DateTime")
    if len(day_rows) != STEPS_PER_DAY:
        return None, None
    return day_rows["DNN_expanding"].values, day_rows["DateTime"].values


def dnn_imputed_flags(forecast_date, dnn_df):
    """Returns the Imputed flag array for the day, or None."""
    day_rows = dnn_df[dnn_df["date_only"] == forecast_date].sort_values("DateTime")
    if len(day_rows) != STEPS_PER_DAY:
        return None
    return day_rows["Imputed"].values


def qr_uncertainty_bands(forecast_date, qr_df):
    """
    Two nested QR-based bands for forecast_date:
      inner: q0.1-q0.9 (80% interval)
      outer: q0.01-q0.99 (98% interval)
    Returns a dict of four 96-value arrays, or None.
    """
    day_rows = qr_df[qr_df["date_only"] == forecast_date].sort_values("DateTime")
    if len(day_rows) != STEPS_PER_DAY:
        return None
    return {
        "inner_lower": day_rows["QR_expanding_q0.1"].values,
        "inner_upper": day_rows["QR_expanding_q0.9"].values,
        "outer_lower": day_rows["QR_expanding_q0.01"].values,
        "outer_upper": day_rows["QR_expanding_q0.99"].values,
    }


def get_calendar_context(forecast_date):
    be_holidays = holidays.Belgium(years=[forecast_date.year - 1, forecast_date.year, forecast_date.year + 1])
    is_holiday = forecast_date in be_holidays

    is_bridge_day = False
    if forecast_date.weekday() == 0:  # Monday
        tuesday = forecast_date + timedelta(days=1)
        is_bridge_day = tuesday in be_holidays
    elif forecast_date.weekday() == 4:  # Friday
        thursday = forecast_date - timedelta(days=1)
        is_bridge_day = thursday in be_holidays

    return {
        "is_holiday": is_holiday,
        "holiday_name": be_holidays.get(forecast_date, ""),
        "day_of_week": forecast_date.strftime("%a"),
        "is_bridge_day": is_bridge_day,
    }


def make_chart(timestamps, forecast, inner_lower=None, inner_upper=None,
                outer_lower=None, outer_upper=None, solar=None, wind=None, renewables_range=None, flagged=None):
    figure = go.Figure()

    if outer_lower is not None and outer_upper is not None:
        figure.add_trace(go.Scatter(x=timestamps, y=outer_upper, mode="lines", line=dict(width=0), showlegend=False))
        figure.add_trace(go.Scatter(
            x=timestamps, y=outer_lower, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(100,100,255,0.10)", name="Extreme range (1st-99th pct)",
        ))

    if inner_lower is not None and inner_upper is not None:
        figure.add_trace(go.Scatter(x=timestamps, y=inner_upper, mode="lines", line=dict(width=0), showlegend=False))
        figure.add_trace(go.Scatter(
            x=timestamps, y=inner_lower, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(100,100,255,0.25)", name="Typical range (10th-90th pct)",
        ))

    figure.add_trace(go.Scatter(x=timestamps, y=forecast, mode="lines+markers", name="DNN Forecast",
                                 marker=dict(size=4)))

    if flagged is not None and np.any(flagged):
        flagged = np.asarray(flagged)
        ts_arr = np.asarray(timestamps)
        f_arr = np.asarray(forecast)
        figure.add_trace(go.Scatter(
            x=ts_arr[flagged], y=f_arr[flagged], mode="markers", name="Flagged (5th/95th pct)",
            marker=dict(size=4, symbol="diamond", color="orange", line=dict(color="white", width=1)),
        ))

    if solar is not None:
        figure.add_trace(go.Scatter(x=timestamps, y=solar, mode="lines", name="Solar (MW)", yaxis="y2",
                                     line=dict(color="orange")))
    if wind is not None:
        figure.add_trace(go.Scatter(x=timestamps, y=wind, mode="lines", name="Wind (MW)", yaxis="y2",
                                     line=dict(color="green")))

    ts_min = pd.Timestamp(np.asarray(timestamps).min())
    ts_max = pd.Timestamp(np.asarray(timestamps).max())

    layout_kwargs = dict(
        xaxis_title="Time of day",
        yaxis_title="EUR / MWh",
        xaxis=dict(tickformat="%H:%M", dtick=3600000, range=[ts_min, ts_max]),  # one labeled tick per hour
    )
    if solar is not None or wind is not None:
        y2_settings = dict(title="MW", overlaying="y", side="right")
        if renewables_range is not None:
            y2_settings["range"] = renewables_range
        layout_kwargs["yaxis2"] = y2_settings

    figure.update_layout(**layout_kwargs)
    return figure


def make_simple_chart(timestamps, values, y_title):

    ts_min = pd.Timestamp(np.asarray(timestamps).min())
    ts_max = pd.Timestamp(np.asarray(timestamps).max())

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=timestamps, y=values, mode="lines+markers", name=y_title, marker=dict(size=4)))
    figure.update_layout(
        xaxis_title="Time of day",
        yaxis_title=y_title,
        xaxis=dict(tickformat="%H:%M", dtick=3600000, range=[ts_min, ts_max]),
    )
    return figure


def make_renewables_chart(timestamps, solar=None, wind=None):
    """Standalone solar + wind chart -- both in MW, so they share one axis."""
    figure = go.Figure()
    if solar is not None:
        figure.add_trace(go.Scatter(x=timestamps, y=solar, mode="lines+markers", name="Solar (MW)",
                                     marker=dict(size=4), line=dict(color="orange")))
    if wind is not None:
        figure.add_trace(go.Scatter(x=timestamps, y=wind, mode="lines+markers", name="Wind total (MW)",
                                     marker=dict(size=4), line=dict(color="green")))
    ts_min = pd.Timestamp(np.asarray(timestamps).min())
    ts_max = pd.Timestamp(np.asarray(timestamps).max())
    figure.update_layout(
        xaxis_title="Time of day",
        yaxis_title="MW",
        xaxis=dict(tickformat="%H:%M", dtick=3600000, range=[ts_min, ts_max]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60),
    )
    return figure


# --------------------------------------------------------------------------
# AUTH HELPERS
# --------------------------------------------------------------------------

def hash_password(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def check_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))


def validate_registration(username, email, password):
    if not username or not email or not password:
        return False, "All fields (Username, Email, and Password) are required."
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return False, "Please enter a valid email address."
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"[0-9]", password):
        return False, "Password must contain both letters and numbers."
    return True, ""


# --------------------------------------------------------------------------
# PAGE 1: REVIEW & ADJUST
# --------------------------------------------------------------------------

def page_review_and_adjust():
    current_user = st.session_state["logged_in_user"]
    current_role = st.session_state["role"]

    dnn_df = get_dnn_df()
    qr_df = get_qr_df()
    be_df = get_be_df()

    with st.sidebar:
        if current_role == "admin":
            users = load_users()
            expert_list = [u for u, d in users.items() if d["role"] == "expert"]
            expert_id = st.selectbox("Expert ID (Admin View)", expert_list) if expert_list else None
        else:
            expert_id = current_user
            st.write(f"**Expert ID:** {expert_id}")

        available_dates = get_available_dates(dnn_df)
        if not available_dates:
            st.error("No complete DNN forecast days available yet.")
            return
        forecast_date = st.date_input(
            "Forecast date (day d+1)",
            value=available_dates[-1],
            min_value=available_dates[0],
            max_value=available_dates[-1],
        )

    forecast, timestamps = dnn_forecast(forecast_date, dnn_df)
    if forecast is None:
        st.error(f"No complete DNN forecast for {forecast_date}.")
        return

    imputed_flags = dnn_imputed_flags(forecast_date, dnn_df)
    if imputed_flags is not None and imputed_flags.any():
        st.warning(
            f"⚠️ This forecast run failed for {forecast_date} — the previous day's forecast was "
            "carried forward. Treat this forecast with extra caution."
        )

    bands = qr_uncertainty_bands(forecast_date, qr_df)

    calendar_ctx = get_calendar_context(forecast_date)
    day_rows = be_df.loc[be_df["date_only"] == forecast_date].sort_values("Date")

    if len(day_rows) == STEPS_PER_DAY:
        net_demand = day_rows["Load_BE"] - day_rows["Solar_BE"] - day_rows["Wind_Offshore_BE"] - day_rows["Wind_Onshore_BE"]
        avg_temp = day_rows["temperature_2m"].mean()
        avg_hum = day_rows["relative_humidity_2m"].mean()
        avg_load_fr = day_rows["Load_FR"].mean()
        context_available = True
    else:
        context_available = False

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Day of week", calendar_ctx["day_of_week"])
    c2.metric("Holiday?", calendar_ctx["holiday_name"] if calendar_ctx["is_holiday"] else "No")
    c3.metric("Bridge day?", "Yes" if calendar_ctx["is_bridge_day"] else "No")
    if context_available:
        c4.metric("Avg. net demand BE (MW)", f"{net_demand.mean():,.0f}")

    if context_available:
        c5, c6, c7 = st.columns(3)
        c5.metric("Avg. temp (°C)", f"{avg_temp:.1f}")
        c6.metric("Avg. humidity (%)", f"{avg_hum:.0f}")
        c7.metric("Avg. French Demand (MW)", f"{avg_load_fr:,.0f}")
    else:
        st.info("Weather/load context not available for this date.")

     # --- Auto-flag volatile slots (5th/95th percentile of this day's own forecast) ---
    low_threshold = np.percentile(forecast, 5)
    high_threshold = np.percentile(forecast, 95)
    flagged = (forecast <= low_threshold) | (forecast >= high_threshold)

    fig = make_chart(
        timestamps, forecast,
        inner_lower=bands["inner_lower"] if bands else None,
        inner_upper=bands["inner_upper"] if bands else None,
        outer_lower=bands["outer_lower"] if bands else None,
        outer_upper=bands["outer_upper"] if bands else None,
        flagged=flagged
    )
    st.plotly_chart(fig, width="stretch")

    if not bands:
        st.caption("QR uncertainty bands unavailable for this date.")

    if context_available:
        weather_col, renewables_col = st.columns(2)

        with weather_col:
            with st.container(border=True):
                st.subheader("Weather")
                weather_choice = st.selectbox("Show hourly:", ["Temperature", "Humidity"])
                if weather_choice == "Temperature":
                    st.plotly_chart(make_simple_chart(day_rows["Date"].values, day_rows["temperature_2m"].values,
                                                       "Temperature (°C)"), width="stretch")
                elif weather_choice == "Humidity":
                    st.plotly_chart(make_simple_chart(day_rows["Date"].values, day_rows["relative_humidity_2m"].values,
                                                       "Humidity (%)"), width="stretch")

        with renewables_col:
                    with st.container(border=True):
                        st.subheader("Solar & Wind")
                        sc1, sc2 = st.columns(2)
                        show_solar = sc1.checkbox("Solar", value=True)
                        show_wind = sc2.checkbox("Wind", value=True)

                        wind_total = day_rows["Wind_Offshore_BE"] + day_rows["Wind_Onshore_BE"]
                        solar_vals = day_rows["Solar_BE"].values if show_solar else None
                        wind_vals = wind_total.values if show_wind else None

                        if solar_vals is None and wind_vals is None:
                            st.info("Select at least one series to display.")
                        else:
                            st.plotly_chart(make_renewables_chart(day_rows["Date"].values, solar=solar_vals, wind=wind_vals),
                                            width="stretch")

    hour_of_slot = np.array([pd.Timestamp(ts).hour for ts in timestamps])
    time_label = [pd.Timestamp(ts).strftime("%H:%M") for ts in timestamps]

    working_df = pd.DataFrame({
        "timestamp_slot": timestamps,
        "hour": hour_of_slot,
        "time_label": time_label,
        "forecast": forecast,
        "adjusted": forecast.copy(),
        "flagged": flagged,
        "load_fr": day_rows["Load_FR"].values if context_available else np.nan,
    })

    key = f"{expert_id}_{forecast_date}"
    already_submitted = has_submitted(expert_id, forecast_date) if expert_id else False
    is_read_only = (current_role == "admin")

    if key not in st.session_state:
        log = load_feedback()
        if not log.empty:
            past_sub = log[(log["expert_id"] == expert_id) & (log["forecast_date"] == forecast_date)]
            if not past_sub.empty:
                past_sub = past_sub.tail(STEPS_PER_DAY).sort_values("timestamp_slot")
                if len(past_sub) == STEPS_PER_DAY:
                    working_df["adjusted"] = past_sub["adjusted"].values
                    working_df["flagged"] = past_sub["flagged"].values
        st.session_state[key] = working_df

    working = st.session_state[key]

    if is_read_only or already_submitted:
        # Read-only view: nothing here can trigger a submission, so there's no
        # rerun-per-edit cost to avoid -- a plain loop (no form) is fine.
        for h in range(24):
            hour_slice = working[working["hour"] == h]
            n_flagged = int(hour_slice["flagged"].sum())
            label = f"{h:02d}:00"
            if n_flagged > 0:
                label += f"  ⚠️ {n_flagged} flagged"
            with st.expander(label, expanded=False):
                st.data_editor(
                    hour_slice[["time_label", "forecast", "adjusted", "flagged"]],
                    column_config={
                        "time_label": st.column_config.TextColumn("Time", disabled=True),
                        "forecast": st.column_config.NumberColumn("DNN forecast", disabled=True, format="%.2f"),
                        "adjusted": st.column_config.NumberColumn("Adjusted", disabled=True, format="%.2f"),
                        "flagged": st.column_config.CheckboxColumn("Flag", disabled=True),
                    },
                    disabled=True,
                    hide_index=True,
                    key=f"editor_{key}_{h}",
                )

        if is_read_only:
            st.info(f"Viewing {expert_id}'s submission (read-only — admins cannot submit on behalf of experts).")
        else:
            st.info(f"You've already submitted feedback for {forecast_date}. Submissions are final.")

    else:
        # Editable view: everything lives inside one form, so editing any cell in
        # any of the 24 hour-tables does NOT trigger a rerun -- only clicking
        # "Submit feedback" does. This is what fixes both the sluggishness (no
        # more full-page rerun on every single cell edit) and the edits reverting
        # after a rerun (nothing overwrites session_state mid-edit anymore).
        with st.form(key=f"feedback_form_{key}"):
            edited_pieces = []
            for h in range(24):
                hour_slice = working[working["hour"] == h]
                n_flagged = int(hour_slice["flagged"].sum())
                label = f"{h:02d}:00"
                if n_flagged > 0:
                    label += f"  ⚠️ {n_flagged} flagged"
                with st.expander(label, expanded=(n_flagged > 0)):
                    edited_hour = st.data_editor(
                        hour_slice[["time_label", "forecast", "adjusted", "flagged"]],
                        column_config={
                            "time_label": st.column_config.TextColumn("Time", disabled=True),
                            "forecast": st.column_config.NumberColumn("DNN forecast", disabled=True, format="%.2f"),
                            "adjusted": st.column_config.NumberColumn("Adjusted", step=1.0, format="%.2f"),
                            "flagged": st.column_config.CheckboxColumn("Flag"),
                        },
                        hide_index=True,
                        key=f"editor_{key}_{h}",
                    )
                    edited_pieces.append(edited_hour)

            confidence = st.slider("How confident are you in these adjustments?", 1, 5, 3)
            submitted = st.form_submit_button("Submit feedback")

        if submitted:
            if not expert_id:
                st.error("Error: No Expert ID found.")
            elif has_submitted(expert_id, forecast_date):
                st.error("A submission already exists for this date. Refresh the page.")
            else:
                edited = pd.concat(edited_pieces, ignore_index=True)
                edited["timestamp_slot"] = working["timestamp_slot"].values
                edited["load_fr"] = working["load_fr"].values

                rows = edited.copy()
                rows["expert_id"] = expert_id
                rows["forecast_date"] = forecast_date
                rows["timestamp"] = dt.datetime.now(dt.timezone.utc).isoformat()
                rows["confidence"] = confidence
                save_feedback(rows)
                st.success(f"Saved {len(rows)} rows for {expert_id} on {forecast_date}.")
                st.rerun()


# --------------------------------------------------------------------------
# PAGE 2: REVEAL & EVALUATE
# --------------------------------------------------------------------------

def get_last_evaluable_ts(now=None):
    """Never evaluate against a delivery day whose day-ahead auction hasn't settled yet."""
    now = now if now is not None else pd.Timestamp.now(tz="Europe/Brussels").tz_localize(None)
    tomorrow_start = now.normalize() + pd.Timedelta(days=1)
    return tomorrow_start - pd.Timedelta(minutes=15)


def page_reveal_and_evaluate():
    st.title("Reveal & Evaluate")

    log = load_feedback()
    if log.empty:
        st.warning("No submissions yet.")
        return

    be_df = get_be_df()

    expert_id = st.selectbox("Expert ID", sorted(log["expert_id"].dropna().unique()))
    available_dates = sorted(log.loc[log["expert_id"] == expert_id, "forecast_date"].dropna().unique())
    forecast_date = st.selectbox("Forecast date", available_dates)

    last_evaluable = get_last_evaluable_ts()
    if pd.Timestamp(forecast_date) > last_evaluable:
        st.info("This delivery day's day-ahead prices haven't settled yet — nothing to reveal.")
        return

    submission = (
        log.loc[
            (log["expert_id"] == expert_id) & (log["forecast_date"] == forecast_date),
            ["timestamp_slot", "forecast", "adjusted", "confidence"],
        ]
        .sort_values("timestamp_slot")
    )

    actuals = be_df.loc[be_df["date_only"] == forecast_date, ["Date", "Price"]].rename(
        columns={"Date": "timestamp_slot", "Price": "actual"}
    )
    evaluation = submission.merge(actuals, on="timestamp_slot", how="inner")

    if evaluation.empty:
        st.info("No realized prices available yet for this date.")
        return

    forecast_mae = (evaluation["forecast"] - evaluation["actual"]).abs().mean()
    adjusted_mae = (evaluation["adjusted"] - evaluation["actual"]).abs().mean()
    confidence_rating = submission["confidence"].iloc[0]

    forecast_metric, adjusted_metric, confidence_metric = st.columns(3)
    forecast_metric.metric("Forecast MAE", f"{forecast_mae:.2f} EUR/MWh")
    adjusted_metric.metric("Adjusted MAE", f"{adjusted_mae:.2f} EUR/MWh")
    confidence_metric.metric("Expert confidence", f"{confidence_rating}/5")

    if adjusted_mae < forecast_mae:
        st.write("Verdict: the expert adjustment improved the forecast.")
    elif adjusted_mae > forecast_mae:
        st.write("Verdict: the expert adjustment worsened the forecast.")
    else:
        st.write("Verdict: the expert adjustment made no difference.")


# --------------------------------------------------------------------------
# PAGE 3: EXPERT SCOREBOARD
# --------------------------------------------------------------------------

def page_expert_scoreboard():
    st.title("Expert Scoreboard")

    log = load_feedback()
    if log.empty:
        st.warning("No submissions yet.")
        return

    be_df = get_be_df()
    last_evaluable = get_last_evaluable_ts()

    results = []
    for (expert_id, forecast_date), group in log.groupby(["expert_id", "forecast_date"]):
        if pd.Timestamp(forecast_date) > last_evaluable:
            continue

        actuals = be_df.loc[be_df["date_only"] == forecast_date, ["Date", "Price"]].rename(
            columns={"Date": "timestamp_slot", "Price": "actual"}
        )
        evaluation = group.merge(actuals, on="timestamp_slot", how="inner")
        if evaluation.empty:
            continue

        forecast_mae = (evaluation["forecast"] - evaluation["actual"]).abs().mean()
        adjusted_mae = (evaluation["adjusted"] - evaluation["actual"]).abs().mean()

        results.append({
            "expert_id": expert_id,
            "forecast_date": forecast_date,
            "forecast_mae": forecast_mae,
            "adjusted_mae": adjusted_mae,
            "improvement": forecast_mae - adjusted_mae,
            "confidence": group["confidence"].iloc[0],
        })

    if not results:
        st.warning("No submissions overlap with settled actual prices yet.")
        return

    results_df = pd.DataFrame(results)

    scoreboard = (
        results_df.groupby("expert_id")
        .agg(
            avg_improvement=("improvement", "mean"),
            days_reviewed=("forecast_date", "nunique"),
            win_rate=("improvement", lambda s: (s > 0).mean()),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
        .sort_values("avg_improvement", ascending=False)
    )
    scoreboard["win_rate"] = (scoreboard["win_rate"] * 100).round(1)
    scoreboard["avg_improvement"] = scoreboard["avg_improvement"].round(2)
    scoreboard["avg_confidence"] = scoreboard["avg_confidence"].round(1)

    st.dataframe(scoreboard, hide_index=True)


# --------------------------------------------------------------------------
# AUTH SCREEN
# --------------------------------------------------------------------------

def auth_screen():
    st.subheader("Welcome to EPF Expert Review")

    auth_mode = st.radio("Choose an option:", ["Log In", "Create Account"], horizontal=True)
    st.divider()

    if auth_mode == "Log In":
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")

        if st.button("Log In"):
            users = load_users()
            if login_user in users and check_password(login_pass, users[login_user]["password"]):
                st.session_state["logged_in_user"] = login_user
                st.session_state["role"] = users[login_user]["role"]
                st.rerun()
            else:
                st.error("Invalid username or password.")

    elif auth_mode == "Create Account":
        new_email = st.text_input("Email Address", key="new_email")
        new_user = st.text_input("New Username", key="new_user")
        new_pass = st.text_input("New Password", type="password", key="new_pass")
        new_role = st.selectbox("Role", EXPERT_ROLES, key="new_role")

        if st.button("Create Account"):
            users = load_users()
            username_input = new_user.strip()
            email_input = new_email.strip().lower()
            email_exists = any(account.get("email") == email_input for account in users.values())
            is_valid, validation_message = validate_registration(username_input, email_input, new_pass)

            if not is_valid:
                st.error(validation_message)
            elif username_input in users:
                st.error("This username is already taken. Please choose another.")
            elif email_exists:
                st.error("This email address is already registered. Please use another or log in.")
            else:
                save_new_user(username_input, hash_password(new_pass), email_input, new_role)
                st.success("Account created successfully! You can now switch to the Log In option.")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    if "logged_in_user" not in st.session_state:
        auth_screen()
        return

    current_user = st.session_state["logged_in_user"]
    current_role = st.session_state["role"]

    with st.sidebar:
        st.write(f"Logged in as: **{current_user}** ({current_role})")
        if st.button("Log Out"):
            st.session_state.clear()
            st.rerun()

        st.divider()

        if current_role == "admin":
            pages = ["Review & Adjust", "Reveal & Evaluate", "Expert Scoreboard"]
        else:
            pages = ["Review & Adjust"]

        page = st.radio("Page", pages)

    if page == "Review & Adjust":
        page_review_and_adjust()
    elif page == "Reveal & Evaluate":
        page_reveal_and_evaluate()
    elif page == "Expert Scoreboard":
        page_expert_scoreboard()


if __name__ == "__main__":
    main()