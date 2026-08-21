"""
EPF Expert Review -- a Streamlit dashboard for human-in-the-loop review of
day-ahead electricity price forecasts for the Belgian market.

WHAT THIS APP IS FOR
---------------------
A neural network (DNN) produces a day-ahead price forecast for Belgium at
15-minute resolution (96 slots/day). Domain experts review that forecast,
optionally adjust individual slots they disagree with, flag anomalies, and
rate their own confidence. Once the delivery day has actually happened and
its day-ahead auction has settled, admins can reveal the realized price and
see whether each expert's adjustment improved or worsened accuracy (MAE)
relative to the raw model forecast -- and aggregate that across experts on a
scoreboard. The goal is to measure the value of human correction on top of
the model, not just to collect opinions.

ROLES
-----
- "expert": can only see and submit on the Review & Adjust page, and only
  for their own submissions. Submissions are final -- no editing after
  submit.
- "admin": can view (but never submit on behalf of) any expert's Review &
  Adjust page, plus the Reveal & Evaluate and Expert Scoreboard pages.

DATA SOURCES
------------
Three CSVs are pulled live from a GitHub repo on every page load (cached
for 30 min to absorb the daily 10AM/14h refreshes without hammering
GitHub): the DNN forecast, the QR (quantile regression) uncertainty bands,
and the realized Belgian market data (price, load, solar, wind, weather).
See the GITHUB LOADING section below for the exact files/columns.

PAGES
-----
1. Review & Adjust       -- the core workflow described above.
2. Deterministic Forecast Analysis -- a separate, unrelated dashboard built
   by a collaborator (LEAR/XGB/DNN/Ensemble model comparison), embedded via
   iframe from its Hugging Face Space. It has its own UI and does not share
   a session, login, or data pipeline with this app.
3. Reveal & Evaluate (admin only) -- compares one expert's one-day
   submission against the realized price once it has settled.
4. Expert Scoreboard (admin only) -- aggregates every evaluated submission
   across all experts into a leaderboard (avg. improvement, win rate, etc.).

THEMING
-------
A dark/light toggle in the sidebar injects CSS (see apply_theme()) and
drives a matching Plotly template (see themed()) so charts and UI chrome
never fall out of sync with each other.
"""

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
st.title('Electricity Price Forecasting -  Expert Review')


# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

# GitHub repo that publishes the forecast/actuals CSVs this app consumes.
# Read access requires GITHUB_TOKEN (set in .env / environment); the repo is
# private, hence the auth header in fetch_csv_from_github() below.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = "margaridamascarenhas"
GITHUB_REPO = "DAM_Forecast_V4"
GITHUB_BRANCH = "main"

DNN_FILE = "DNN_forecasts_10AM.csv"     # DNN point forecast + Imputed flag
QR_FILE = "QR_forecasts_10AM.csv"       # Quantile regression uncertainty bands
BE_DATA_FILE = "Data_BE_UTC.csv"        # Realized price, load, weather, renewables

STEPS_PER_DAY = 96  # 15-minute resolution: 24h * 4

EXPERT_ROLES = ["expert"]  # roles selectable at self-registration (no public admin signup)



# --------------------------------------------------------------------------
# THEME (dark / light mode toggle)
#
# get_palette() is the single source of truth for both layers below:
#   - apply_theme() uses it to inject CSS that themes Streamlit's own chrome
#     (sidebar, buttons, inputs, popovers, calendar, icons, etc.)
#   - themed() uses it to give every Plotly chart matching colors, so charts
#     never look out of sync with the rest of the page.
# --------------------------------------------------------------------------


def get_palette(dark: bool) -> dict:
    """Single source of truth for theme colors, shared by the injected CSS
    and the Plotly charts, so both layers always agree on what's readable."""
    if dark:
        return {
            "bg": "#0e1117",
            "bg_secondary": "#161b22",
            "sidebar_bg": "#161b22",
            "card_bg": "#1c222b",
            "text": "#e6edf3",
            "text_muted": "#9aa5b1",
            "border": "#2d3540",
            "grid": "#2d3540",
            "input_bg": "#1c222b",
            "accent": "#6366f1",
            "accent_text": "#ffffff",
        }
    return {
        "bg": "#ffffff",
        "bg_secondary": "#f6f7f9",
        "sidebar_bg": "#f6f7f9",
        "card_bg": "#ffffff",
        "text": "#1f2328",
        "text_muted": "#57606a",
        "border": "#d7dbe0",
        "grid": "#e3e6ea",
        "input_bg": "#ffffff",
        "accent": "#4f46e5",
        "accent_text": "#ffffff",
    }


def apply_theme():
    """Renders the dark/light toggle in the sidebar and injects matching CSS.
    Dark mode is the default. Call this once, first thing, in main()."""
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = True  # dark mode by default

    st.sidebar.toggle("🌙 Dark mode", key="dark_mode")
    dark = st.session_state["dark_mode"]
    palette = get_palette(dark)

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background-color: {palette['bg']};
            color: {palette['text']};
        }}
        [data-testid="stSidebar"] {{
            background-color: {palette['sidebar_bg']};
            border-right: 1px solid {palette['border']};
        }}
        [data-testid="stSidebar"] * {{
            color: {palette['text']} !important;
        }}
        h1, h2, h3, h4, h5, h6, p, label, span, li, .stMarkdown {{
            color: {palette['text']};
        }}
        [data-testid="stMetric"] {{
            background-color: {palette['card_bg']};
            border: 1px solid {palette['border']};
            border-radius: 10px;
            padding: 0.75rem 1rem;
        }}
        [data-testid="stMetricLabel"] {{
            color: {palette['text_muted']} !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {palette['text']} !important;
        }}
        [data-testid="stExpander"], [data-testid="stForm"] {{
            background-color: {palette['card_bg']};
            border: 1px solid {palette['border']};
            border-radius: 10px;
        }}
        div[data-baseweb="input"], div[data-baseweb="select"], div[data-baseweb="textarea"] {{
            background-color: {palette['input_bg']} !important;
            border-color: {palette['border']} !important;
        }}
        input, textarea {{
            background-color: {palette['input_bg']} !important;
            color: {palette['text']} !important;
        }}
        .stButton > button, .stFormSubmitButton > button {{
            background-color: {palette['accent']};
            color: {palette['accent_text']};
            border: none;
            border-radius: 8px;
        }}
        .stButton > button:hover, .stFormSubmitButton > button:hover {{
            filter: brightness(1.1);
        }}
        hr {{
            border-color: {palette['border']};
        }}
        [data-testid="stDataFrame"], [data-testid="stDataEditor"] {{
            border: 1px solid {palette['border']};
            border-radius: 8px;
        }}
        [data-testid="stAlert"] {{
            background-color: {palette['card_bg']};
            color: {palette['text']} !important;
            border: 1px solid {palette['border']};
        }}
        [data-testid="stAlert"] * {{
            color: {palette['text']} !important;
        }}
        /* Selectbox/radio dropdown menus render in a portal attached to
           <body>, outside the sidebar/app containers above, so they need
           their own rule or their text is invisible against the popup's
           own background in light mode. */
        div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {{
            background-color: {palette['card_bg']} !important;
        }}
        div[data-baseweb="popover"] *, div[data-baseweb="menu"] *, ul[role="listbox"] * {{
            color: {palette['text']} !important;
        }}
        li[role="option"]:hover, li[aria-selected="true"] {{
            background-color: {palette['bg_secondary']} !important;
        }}
        /* The date picker's month/year header lives in a separate baseweb
        wrapper from the day grid itself — cover both, or the header text
        stays stuck on its default color regardless of mode. */
        div[data-baseweb="datepicker"], div[data-baseweb="calendar"] {{
            background-color: {palette['card_bg']} !important;
        }}
        div[data-baseweb="datepicker"] *, div[data-baseweb="calendar"] * {{
            color: {palette['text']} !important;
        }}
        div[data-baseweb="calendar"] [role="gridcell"] > div {{
            background-color: transparent !important;
        }}
        /* Icon glyphs (password show/hide, calendar nav arrows, expander
        chevrons, sidebar icons) are SVGs with their own fixed color that
        doesn't follow the page text color automatically — scoped to
        Streamlit's own UI chrome only, never the Plotly charts, which
        manage their own colors via the template. */
        button svg, [role="button"] svg,
        div[data-baseweb="input"] svg, div[data-baseweb="select"] svg,
        div[data-baseweb="popover"] svg, div[data-baseweb="calendar"] svg,
        div[data-baseweb="datepicker"] svg,
        [data-testid="stExpander"] svg, [data-testid="stSidebar"] svg {{
            fill: {palette['text']} !important;
            stroke: {palette['text']} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return dark


def current_plotly_template():
    """Plotly template name matching the active dark/light toggle state."""
    return "plotly_dark" if st.session_state.get("dark_mode", True) else "plotly_white"


def themed(figure):
    """Applies the current dark/light template and makes the chart background
    transparent so it blends with the page. Text, legend, and gridline colors
    are set explicitly (not just left to the template default) so they can
    never end up washed out or invisible in either mode."""
    dark = st.session_state.get("dark_mode", True)
    palette = get_palette(dark)

    figure.update_layout(
        template=current_plotly_template(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=palette["text"]),
        legend=dict(
            font=dict(color=palette["text"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor=palette["grid"],
            zerolinecolor=palette["grid"],
            linecolor=palette["border"],
            tickfont=dict(color=palette["text_muted"]),
            title_font=dict(color=palette["text"]),
        ),
        yaxis=dict(
            gridcolor=palette["grid"],
            zerolinecolor=palette["grid"],
            linecolor=palette["border"],
            tickfont=dict(color=palette["text_muted"]),
            title_font=dict(color=palette["text"]),
        ),
        yaxis2=dict(
            tickfont=dict(color=palette["text_muted"]),
            title_font=dict(color=palette["text"]),
        ),
        hoverlabel=dict(
            font=dict(color=palette["text"]),
            bgcolor=palette["card_bg"],
            bordercolor=palette["border"],
        ),
    )
    return figure


# --------------------------------------------------------------------------
# GITHUB DATA LOADING (live, cached with a TTL so daily 10AM/14h updates get picked up)
# --------------------------------------------------------------------------

def fetch_csv_from_github(owner, repo, branch, path, fname, token, usecols=None):
    """Downloads a single CSV file's raw content from a GitHub repo and
    parses it into a DataFrame. `token` is required for private repos (like
    this one) -- passed as a Bearer token; requests with no token still work
    against public repos but will 404/403 here since GITHUB_REPO is private."""
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}/{fname}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.get(raw_url, headers=headers, timeout=30)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), usecols=usecols)


@st.cache_data(ttl=1800)  # 30 min -- catches both the 10AM and 14h daily refreshes without hammering GitHub
def get_dnn_df():
    """DNN point forecast: one row per 15-min slot, with DateTime,
    DNN_expanding (the forecast value), and Imputed (True = that day's model
    run failed and the previous day's forecast was carried forward)."""
    # Only 3 columns exist in this file already -- nothing to trim.
    df = fetch_csv_from_github(GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, "Forecast", DNN_FILE, GITHUB_TOKEN)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["date_only"] = df["DateTime"].dt.date
    return df.sort_values("DateTime").reset_index(drop=True)


@st.cache_data(ttl=1800)
def get_qr_df():
    """Quantile regression uncertainty bands for the DNN forecast: the
    10th/90th percentile columns become the chart's "typical range", and the
    1st/99th percentile columns become the wider "extreme range"."""
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
    """Realized Belgian market data: settled day-ahead price plus load,
    solar, wind, and weather -- everything needed for the context metrics,
    Reveal & Evaluate's "actual" price, and the Solar/Wind and Weather
    sub-charts on Review & Adjust."""
    # Already a lean 9-column file -- nothing to trim.
    df = fetch_csv_from_github(GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH, "datasets", BE_DATA_FILE, GITHUB_TOKEN)
    df["Date"] = pd.to_datetime(df["Date"])
    df["date_only"] = df["Date"].dt.date
    return df.sort_values("Date").reset_index(drop=True)


init_db()  # creates users/feedback tables in SQLite if they don't exist yet


# --------------------------------------------------------------------------
# SHARED HELPER FUNCTIONS
# --------------------------------------------------------------------------

def get_available_dates(dnn_df):
    """Dates with a complete 96-slot DNN forecast. A date with fewer rows
    means the model run for that day is only partially present (e.g. still
    in progress or truncated) -- excluded so the UI never shows a half-day."""
    counts = dnn_df.groupby("date_only").size()
    return sorted(counts[counts == STEPS_PER_DAY].index)


def dnn_forecast(forecast_date, dnn_df):
    """Returns (values_array, timestamps) for a 96-slot day, or (None, None)
    if that date doesn't have a complete forecast."""
    day_rows = dnn_df[dnn_df["date_only"] == forecast_date].sort_values("DateTime")
    if len(day_rows) != STEPS_PER_DAY:
        return None, None
    return day_rows["DNN_expanding"].values, day_rows["DateTime"].values


def dnn_imputed_flags(forecast_date, dnn_df):
    """Returns the per-slot Imputed flag array for the day, or None. All 96
    values are identical in practice (the flag is set at the day level, just
    stored per-slot) -- .any() is enough to know if the whole day was imputed."""
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
    """Belgian-holiday and bridge-day context for the day being reviewed --
    surfaced as metrics on Review & Adjust since both are known drivers of
    unusual demand/price shapes an expert should factor into their review.
    A "bridge day" is a working day sandwiched between a holiday and a
    weekend (Monday after a Tuesday holiday, or Friday before a Monday
    holiday) -- often behaves like a de facto holiday for demand purposes."""
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
    """The main Review & Adjust chart: DNN forecast line, the two nested QR
    uncertainty bands (drawn outer-then-inner so the darker inner band sits
    on top), auto-flagged anomaly markers, and optional solar/wind traces on
    a secondary y-axis (MW). All inputs are optional except timestamps and
    forecast -- bands/renewables are simply omitted from the figure if not
    supplied, rather than erroring."""
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

    # Anomaly markers: slots outside this day's own 5th/95th percentile
    # (computed by the caller, see the flagging comment in page_review_and_adjust).
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
    return themed(figure)


def make_simple_chart(timestamps, values, y_title):
    """A single-series hourly chart used for the Weather sub-chart
    (Temperature or Humidity, one at a time, selected via a dropdown)."""

    ts_min = pd.Timestamp(np.asarray(timestamps).min())
    ts_max = pd.Timestamp(np.asarray(timestamps).max())

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=timestamps, y=values, mode="lines+markers", name=y_title, marker=dict(size=4)))
    figure.update_layout(
        xaxis_title="Time of day",
        yaxis_title=y_title,
        xaxis=dict(tickformat="%H:%M", dtick=3600000, range=[ts_min, ts_max]),
    )
    return themed(figure)


def make_renewables_chart(timestamps, solar=None, wind=None):
    """Standalone solar + wind chart -- both in MW, so they share one axis
    (unlike the main chart's secondary axis, this one doesn't need to share
    space with a EUR/MWh price series)."""
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
    return themed(figure)


# --------------------------------------------------------------------------
# AUTH HELPERS
# --------------------------------------------------------------------------

def hash_password(password):
    """One-way bcrypt hash for storing a new user's password. bcrypt embeds
    its own random salt in the output, so no separate salt column is needed
    in the users table -- check_password() below re-derives it from the hash."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def check_password(password, hashed_password):
    """Verifies a login attempt against the stored bcrypt hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))


def validate_registration(username, email, password):
    """Self-registration field checks. Returns (is_valid, error_message) --
    error_message is empty when is_valid is True. Duplicate username/email
    checks happen separately in auth_screen() since they need a DB lookup."""
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
    """The core page: load today's (or a selected) forecast, show context and
    charts, and let an expert review/adjust it 15-minute-slot by slot.

    Behavior differs by who's looking and whether feedback was submitted:
      - expert, not yet submitted -> editable (see the st.form block below)
      - expert, already submitted -> read-only, "already submitted" notice
      - admin (any state)         -> always read-only; admins can view any
        expert's work but can never submit on their behalf
    """
    current_user = st.session_state["logged_in_user"]
    current_role = st.session_state["role"]

    dnn_df = get_dnn_df()
    qr_df = get_qr_df()
    be_df = get_be_df()

    with st.sidebar:
        if current_role == "admin":
            # Admins pick which expert's work to view; experts only ever see their own.
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

    # Warn if this day's forecast is a stale carry-forward from a failed model run.
    imputed_flags = dnn_imputed_flags(forecast_date, dnn_df)
    if imputed_flags is not None and imputed_flags.any():
        st.warning(
            f"⚠️ This forecast run failed for {forecast_date} — the previous day's forecast was "
            "carried forward. Treat this forecast with extra caution."
        )

    bands = qr_uncertainty_bands(forecast_date, qr_df)

    calendar_ctx = get_calendar_context(forecast_date)
    day_rows = be_df.loc[be_df["date_only"] == forecast_date].sort_values("Date")

    # Weather/load context is only available once the actuals feed has caught
    # up to this date -- e.g. tomorrow's forecast reviewed today won't have it yet.
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
    # Relative to THIS day's own distribution, not a fixed EUR/MWh threshold --
    # so a generally-volatile day and a generally-calm day each get flagged
    # relative to their own baseline, rather than one fixed cutoff favoring
    # whichever kind of day happens to be more extreme in absolute terms.
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

    # Solar & Wind and Weather sit one under the other (not side-by-side):
    # solar+wind are genuinely linked (their combined dip drives net demand
    # and price spikes, see net_demand above), so they share one chart by
    # default; Weather's two series are independent context, so they're a
    # single dropdown ("Hide" by default) instead of a second always-on chart.
    if context_available:
        with st.container(border=True):
            st.subheader("Solar & Wind - Renewables")
            sc1, sc2 = st.columns(2)
            show_solar = sc1.toggle("Solar", value=True)
            show_wind = sc2.toggle("Wind", value=True)

            wind_total = day_rows["Wind_Offshore_BE"] + day_rows["Wind_Onshore_BE"]
            solar_vals = day_rows["Solar_BE"].values if show_solar else None
            wind_vals = wind_total.values if show_wind else None

            if solar_vals is None and wind_vals is None:
                st.info("Select at least one series to display.")
            else:
                st.plotly_chart(make_renewables_chart(day_rows["Date"].values, solar=solar_vals, wind=wind_vals),
                                width="stretch")

        with st.container(border=True):
            st.subheader("Weather")
            weather_choice = st.selectbox("Show hourly:", ["Hide", "Temperature", "Humidity"])
            if weather_choice == "Temperature":
                st.plotly_chart(make_simple_chart(day_rows["Date"].values, day_rows["temperature_2m"].values,
                                                "Temperature (°C)"), width="stretch")
            elif weather_choice == "Humidity":
                st.plotly_chart(make_simple_chart(day_rows["Date"].values, day_rows["relative_humidity_2m"].values,
                                                "Humidity (%)"), width="stretch")

    hour_of_slot = np.array([pd.Timestamp(ts).hour for ts in timestamps])
    time_label = [pd.Timestamp(ts).strftime("%H:%M") for ts in timestamps]

    # "adjusted" starts as an exact copy of "forecast", pre-rounded to 2
    # decimals here so it can never drift from the forecast column when
    # Streamlit's data editor re-serializes the editable column below --
    # the two columns need to be bit-identical until the expert actually
    # changes a value.
    working_df = pd.DataFrame({
        "timestamp_slot": timestamps,
        "hour": hour_of_slot,
        "time_label": time_label,
        "forecast": np.round(forecast, 2),
        "adjusted": np.round(forecast, 2),
        "flagged": flagged,
        "load_fr": day_rows["Load_FR"].values if context_available else np.nan,
    })

    key = f"{expert_id}_{forecast_date}"  # one working copy per (expert, date) pair in session state
    already_submitted = has_submitted(expert_id, forecast_date) if expert_id else False
    is_read_only = (current_role == "admin")

    if key not in st.session_state:
        # First time this (expert, date) combo is opened this session: if the
        # expert has a prior unsubmitted session for this exact date (they
        # navigated away and came back before hitting Submit), restore their
        # in-progress edits instead of resetting to the raw forecast.
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
                        "adjusted": st.column_config.NumberColumn("Adjusted", format="%.2f"),
                        "flagged": st.column_config.CheckboxColumn("Flag", disabled=True),
                    },
                    disabled=True,  # the whole grid is non-interactive; per-column disabled flags above are redundant but explicit
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
        # NB: no `step` on the Adjusted NumberColumn -- even a small step value
        # can snap the initial value to a step-aligned grid on first render in
        # some Streamlit versions, silently rounding away precision.
        with st.form(key=f"feedback_form_{key}"):
            edited_pieces = []
            for h in range(24):
                hour_slice = working[working["hour"] == h]
                n_flagged = int(hour_slice["flagged"].sum())
                label = f"{h:02d}:00"
                if n_flagged > 0:
                    label += f"  ⚠️ {n_flagged} flagged"
                with st.expander(label, expanded=(n_flagged > 0)):  # auto-open hours containing a flagged slot
                    edited_hour = st.data_editor(
                        hour_slice[["time_label", "forecast", "adjusted", "flagged"]],
                        column_config={
                            "time_label": st.column_config.TextColumn("Time", disabled=True),
                            "forecast": st.column_config.NumberColumn("DNN forecast", disabled=True, format="%.2f"),
                            "adjusted": st.column_config.NumberColumn("Adjusted", step=0.01, format="%.2f"),
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
                # Guards against a double-submit race (e.g. two tabs open on the same date).
                st.error("A submission already exists for this date. Refresh the page.")
            else:
                # Recombine the 24 separately-edited hour tables back into one
                # 96-row frame, then reattach the fields the editor never
                # touched (timestamp_slot, load_fr) before saving to SQLite.
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
                st.rerun()  # forces the page back into the read-only branch above


# --------------------------------------------------------------------------
# PAGE 2: REVEAL & EVALUATE
# --------------------------------------------------------------------------

def get_last_evaluable_ts(now=None):
    """Never evaluate against a delivery day whose day-ahead auction hasn't
    settled yet. Belgian day-ahead prices for a delivery day are published
    the day before delivery -- so "today's" prices are already known, but
    "tomorrow's" (the day currently being forecast, relative to `now`) are
    not, even if the actuals feed happens to already contain a stale/
    placeholder value for it. `now` is injectable for testing; production
    calls always use the default (current time in Europe/Brussels)."""
    now = now if now is not None else pd.Timestamp.now(tz="Europe/Brussels").tz_localize(None)
    tomorrow_start = now.normalize() + pd.Timedelta(days=1)
    return tomorrow_start - pd.Timedelta(minutes=15)


def page_reveal_and_evaluate():
    """Admin-only page: pick one expert's one-day submission and, once that
    day's prices have settled, compare the original DNN forecast and the
    expert's adjusted values against the realized price (MAE for each) to
    see whether the human adjustment helped, hurt, or made no difference."""
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

    # Join the saved submission to the realized price by timestamp -- this is
    # the only place "forecast"/"adjusted" (saved at submission time) meet
    # "actual" (fetched live), so MAE here always reflects the true settled price.
    actuals = be_df.loc[be_df["date_only"] == forecast_date, ["Date", "Price"]].rename(
        columns={"Date": "timestamp_slot", "Price": "actual"}
    )
    evaluation = submission.merge(actuals, on="timestamp_slot", how="inner")

    if evaluation.empty:
        st.info("No realized prices available yet for this date.")
        return

    forecast_mae = (evaluation["forecast"] - evaluation["actual"]).abs().mean()
    adjusted_mae = (evaluation["adjusted"] - evaluation["actual"]).abs().mean()
    confidence_rating = submission["confidence"].iloc[0]  # constant across the day's 96 rows

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
    """Admin-only page: aggregates every (expert, date) submission that has
    a settled actual price into a per-expert leaderboard -- average
    improvement in MAE, days reviewed, win rate (% of days where the
    adjustment beat the raw forecast), and average stated confidence."""
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
            continue  # skip days that haven't settled yet, same rule as Reveal & Evaluate

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
            "improvement": forecast_mae - adjusted_mae,  # positive = the expert helped
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
    """Login / self-registration screen, shown instead of any page content
    when nobody is logged in yet (see main()). Self-registration only offers
    the "expert" role (EXPERT_ROLES) -- admin accounts must be created
    directly in the database, not through this UI."""
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


def page_embedded_dashboard():
    """Embeds a collaborator's separate model-comparison dashboard (LEAR/
    XGB/DNN/Ensemble), hosted on its own Hugging Face Space, via an iframe.
    Deliberately not integrated any deeper than this: it's a different
    codebase with its own UI/theme, no shared session or login, and no
    dependency on this app's SQLite database -- if the Space URL ever
    changes, only this one string needs updating."""
    st.title("Margarida's Dashboard")
    st.caption(
        "Her dashboard, hosted on Hugging Face Spaces and embedded here via an "
        "iframe -- it has its own LEAR/XGB/DNN/Ensemble views and does not share "
        "a session or login with this app."
    )
    st.iframe("https://eds-lab-dam-price-forecast.hf.space/", height=1400)

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    """Entry point. Applies the theme first (so even the login screen is
    themed), gates everything behind login, then renders the sidebar nav and
    routes to the selected page. Available pages depend on role -- see the
    module docstring at the top of this file for what each role can see."""
    apply_theme()

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
            pages = ["Review & Adjust","Deterministic Forecast Analysis", "Reveal & Evaluate", "Expert Scoreboard"]
        else:
            pages = ["Review & Adjust", "Deterministic Forecast Analysis"]

        page = st.radio("Page", pages)

    if page == "Review & Adjust":
        page_review_and_adjust()
    elif page == "Reveal & Evaluate":
        page_reveal_and_evaluate()
    elif page == "Expert Scoreboard":
        page_expert_scoreboard()
    elif page == "Deterministic Forecast Analysis":
        page_embedded_dashboard()


if __name__ == "__main__":
    main()