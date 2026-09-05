"""
EPF Expert Review -- a Streamlit dashboard for human-in-the-loop review of
day-ahead electricity price forecasts for the Belgian market.

WHAT THIS APP IS FOR
---------------------
A neural network (DNN) produces a day-ahead price forecast for Belgium at
15-minute resolution (96 slots/day). Domain experts review that forecast
and adjust it by dragging the curve on a chart -- see draggable_curve, a
custom React component -- rather than editing 96 individual numbers by
hand, and rate their own confidence. Once the delivery day has actually
happened and its day-ahead auction has settled, admins can reveal the
realized price and see whether each expert's adjustment improved or
worsened accuracy (MAE) relative to the raw model forecast -- and aggregate
that across experts on a scoreboard. The goal is to measure the value of
human correction on top of the model, not just to collect opinions.
Anomaly flagging is fully automatic (5th/95th percentile of that day's own
forecast) -- there's no manual per-slot flag override anymore, now that
editing happens via drag rather than a per-slot table.

ROLES
-----
- "expert": can only see and submit on the Review & Adjust page, and only
  for their own submissions. Submissions are final -- no editing after
  submit.
- "admin": can view (but never submit on behalf of) any expert's Review &
  Adjust page, plus the Reveal & Evaluate and Expert Scoreboard pages.

DATA SOURCES
------------
Two CSVs are pulled live from a GitHub repo on every page load (cached
for 30 min to absorb the daily 10AM/14h refreshes without hammering
GitHub): the DNN forecast and the realized Belgian market data (price,
load, solar, wind, weather). See the GITHUB LOADING section below for the
exact files/columns.

UNCERTAINTY BANDS
------------------
The chart's uncertainty band is no longer a separate quantile-regression
(QR) model's output. It's a single 80%-coverage margin, conformal-
calibrated from the DNN forecast's own settled residuals via ACI (Adaptive
Conformal Inference) -- see get_aci_margin(). Chosen over the alternative
(WCP, a single fixed margin from historical residuals) because ACI
self-adjusts: if recent predictions have been missing the realized price,
the margin widens on its own; if they've been comfortably covering it, the
margin relaxes -- no manual recalibration needed after a volatile stretch.
Ported from a colleague's conformal-prediction notebook.

PAGES
-----
1. Review & Adjust       -- the core workflow described above.
2. Deterministic Forecast Analysis -- DNN forecast vs. actual settled price
   over the last 14 days. This is the DNN-only slice of a broader LEAR/XGB/
   DNN/Ensemble comparison a collaborator built separately; rendered
   natively here with this app's own data pipeline and theme, not embedded
   from her dashboard.
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
from collections import deque
from datetime import date, datetime, timedelta, timezone

import bcrypt
import holidays
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dotenv import load_dotenv

from db import init_db, load_users, save_new_user, save_feedback, load_feedback, has_submitted
from draggable_curve import draggable_curve

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
        stays stuck on its default color regardless of mode. Background is
        applied to every descendant, not just the outer container: unlike
        color, background-color doesn't cascade down through nested elements
        -- a sub-element with its own background (e.g. the header bar) keeps
        it regardless of what the parent container is set to. */
        div[data-baseweb="datepicker"], div[data-baseweb="calendar"],
        div[data-baseweb="datepicker"] *, div[data-baseweb="calendar"] * {{
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


def _forecast_vs_actual(dnn_df, be_df):
    """Settled (forecast, actual) pairs -- shared prep for both conformal
    methods below. Never includes the currently-forecast (unsettled) day,
    same cutoff rule used everywhere else in this app.

    Rows with a missing forecast or actual value are dropped, not just left
    in as NaN: ACI's residual pool is a fixed-size rolling window, so a
    single NaN entering it poisons every np.quantile() call downstream
    until that one value finally slides back out -- including, in the worst
    case, the very last one, silently returning a NaN margin overall."""
    last_evaluable = get_last_evaluable_ts()
    merged = (
        dnn_df.set_index("DateTime")["DNN_expanding"].rename("forecast")
        .to_frame()
        .join(be_df.set_index("Date")["Price"].rename("actual"), how="inner")
        .sort_index()
    )
    merged = merged.loc[merged.index <= last_evaluable]
    return merged.dropna(subset=["forecast", "actual"])


@st.cache_data(ttl=1800)
def get_aci_margin(dnn_df, be_df, alpha=0.2, gamma=0.01, calibration_days=30, min_calibration_days=5):
    """Adaptive Conformal Inference margin, ported from method_ACI() in the
    conformal-prediction notebook: after an initial calibration window,
    replays every subsequent settled slot one at a time, self-correcting a
    target quantile level (alpha_t) based on whether each prediction
    actually covered the realized price. Returns the margin q as of the END
    of that replay -- i.e. "right now" -- which is what gets applied to the
    (unsettled) date currently being reviewed.

    The calibration window is ADAPTIVE, not a hard 30-day requirement: it
    uses up to `calibration_days` of settled history, but shrinks down to
    whatever's actually available (as low as `min_calibration_days`) rather
    than refusing to produce a band at all just because the model hasn't
    accumulated a full 30 days yet -- relevant right now since the DNN model
    is newly launched and doesn't have that much history. At least one day
    is always reserved for the replay itself, or there's nothing to adapt.

    This is genuinely sequential (each step depends on the previous one's
    updated alpha_t), so it can't be vectorized -- the ttl cache is what
    keeps it from being recomputed on every page interaction.
    Returns None only if there's less than min_calibration_days + 1 days of
    settled history total -- genuinely not enough to do anything with yet."""
    merged = _forecast_vs_actual(dnn_df, be_df)

    total_days = len(merged) // STEPS_PER_DAY
    if total_days < min_calibration_days + 1:
        return None

    cal_days = min(calibration_days, total_days - 1)  # leave >=1 day for the replay
    n_cal = cal_days * STEPS_PER_DAY

    cal, val = merged.iloc[:n_cal], merged.iloc[n_cal:]
    eps_pool = deque(np.abs(cal["actual"].values - cal["forecast"].values))
    alpha_t = alpha
    q = None

    for mu, y in zip(val["forecast"].values, val["actual"].values):
        q = np.quantile(np.array(eps_pool), 1 - alpha_t, method="linear")
        covered = (y >= mu - q) and (y <= mu + q)
        alpha_t = np.clip(alpha_t + gamma * (alpha - (0 if covered else 1)), 1e-6, 1 - 1e-6)
        eps_pool.append(abs(y - mu))
        eps_pool.popleft()

    return float(q) if q is not None else None


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


def get_dnn_history_window(dnn_df, be_df, days=14):
    """DNN forecast vs actual over the last `days` days -- windowed exactly
    the way Margarida's dashboard_app.py does it in Section 1
    (get_plot_window_from_forecast + plot_two_series_allow_missing_actual),
    ported to this app's own dnn_df/be_df rather than her separate
    LEAR/XGB/Ensemble pipeline. Nothing else from her file (the scatter
    diagnostics in Section 2, the full-history MAE/rMAE tables in Section 3)
    is included here -- this is only the Section 1 chart, DNN-only.

    Two behaviors carried over deliberately, since they differ from a more
    "obvious" implementation:
      - The window ends at the LATEST available forecast timestamp, not the
        latest settled day -- so it can include tomorrow's not-yet-settled
        forecast, same as hers.
      - Actual is reindexed onto the forecast's own dates rather than
        inner-joined, so an unsettled day still shows its forecast line
        (just with a gap where the actual price isn't in yet) instead of
        disappearing from the window entirely.
    """
    forecast = dnn_df.set_index("DateTime")["DNN_expanding"].rename("forecast").sort_index()
    if forecast.empty:
        return pd.DataFrame(columns=["DateTime", "forecast", "actual"])

    plot_end = forecast.index.max()
    plot_start = plot_end - pd.Timedelta(days=days)
    forecast_window = forecast.loc[(forecast.index >= plot_start) & (forecast.index <= plot_end)]

    actual = be_df.set_index("Date")["Price"].rename("actual")
    actual = actual[~actual.index.duplicated(keep="last")].sort_index()
    actual_window = actual.reindex(forecast_window.index)

    history_df = pd.concat([forecast_window, actual_window], axis=1).reset_index()
    return history_df.rename(columns={"index": "DateTime"})


def make_chart(timestamps, forecast, band_lower=None, band_upper=None, band_label="80% interval",
                solar=None, wind=None, renewables_range=None, flagged=None):
    """The main Review & Adjust chart: DNN forecast line, one uncertainty
    band (now from ACI conformal calibration rather than the retired QR
    model -- see get_aci_margin), auto-flagged anomaly
    markers, and optional solar/wind traces on a secondary y-axis (MW). All
    inputs are optional except timestamps and forecast -- the band/
    renewables are simply omitted from the figure if not supplied, rather
    than erroring."""
    figure = go.Figure()

    if band_lower is not None and band_upper is not None:
        figure.add_trace(go.Scatter(x=timestamps, y=band_upper, mode="lines", line=dict(width=0), showlegend=False))
        figure.add_trace(go.Scatter(
            x=timestamps, y=band_lower, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(100,100,255,0.35)", name=band_label,
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


def make_comparison_chart(timestamps, forecast, adjusted):
    """DNN forecast vs. the expert's submitted adjustment, as two overlaid
    lines. Replaces the old 24-hourly-table read-only view: a single
    glanceable chart instead of 24 collapsed tables to click through. Only
    used for already-submitted or admin-viewed submissions -- live editing
    happens via the draggable_curve widget instead, not this chart."""
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=timestamps, y=forecast, mode="lines", name="DNN Forecast",
                                 line=dict(width=2)))
    figure.add_trace(go.Scatter(x=timestamps, y=adjusted, mode="lines", name="Expert Adjusted",
                                 line=dict(width=2, dash="dash")))
    ts_min = pd.Timestamp(np.asarray(timestamps).min())
    ts_max = pd.Timestamp(np.asarray(timestamps).max())
    figure.update_layout(
        xaxis_title="Time of day",
        yaxis_title="EUR / MWh",
        xaxis=dict(tickformat="%H:%M", dtick=3600000, range=[ts_min, ts_max]),
    )
    return themed(figure)


def make_history_chart(history_df):
    """Predicted vs actual price over a multi-day window, matching the look
    of Margarida's dashboard_app.py chart (DNN's terracotta line color, line
    weights, unified hover, range slider, title style) but kept on our own
    themed() wrapper instead of her fixed plotly_white template -- 'Actual'
    uses the theme's own text color instead of her hardcoded near-black, so
    it stays readable rather than nearly invisible against a dark background."""
    dark = st.session_state.get("dark_mode", True)
    palette = get_palette(dark)

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=history_df["DateTime"], y=history_df["actual"],
                                 mode="lines", name="Actual",
                                 line=dict(width=2.4, color=palette["text"])))
    figure.add_trace(go.Scatter(x=history_df["DateTime"], y=history_df["forecast"],
                                 mode="lines", name="DNN forecast",
                                 line=dict(width=2.2, color="#C97B63")))  # Margarida's DNN color
    figure.update_layout(
        title=dict(text="DNN vs Actual", x=0.01, xanchor="left", font=dict(size=20)),
        xaxis_title="DateTime", yaxis_title="EUR / MWh",
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True)),
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
            key="review_forecast_date",  # explicit key so this survives a page switch
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

    # Uncertainty band: ACI margin (see get_aci_margin), not the retired QR
    # model -- a single symmetric margin around the point forecast,
    # calibrated from settled (forecast, actual) history and continuously
    # self-adjusted (see the docstring for why ACI specifically).
    margin = get_aci_margin(dnn_df, be_df, alpha=0.2, gamma=0.01, calibration_days=30)
    band_label = "80% interval (ACI)"

    calendar_ctx = get_calendar_context(forecast_date)
    day_rows = be_df.loc[be_df["date_only"] == forecast_date].sort_values("Date")

    # Weather/load context is only available once the actuals feed has caught
    # up to this date -- e.g. tomorrow's forecast reviewed today won't have it yet.
    if len(day_rows) == STEPS_PER_DAY:
        be_demand = day_rows["Load_BE"].mean()   # gross, matching FR below (not net of solar/wind)
        avg_load_fr = day_rows["Load_FR"].mean()   # gross -- no Solar_FR/Wind_FR columns exist to net out anyway
        avg_temp = day_rows["temperature_2m"].mean()
        avg_hum = day_rows["relative_humidity_2m"].mean()
        context_available = True
    else:
        context_available = False

    c1, c2, c3 = st.columns(3)
    c1.metric("Day of week", calendar_ctx["day_of_week"])
    c2.metric("Holiday?", calendar_ctx["holiday_name"] if calendar_ctx["is_holiday"] else "No")
    c3.metric("Bridge day?", "Yes" if calendar_ctx["is_bridge_day"] else "No")

    # Net demand, French demand, and the temp/humidity averages are single
    # numbers -- they fit fine in the sidebar. The toggles below control
    # whether the actual hourly charts render, but the charts themselves
    # appear on the main page (see after Solar & Wind), not in the sidebar --
    # the sidebar widget just captures the on/off state.
    with st.sidebar:
        st.divider()
        st.subheader("Day info")
        if context_available:
            st.metric("Avg. BE demand (MW)", f"{be_demand:,.0f}")
            st.metric("Avg. French Demand (MW)", f"{avg_load_fr:,.0f}")
            st.metric("Avg. temp (°C)", f"{avg_temp:.1f}")
            st.metric("Avg. humidity (%)", f"{avg_hum:.0f}")

            show_temp_sidebar = st.toggle("Show Temperature plot")
            show_hum_sidebar = st.toggle("Show Humidity plot")
        else:
            st.info("Weather/load context not available for this date.")
            show_temp_sidebar = False
            show_hum_sidebar = False

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
        band_lower=(forecast - margin) if margin is not None else None,
        band_upper=(forecast + margin) if margin is not None else None,
        band_label=band_label,
        flagged=flagged
    )
    st.plotly_chart(fig, width="stretch")

    if margin is None:
        st.caption("Not enough settled history yet to calibrate an ACI band for this date.")
    else:
        st.caption(f"Current ACI margin: ± {margin:.2f} EUR/MWh")

    # Solar & Wind and Weather sit one under the other (not side-by-side):
    # Solar+wind are genuinely linked (their combined dip drives net demand
    # and price spikes -- Load_BE - Solar_BE - Wind_BE would be net demand),
    # so they share one chart here.
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

        # Driven by the "Show Temperature/Humidity plot" toggles in the
        # sidebar's Day info section -- the widgets live there, but the
        # actual charts render here on the main page, full width, rather
        # than cramped into the sidebar.
        if show_temp_sidebar and show_hum_sidebar:
            wc1, wc2 = st.columns(2)
            with wc1:
                st.plotly_chart(
                    make_simple_chart(day_rows["Date"].values, day_rows["temperature_2m"].values, "Temperature (°C)"),
                    width="stretch",
                )
            with wc2:
                st.plotly_chart(
                    make_simple_chart(day_rows["Date"].values, day_rows["relative_humidity_2m"].values, "Humidity (%)"),
                    width="stretch",
                )
        elif show_temp_sidebar:
            st.plotly_chart(
                make_simple_chart(day_rows["Date"].values, day_rows["temperature_2m"].values, "Temperature (°C)"),
                width="stretch",
            )
        elif show_hum_sidebar:
            st.plotly_chart(
                make_simple_chart(day_rows["Date"].values, day_rows["relative_humidity_2m"].values, "Humidity (%)"),
                width="stretch",
            )

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
        # Read-only view: a static comparison chart instead of 24 tables --
        # nothing here is interactive, so there's no per-cell state to manage.
        st.subheader("Submitted adjustment")
        st.plotly_chart(
            make_comparison_chart(working["timestamp_slot"].values, working["forecast"].values,
                                   working["adjusted"].values),
            width="stretch",
        )

        if is_read_only:
            st.info(f"Viewing {expert_id}'s submission (read-only — admins cannot submit on behalf of experts).")
        else:
            st.info(f"You've already submitted feedback for {forecast_date}. Submissions are final.")

    else:
        # Editable view: dragging IS the edit -- draggable_curve() updates
        # st.session_state[key]["adjusted"] and reruns on every drag-release
        # (see the component's own __init__.py), so `working` is already
        # current by the time Submit is clicked. No form needed: unlike the
        # old 24-table version, there's nothing left to batch -- the slider
        # and button are the only two other widgets on this branch, and
        # Streamlit reads their current values at click-time regardless.
        #
        # Trade-off worth knowing: the old per-slot "Flag" checkbox let an
        # expert manually flag/unflag individual slots. That's gone now --
        # `flagged` is purely the auto-detected 5th/95th percentile array
        # computed above, with no manual override.
        st.subheader("Drag to adjust")
        st.caption("Drag any point on the curve below — nearby points within 2 hours shift too.")
        dragged = draggable_curve(
            values=working["adjusted"].tolist(),
            labels=working["time_label"].tolist(),
            radius=8,
            height=360,
            key=f"drag_{key}",
        )
        if dragged != working["adjusted"].tolist():
            working["adjusted"] = dragged
            st.session_state[key] = working
            st.rerun()

        confidence = st.slider("How confident are you in these adjustments?", 1, 5, 3)
        submitted = st.button("Submit feedback")

        if submitted:
            if not expert_id:
                st.error("Error: No Expert ID found.")
            elif has_submitted(expert_id, forecast_date):
                # Guards against a double-submit race (e.g. two tabs open on the same date).
                st.error("A submission already exists for this date. Refresh the page.")
            else:
                rows = working.copy()
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


def page_dnn_history():
    """DNN-only slice of Margarida's Section 1 chart (forecast vs actual,
    last N days) -- natively rendered in this app's own theme using
    dnn_df/be_df, no iframe, no external dependency on her Hugging Face
    Space. Deliberately just the chart: her Section 2 (scatter diagnostics)
    and Section 3 (full-history MAE/rMAE tables) aren't part of this page."""
    st.title("Deterministic Forecast Analysis — DNN")
    st.caption("DNN forecast vs. actual settled price, last 14 days.")

    dnn_df = get_dnn_df()
    be_df = get_be_df()

    history_df = get_dnn_history_window(dnn_df, be_df, days=14)

    if history_df.empty:
        st.info("No DNN forecast data available yet.")
        return

    st.plotly_chart(make_history_chart(history_df), width="stretch")

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
        page_dnn_history()


if __name__ == "__main__":
    main()