import streamlit as st
import datetime as dt
import os
from datetime import date, datetime, timedelta, timezone
import holidays
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import os
import bcrypt

# Run the Streamlit Dashboard using 'streamlit run app.py'

st.set_page_config(page_title='EPF Expert Review', layout='wide')
st.title('EPF Expert Review')


# CONSTANTS
FEEDBACK_LOG_PATH = "expert_feedback.csv"
USERS_FILE_PATH = "users.json"

# DATA LOADING
@st.cache_data
def load_be_data():
    df = pd.read_csv("datasets/BE_Data_UTC.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["date_only"] = df["Date"].dt.date
    df["hour"] = df["Date"].dt.hour
    return df.sort_values("Date").reset_index(drop=True)

df = load_be_data()

# SHARED HELPER FUNCTIONS
def get_available_dates(df):
    # Dates that have at least 7 days of history behind them (needed by naive_forecast).
    all_dates = sorted(df["date_only"].unique())
    earliest = all_dates[0]
    cutoff = earliest + timedelta(days=7)
    usable = [d for d in all_dates if d >= cutoff]
    return usable

def naive_forecast(forecast_date, df):
    if forecast_date.weekday() in (1, 2, 3, 4):
        lag_days = 1
    else:
        lag_days = 7

    source_date = forecast_date - timedelta(days=lag_days)
    source_rows = df[df["date_only"] == source_date].sort_values("hour")

    if len(source_rows) != 24:
        return None, None

    return source_rows["Price"].values, source_date

def make_chart(hours, forecast, lower, upper):
    # Forecast line with a shaded uncertainty band.
    figure = go.Figure()

    upper_trace = go.Scatter(x=hours, y=upper, mode="lines", line=dict(width=0), showlegend=False)
    figure.add_trace(upper_trace)

    lower_trace = go.Scatter(
        x=hours, y=lower, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(100,100,255,0.2)", name="Uncertainty band",
    )
    figure.add_trace(lower_trace)

    scatter = go.Scatter(x=hours, y=forecast, mode="lines+markers", name="Naive Forecast")
    figure.add_trace(scatter)

    figure.update_layout(
        xaxis_title="Hour of day",
        yaxis_title="EUR / MWh",
        xaxis=dict(dtick=1),
    )

    return figure

def save_feedback(rows_df):
    # Append submitted rows to the CSV log, creating it if it doesn't exist (or is empty)
    if os.path.exists(FEEDBACK_LOG_PATH) and os.path.getsize(FEEDBACK_LOG_PATH) > 0:
        existing = pd.read_csv(FEEDBACK_LOG_PATH)
        combined = pd.concat([existing, rows_df], ignore_index=True)
    else:
        combined = rows_df
    combined.to_csv(FEEDBACK_LOG_PATH, index=False)

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
        "day_of_week": forecast_date.strftime("%A"),
        "is_bridge_day": is_bridge_day,
    }

def make_simple_chart(hours, values, y_title):
    figure = go.Figure()
    scatter = go.Scatter(x=hours, y=values, mode="lines+markers", name=y_title)
    figure.add_trace(scatter)
    figure.update_layout(
        xaxis_title="Hour of day",
        yaxis_title=y_title,
        xaxis=dict(dtick=1),
    )
    return figure

# st.write(get_calendar_context(date(2024, 1, 1)))   # known Belgian holiday — New Year's Day
# st.write(get_calendar_context(date(2024, 1, 15)))  # a random Monday, not a holiday

# HASHING HELPER FUNCTIONS
def hash_password(password):
    # bcrypt requires bytes, so we encode the string to utf-8
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    # Store as a regular string in the JSON file
    return hashed.decode('utf-8')

def check_password(password, hashed_password):
    # Verify the provided password against the stored hash
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

# USER MANAGEMENT FUNCTIONS
def load_users():
    if not os.path.exists(USERS_FILE_PATH):
        # Hash the default admin password upon creation and give them a default email
        hashed_admin = hash_password("admin_password")
        default_users = {
            "admin": {
                "password": hashed_admin, 
                "role": "admin",
                "email": "admin@example.com"
            }
        }
        with open(USERS_FILE_PATH, "w") as f:
            json.dump(default_users, f)
        return default_users
    
    with open(USERS_FILE_PATH, "r") as f:
        return json.load(f)

def save_new_user(username, password, email, role):
    users = load_users()
    # Hash the new user's password and include their email address
    hashed_pass = hash_password(password)
    users[username] = {
        "password": hashed_pass, 
        "role": role, 
        "email": email.strip().lower() # Normalize emails to lowercase
    }
    with open(USERS_FILE_PATH, "w") as f:
        json.dump(users, f)


# PAGE 1: REVIEW & ADJUST
def page_review_and_adjust():
    current_user = st.session_state["logged_in_user"]
    current_role = st.session_state["role"]

    with st.sidebar:
        # ROLE-BASED ACCESS DEFINES THE EXPERT_ID
        if current_role == "admin":
            users = load_users()
            expert_list = [u for u, d in users.items() if d["role"] == "expert"]
            
            # 1. If Admin: expert_id is defined by whichever expert they select from the dropdown
            expert_id = st.selectbox("Expert ID (Admin View)", expert_list)
            
        else:
            # 2. If Expert: expert_id is strictly defined as their own username
            expert_id = current_user
            st.write(f"**Expert ID:** {expert_id}")

        available_dates = get_available_dates(df)
        forecast_date = st.date_input(
            "Forecast date (day d+1)",
            value=available_dates[-1],
            min_value=available_dates[0],
            max_value=available_dates[-1],
        )
  
    forecast, source_date = naive_forecast(forecast_date, df)
    hours = np.arange(24)

    calendar_ctx = get_calendar_context(forecast_date)
    day_rows = df.loc[df["date_only"] == source_date].sort_values("hour")
    net_demand = day_rows["Load"] - day_rows["Solar"] - day_rows["Wind"]
    avg_temp = day_rows["Temp"].mean()
    avg_hum = day_rows["Hum"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Day of week", calendar_ctx["day_of_week"][:3])
    c2.metric("Holiday?", calendar_ctx["holiday_name"] if calendar_ctx["is_holiday"] else "No")
    c3.metric("Bridge day?", "Yes" if calendar_ctx["is_bridge_day"] else "No")
    c4.metric("Avg. net demand (MW)", f"{net_demand.mean():,.0f}")
    c5.metric("Avg. temp (°C)", f"{avg_temp:.1f}")
    c6.metric("Avg. humidity (%)", f"{avg_hum:.0f}")

    band_width = 10  # EUR/MWh, placeholder -- to compute properly later using Margarida's uncertainty
    lower = forecast - band_width
    upper = forecast + band_width

    if forecast is not None:
        fig = make_chart(hours, forecast, lower, upper)
        st.plotly_chart(fig)

    weather_choice = st.selectbox("Show hourly:", ["Temperature", "Humidity"])
    if weather_choice == "Temperature":
        fig2 = make_simple_chart(hours, day_rows["Temp"].values, "Temperature (°C)")
        st.plotly_chart(fig2)
    elif weather_choice == "Humidity":
        fig2 = make_simple_chart(hours, day_rows["Hum"].values, "Humidity (%)")
        st.plotly_chart(fig2)

    low_threshold = np.percentile(forecast, 5)
    high_threshold = np.percentile(forecast, 95)

    flagged = (forecast <= low_threshold) | (forecast >= high_threshold)

    working_df = pd.DataFrame({
       "hour": hours,
       "forecast": forecast,
       "adjusted": forecast.copy(),
       "flagged": flagged,
       "price_ch": day_rows["Price_CH"].values,
       "price_de_lu": day_rows["Price_DE_LU_15min"].values,
       "price_at": day_rows["Price_AT_15min"].values,
    })

    key = f"{expert_id}_{forecast_date}"

    if key not in st.session_state:
        # --- THE FIX: Try to load past adjustments from the CSV ---
        if os.path.exists(FEEDBACK_LOG_PATH) and os.path.getsize(FEEDBACK_LOG_PATH) > 0:
            log = pd.read_csv(FEEDBACK_LOG_PATH)
            log["forecast_date"] = pd.to_datetime(log["forecast_date"]).dt.date
            
            # Find past submissions for this specific expert and date
            past_sub = log[(log["expert_id"] == expert_id) & (log["forecast_date"] == forecast_date)]
            
            if not past_sub.empty:
                # Take the most recent 24 rows and sort them properly
                past_sub = past_sub.tail(24).sort_values("hour")
                
                # Safely overwrite the default columns with the saved work
                if len(past_sub) == 24:
                    working_df["adjusted"] = past_sub["adjusted"].values
                    working_df["flagged"] = past_sub["flagged"].values

        st.session_state[key] = working_df

    working = st.session_state[key]
    edited = st.data_editor(working, 
                            column_config={
           "price_ch": st.column_config.NumberColumn("CH price", disabled=True),
           "price_de_lu": st.column_config.NumberColumn("DE-LU price", disabled=True),
           "price_at": st.column_config.NumberColumn("AT price", disabled=True),
           "flagged": st.column_config.CheckboxColumn("Flag volatility"),
       },
       key=f"editor_{key}")

    st.session_state[key] = edited

    if st.button("Submit feedback"):
            # --- NEW SAFEGUARD ---
            if not expert_id:
                st.error("Error: No Expert ID found. Cannot save data. (If you are an Admin, ensure an expert exists in the system first).")
            else:
                rows = edited.copy()
                rows["expert_id"] = expert_id
                rows["forecast_date"] = forecast_date
                rows["timestamp"] = dt.datetime.now(dt.timezone.utc).isoformat()
                save_feedback(rows)
                st.success(f"Saved {len(rows)} rows for {expert_id} on {forecast_date}.")


# PAGE 2: REVEAL & EVALUATE
def page_reveal_and_evaluate():
    st.title("Reveal & Evaluate")

    if not os.path.exists(FEEDBACK_LOG_PATH) or os.path.getsize(FEEDBACK_LOG_PATH) == 0:
        st.warning("No submissions yet.")
        return

    log = pd.read_csv(FEEDBACK_LOG_PATH)
    if log.empty:
        st.warning("No submissions yet.")
        return

    expert_id = st.selectbox("Expert ID", sorted(log["expert_id"].dropna().unique()))
    log["forecast_date"] = pd.to_datetime(log["forecast_date"]).dt.date
    available_dates = sorted(log.loc[log["expert_id"] == expert_id, "forecast_date"].dropna().unique())
    forecast_date = st.selectbox("Forecast date", available_dates)

    submission = (
        log.loc[
            (log["expert_id"] == expert_id) & (log["forecast_date"] == forecast_date),
            ["hour", "forecast", "adjusted"],
        ]
        .sort_values("hour")
    )
    actuals = df.loc[df["date_only"] == forecast_date, ["hour", "Price"]].rename(columns={"Price": "actual"})
    evaluation = submission.merge(actuals, on="hour", how="inner")

    forecast_mae = (evaluation["forecast"] - evaluation["actual"]).abs().mean()
    adjusted_mae = (evaluation["adjusted"] - evaluation["actual"]).abs().mean()

    forecast_metric, adjusted_metric = st.columns(2)
    forecast_metric.metric("Forecast MAE", f"{forecast_mae:.2f} EUR/MWh")
    adjusted_metric.metric("Adjusted MAE", f"{adjusted_mae:.2f} EUR/MWh")

    if adjusted_mae < forecast_mae:
        st.write("Verdict: the expert adjustment improved the forecast.")
    elif adjusted_mae > forecast_mae:
        st.write("Verdict: the expert adjustment worsened the forecast.")
    else:
        st.write("Verdict: the expert adjustment made no difference.")


# PAGE 3: Expert Scoreboard
def page_expert_scoreboard():
    st.title("Expert Scoreboard")

    if not os.path.exists(FEEDBACK_LOG_PATH) or os.path.getsize(FEEDBACK_LOG_PATH) == 0:
        st.warning("No submissions yet.")
        return

    log = pd.read_csv(FEEDBACK_LOG_PATH)
    if log.empty:
        st.warning("No submissions yet.")
        return

    log["forecast_date"] = pd.to_datetime(log["forecast_date"]).dt.date

    results = []
    for (expert_id, forecast_date), group in log.groupby(["expert_id", "forecast_date"]):
        actuals = df.loc[df["date_only"] == forecast_date, ["hour", "Price"]].rename(columns={"Price": "actual"})
        evaluation = group.merge(actuals, on="hour", how="inner")

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
        })

    if not results:
        st.warning("No submissions overlap with available actual prices yet.")
        return

    results_df = pd.DataFrame(results)

    scoreboard = (
        results_df.groupby("expert_id")
        .agg(
            avg_improvement=("improvement", "mean"),
            days_reviewed=("forecast_date", "nunique"),
            win_rate=("improvement", lambda s: (s > 0).mean()),
        )
        .reset_index()
        .sort_values("avg_improvement", ascending=False)
    )
    scoreboard["win_rate"] = (scoreboard["win_rate"] * 100).round(1)
    scoreboard["avg_improvement"] = scoreboard["avg_improvement"].round(2)

    st.dataframe(scoreboard, hide_index=True)

def auth_screen():
    st.subheader("Welcome to EPF Expert Review")
    
    # Use a horizontal radio button instead of tabs to prevent state resets
    auth_mode = st.radio("Choose an option:", ["Log In", "Create Account"], horizontal=True)
    st.divider()
    
    # ------------------ LOG IN ------------------
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
                
    # -------------- CREATE ACCOUNT --------------
    elif auth_mode == "Create Account":
        new_email = st.text_input("Email Address", key="new_email")
        new_user = st.text_input("New Username", key="new_user")
        new_pass = st.text_input("New Password", type="password", key="new_pass")
        new_role = st.selectbox("Role", ["expert", "admin"], key="new_role")
        
        if st.button("Create Account"):
            users = load_users()
            
            # Clean up inputs
            username_input = new_user.strip()
            email_input = new_email.strip().lower()
            
            # Check if email already exists in the system
            email_exists = any(account.get("email") == email_input for account in users.values())
            
            if not username_input or not email_input or not new_pass:
                st.error("All fields (Username, Email, and Password) are required.")
            elif username_input in users:
                st.error("This username is already taken. Please choose another.")
            elif email_exists:
                st.error("This email address is already registered. Please use another or log in.")
            else:
                # Pass the selected role to the save function
                save_new_user(username_input, new_pass, email_input, new_role)
                st.success("Account created successfully! You can now switch to the Log In option.")


# MAIN
def main():
    # Gate the app with the authentication screen
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
        
        # ROLE-BASED NAVIGATION: Hide evaluation pages from experts
        if current_role == "admin":
            # Admins can see everything
            pages = ["Review & Adjust", "Reveal & Evaluate", "Expert Scoreboard"]
        else:
            # Experts can only see the adjustment page
            pages = ["Review & Adjust"]
            
        page = st.radio("Page", pages)

    # Route to the selected page
    if page == "Review & Adjust":
        page_review_and_adjust()
    elif page == "Reveal & Evaluate":
        page_reveal_and_evaluate()
    elif page == "Expert Scoreboard":
        page_expert_scoreboard()


if __name__ == "__main__":
    main()
