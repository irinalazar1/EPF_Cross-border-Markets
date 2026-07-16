import streamlit as st
import os
from datetime import date, datetime, timedelta, timezone
import holidays
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Run the Streamlit Dashboard using 'streamlit run app.py'


st.set_page_config(page_title='EPF Expert Review', layout='wide')
st.title('EPF Expert Review')

# with st.sidebar:
#     expert = st.selectbox("label here", ["a", "b", "c"])

# st.write(f"...")

EXPERT_IDS = ["expert_1", "expert_2", "expert_3", "expert_4", "expert_5"]

@st.cache_data
def load_be_data():

    df = pd.read_csv("datasets/BE_Data_UTC.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["date_only"] = df["Date"].dt.date
    df["hour"] = df["Date"].dt.hour
    
    return df.sort_values("Date").reset_index(drop=True) #cleans up the row numbers after sorting so they run 0, 1, 2, ... instead of being scrambled.

df = load_be_data()
st.write(df.shape)
st.write(df.head())

def get_available_dates(df):
    all_dates = sorted(df["date_only"].unique())
    earliest = all_dates[0]
    cutoff = earliest + timedelta(days=7)
    usable = [d for d in all_dates if d >= cutoff]
    return usable
