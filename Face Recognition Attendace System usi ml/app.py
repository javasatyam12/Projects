
import streamlit as st
import pandas as pd
import time
import os
from datetime import datetime

# File where the attendance is stored
attendance_file = "attendance.csv"

# Check if the CSV file exists
if os.path.isfile(attendance_file):
    # Load the CSV file into a DataFrame
    attendance_df = pd.read_csv(attendance_file)
else:
    # Create an empty DataFrame if the CSV doesn't exist
    attendance_df = pd.DataFrame(columns=["Name", "Timestamp"])

# Streamlit UI
st.title("Real-Time Attendance")

st.write("Attendance data collected from webcam")

# Display the attendance data
st.dataframe(attendance_df)

# Get the current date and time
current_time = time.strftime('%H:%M:%S')
current_date = datetime.now().strftime('%d-%m-%Y')

# Display the last updated date and time
st.text(f"Last updated on: {current_date} at {current_time}")

