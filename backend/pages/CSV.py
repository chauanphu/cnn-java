import streamlit as st
import pandas as pd
import io

st.title("CSV Upload")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    dataframe = pd.read_csv(io.StringIO(uploaded_file.decode('utf-8')))