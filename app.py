import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import train_models

st.set_page_config(page_title="Model Evaluation Dashboard", layout="wide")

st.title("📊 AI Model Evaluation Dashboard")


df = train_models()


st.subheader("📌 Model Comparison Table")
st.dataframe(df)


model_name = st.selectbox("🔍 Select Model", df["Model"])

selected = df[df["Model"] == model_name].iloc[0]


st.subheader("🧠 Selected Model Metrics")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", selected["Accuracy"])
    st.metric("Precision", selected["Precision"])

with col2:
    st.metric("Recall", selected["Recall"])
    st.metric("F1-score", selected["F1-score"])


st.subheader("📈 Metrics Comparison")

fig, ax = plt.subplots(figsize=(10,5))

df_plot = df.set_index("Model")

df_plot.plot(kind="bar", ax=ax)

plt.xticks(rotation=0)
plt.ylabel("Score")
plt.title("Model Performance Comparison")

st.pyplot(fig)


st.subheader("📁 Download Results")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="results.csv",
    mime="text/csv"
)