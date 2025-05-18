import threading
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from src.vae_few_shot import run_vae_few_shot
from src.vae_single_shot import run_vae_single_shot
from src.vae_rag import run_vae_rag
from src.bert_approach import run_bert_approach

st.set_page_config(page_title="Anomaly Detection ", layout="wide")
st.title("🔍 Anomaly Detection using Generative AI")

# Upload CSV
uploaded = st.file_uploader("Upload CSV file", type=["csv"])
if not uploaded:
    st.info("Please upload a CSV to proceed.")
    st.stop()

@st.cache_data
def load_data(f):
    return pd.read_csv(f)

df = load_data(uploaded)

# Scrollable Dataset Exploration
with st.expander("1) Dataset Exploration", expanded=True):
    num_cols = df.select_dtypes(include="number").columns.tolist()
    st.dataframe(df.head(), height=200)
    st.dataframe(df.describe(), height=200)
    for col in num_cols:
        fig, ax = plt.subplots()
        df[col].hist(ax=ax)
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)


# Now define the tabs
tabs = st.tabs([
    "VAE Few-Shot",
    "VAE Single-Shot",
    "VAE + RAG",
    "BERT Approach",
])


# VAE Few-Shot Tab
with tabs[0]:
    st.header("VAE Few-Shot")
    # 1) Run & store in session_state
    if st.button("Start Detection (VAE Few-Shot)"):
        log_area = st.empty()
        with st.spinner("Running VAE Few-Shot…"):
            st.session_state["few_shot_res"] = run_vae_few_shot(
                df, progress_callback=log_area.write
            )
        st.success("✅ Detection Complete!")
    # 2) Always display if available
    if "few_shot_res" in st.session_state:
        result = st.session_state["few_shot_res"]
        st.subheader("Anomaly Scores & Predictions")
        st.dataframe(result.head(), height=200)
        fig, ax = plt.subplots()
        result["anomaly_score"].hist(ax=ax)
        st.pyplot(fig)

        if "type" in result:
            y_true = result["type"].astype(int)
            y_vae  = (result["anomaly_score"] > result["threshold"]).astype(int)
            y_fs   = result["predicted_label"].map({"Normal":0, "Anomaly":1})

            st.subheader("🔍 VAE-Only Performance")
            cm1 = confusion_matrix(y_true, y_vae)
            st.write(cm1)
            st.text(classification_report(y_true, y_vae))

            st.subheader("🤖 VAE + Few-Shot Performance")
            cm2 = confusion_matrix(y_true, y_fs)
            st.write(cm2)
            st.text(classification_report(y_true, y_fs))

# VAE Single-Shot Tab
with tabs[1]:
    st.header("VAE Single-Shot")
    if st.button("Start Detection (VAE Single-Shot)"):
        log_area = st.empty()
        with st.spinner("Running VAE Single-Shot…"):
            st.session_state["single_shot_res"] = run_vae_single_shot(
                df, progress_callback=log_area.write
            )
        st.success("✅ Detection Complete!")
    if "single_shot_res" in st.session_state:
        result = st.session_state["single_shot_res"]
        st.subheader("Anomaly Scores & Predictions")
        st.dataframe(result.head(), height=200)
        fig, ax = plt.subplots()
        result["anomaly_score"].hist(ax=ax)
        st.pyplot(fig)

        if "type" in result:
            y_true = result["type"].astype(int)
            y_vae  = (result["anomaly_score"] > result["threshold"]).astype(int)
            y_ss   = result["predicted_label"].map({"Normal":0, "Anomaly":1})

            st.subheader("🔍 VAE-Only Performance")
            cm1 = confusion_matrix(y_true, y_vae)
            st.write(cm1)
            st.text(classification_report(y_true, y_vae))

            st.subheader("🤖 VAE + Single-Shot Performance")
            cm2 = confusion_matrix(y_true, y_ss)
            st.write(cm2)
            st.text(classification_report(y_true, y_ss))


# VAE + RAG Tab
with tabs[2]:
    st.header("VAE + RAG")
    if st.button("Start Detection (VAE + RAG)"):
        log_area = st.empty()
        with st.spinner("Running VAE + RAG…"):
            st.session_state["rag_res"] = run_vae_rag(
                df, progress_callback=log_area.write
            )
        st.success("✅ Detection Complete!")
    if "rag_res" in st.session_state:
        result = st.session_state["rag_res"]
        st.subheader("Anomaly Scores & Predictions")
        st.dataframe(result.head(), height=200)
        fig, ax = plt.subplots()
        result["anomaly_score"].hist(ax=ax)
        ax.set_title("Anomaly Score Distribution")
        st.pyplot(fig)

        if "type" in result.columns:
            y_true = result["type"].astype(int)
            y_vae  = (result["anomaly_score"] > result["threshold"]).astype(int)
            y_rag  = result["predicted_label"].map({"Normal":0,"Anomaly":1}).astype(int)

            st.subheader("🔍 VAE-Only Performance")
            cm1 = confusion_matrix(y_true, y_vae)
            st.write(pd.DataFrame(cm1,
                        index=["True Normal","True Anomaly"],
                        columns=["Pred Normal","Pred Anomaly"]))
            st.text(classification_report(y_true, y_vae,
                        target_names=["Normal","Anomaly"]))

            st.subheader("🤖 VAE + RAG Performance")
            cm2 = confusion_matrix(y_true, y_rag)
            st.write(pd.DataFrame(cm2,
                        index=["True Normal","True Anomaly"],
                        columns=["Pred Normal","Pred Anomaly"]))
            st.text(classification_report(y_true, y_rag,
                        target_names=["Normal","Anomaly"]))

# BERT Approach Tab
with tabs[3]:
    st.header("BERT Approach")

    # 1) Trigger & store result
    if st.button("Start Detection (BERT Approach)"):
        log_area = st.empty()
        with st.spinner("Running BERT Approach…"):
            # run_bert_approach currently doesn't stream logs, but you could add a progress_callback arg if desired
            st.session_state["bert_res"] = run_bert_approach(df)
        st.success("✅ Detection Complete!")

    # 2) Always render if we have it
    if "bert_res" in st.session_state:
        result = st.session_state["bert_res"]

        st.subheader("Predictions & Anomaly Probabilities")
        st.dataframe(result.head(), height=200)

        fig, ax = plt.subplots()
        result["anomaly_score"].hist(ax=ax)
        ax.set_title("Anomaly Score Distribution")
        st.pyplot(fig)

        # 3) Performance metrics
        if "type" in result.columns:
            y_true = result["type"].astype(int)
            y_pred = result["predicted_label"].map({"Normal":0, "Anomaly":1}).astype(int)

            st.subheader("🔍 BERT Performance")
            cm = confusion_matrix(y_true, y_pred)
            st.write(
                pd.DataFrame(
                    cm,
                    index=["True Normal", "True Anomaly"],
                    columns=["Pred Normal", "Pred Anomaly"]
                )
            )
            st.text(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

