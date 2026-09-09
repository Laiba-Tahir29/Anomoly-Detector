# =====================================================
# DAY 7 — Streamlit App
# Project: "Bhaai" - Sequence-Based Anomalous Behavior Detector
# =====================================================
# HOW TO RUN THIS ON KAGGLE:
# 1. Save this as app.py in /kaggle/working/
# 2. In a notebook cell, run:
#      !pip install streamlit pyngrok -q
#      !streamlit run /kaggle/working/app.py &>/kaggle/working/logs.txt &
#      (then use pyngrok or Kaggle's built-in tunneling to view it)
# OR simply run it locally after downloading all the .pkl files.
# =====================================================

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

st.set_page_config(page_title="Bhaai - Anomaly Detector", layout="wide")

DATA_DIR = "data/"
DATASET_PATH = "data/"  # products.csv should also be copied into the data/ folder

# -----------------------------------------------------
# Load all saved artifacts (cached so it only loads once)
# -----------------------------------------------------
@st.cache_resource
def load_everything():
    with open(DATA_DIR + "labeled_data.pkl", "rb") as f:
        labeled_data = pickle.load(f)
    with open(DATA_DIR + "item2idx.pkl", "rb") as f:
        item2idx = pickle.load(f)
    with open(DATA_DIR + "idx2item.pkl", "rb") as f:
        idx2item = pickle.load(f)
    with open(DATA_DIR + "gru_model_config.pkl", "rb") as f:
        model_config = pickle.load(f)
    with open(DATA_DIR + "gru_threshold.pkl", "rb") as f:
        threshold_info = pickle.load(f)

    products = pd.read_csv(DATASET_PATH + "products.csv")
    product_id_to_name = products.set_index("product_id")["product_name"].to_dict()

    # Rebuild + load GRU model
    class GRUNextItemModel(nn.Module):
        def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x):
            embedded = self.embedding(x)
            _, hidden = self.gru(embedded)
            hidden = hidden.squeeze(0)
            return self.fc(hidden)

    model = GRUNextItemModel(
        vocab_size=model_config["vocab_size"],
        embed_dim=model_config["embed_dim"],
        hidden_dim=model_config["hidden_dim"]
    )
    model.load_state_dict(torch.load(DATA_DIR + "gru_model.pt", map_location="cpu"))
    model.eval()

    return labeled_data, item2idx, idx2item, model, model_config, threshold_info, product_id_to_name

labeled_data, item2idx, idx2item, model, model_config, threshold_info, product_id_to_name = load_everything()
MAX_SEQ_LEN = model_config["max_seq_len"]
BEST_THRESHOLD = threshold_info["best_threshold"]

# -----------------------------------------------------
# Helper functions
# -----------------------------------------------------
def sequence_to_product_names(seq_idx, max_items=16):
    names = []
    for idx in seq_idx[-max_items:]:
        product_id = idx2item.get(idx)
        name = product_id_to_name.get(product_id, f"Unknown item ({product_id})")
        names.append(name)
    return names

def score_sequence_gru(seq, max_len=MAX_SEQ_LEN):
    step_scores = []
    with torch.no_grad():
        for i in range(1, len(seq)):
            window = seq[max(0, i - max_len):i]
            target = seq[i]
            padded = [0] * (max_len - len(window)) + window
            input_tensor = torch.tensor([padded], dtype=torch.long)
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
            step_scores.append(probs[target].item())
    return step_scores

def build_prompt(seq_names, step_scores):
    transition_lines = []
    for i in range(len(step_scores)):
        if i + 1 < len(seq_names):
            item_from, item_to, score = seq_names[i], seq_names[i + 1], step_scores[i]
            flag = "  ← very unexpected" if score < 0.01 else ""
            transition_lines.append(f"{item_from} → {item_to}    confidence: {score:.4f}{flag}")
    transitions_str = "\n".join(transition_lines[-15:])

    return f"""You are analyzing a user's e-commerce purchase sequence for anomaly detection.

Here is the sequence of purchases with the model's confidence for each transition
(lower confidence = more unexpected given the user's normal pattern):

{transitions_str}

In 2-3 plain-English sentences, explain to a non-technical reader:
1. What looks unusual about this sequence
2. Which specific transition (item → item) is the suspicious one, and why it stands out

Be concise and specific. Reference the actual item names, not scores."""

def call_llm(prompt):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        return "[No API key configured — add GEMINI_API_KEY to Streamlit secrets]"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )
    result = response.json()
    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return f"[Error getting explanation: {result}]"

# -----------------------------------------------------
# UI
# -----------------------------------------------------
st.title("🔍 Bhaai — Sequence-Based Anomaly Detector")
st.caption("Detects unusual purchase patterns without labeled fraud data, using a GRU sequence model + LLM explanations.")

user_list = list(labeled_data.keys())
selected_user = st.selectbox("Select a user to inspect:", user_list)

if selected_user:
    entry = labeled_data[selected_user]
    seq_idx = entry["sequence_idx"]
    seq_names = sequence_to_product_names(seq_idx)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Purchase History (most recent items)")
        st.write(", ".join(seq_names))

    with col2:
        st.subheader("Ground Truth")
        st.write("Anomaly injected:" , "✅ Yes" if entry["is_anomalous"] else "❌ No")

    if st.button("🔎 Run Anomaly Detection"):
        with st.spinner("Scoring sequence with GRU model..."):
            step_scores = score_sequence_gru(seq_idx)
            min_score = min(step_scores)
            is_flagged = min_score < BEST_THRESHOLD

        st.subheader("Model Result")
        if is_flagged:
            st.error(f"🚩 FLAGGED as anomalous (min confidence: {min_score:.5f}, threshold: {BEST_THRESHOLD:.5f})")
        else:
            st.success(f"✅ Looks normal (min confidence: {min_score:.5f}, threshold: {BEST_THRESHOLD:.5f})")

        if is_flagged:
            with st.spinner("Generating explanation..."):
                prompt = build_prompt(seq_names, step_scores)
                explanation = call_llm(prompt)
            st.subheader("🧠 LLM Explanation")
            st.write(explanation)

st.divider()
st.caption("Built as a Mitacs research prototype — Markov chain baseline + GRU sequence model + LLM explanation layer.")