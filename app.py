# =====================================================
# Streamlit App — Sequence-Based Anomaly Detector
# =====================================================

import streamlit as st
import pickle
import os

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import plotly.graph_objects as go
from html import escape


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Anomaly Detector", layout="wide", page_icon="🛰️")

DATA_DIR = "data/"
DATASET_PATH = "data/"


# =====================================================
# CUSTOM STYLING
# =====================================================

st.markdown("""<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.hero { padding: 2rem 0 1rem 0; border-bottom: 1px solid #23252f; margin-bottom: 2rem; }
.hero-title { font-size: 2.2rem; font-weight: 800; color: #f3f4f6; margin-bottom: 0.3rem; letter-spacing: -0.02em; }
.hero-subtitle { color: #8b8fa3; font-size: 1rem; margin-top: 0; }
.hero-kicker { color: #65e6d1; font-size: 0.72rem; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 0.5rem; }
.badge-row { display: flex; gap: 0.5rem; margin-top: 0.8rem; }
.badge { background: rgba(101,230,209,0.07); border: 1px solid rgba(101,230,209,0.2); color: #a8dcd4; padding: 0.25rem 0.7rem; border-radius: 6px; font-size: 0.78rem; font-weight: 500; }
.card { background: #14151c; border: 1px solid #23252f; border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1.2rem; }
.card-label { color: #6b6f83; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.6rem; }
.item-chip { display: inline-block; background: #1c1e29; border: 1px solid #2a2d3a; color: #d1d3de; padding: 0.25rem 0.65rem; border-radius: 20px; font-size: 0.85rem; margin: 0.15rem 0.25rem 0.15rem 0; }
.scene-card { background: radial-gradient(circle at 90% 10%, rgba(101,230,209,0.09), transparent 36%), #11151b; border: 1px solid #24343a; border-radius: 16px; padding: 1rem 1rem 0.4rem; margin: 1.25rem 0; }
.scene-title { color: #e4fbf6; font-size: 1.05rem; font-weight: 750; margin-bottom: 0.2rem; }
.scene-copy { color: #7f9b9b; font-size: 0.86rem; margin-bottom: 0.4rem; }
.pill { display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px; font-size: 0.85rem; font-weight: 700; }
.pill-yes { background: rgba(248, 113, 113, 0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
.pill-no { background: rgba(74, 222, 128, 0.12); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
.stButton > button { background: linear-gradient(135deg, #4f46e5, #7c3aed); color: white; border: none; border-radius: 10px; padding: 0.65rem 1.8rem; font-weight: 600; font-size: 0.95rem; box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25); }
.stButton > button:hover { box-shadow: 0 6px 18px rgba(79, 70, 229, 0.4); }
.result-flagged { background: linear-gradient(135deg, rgba(248,113,113,0.08), rgba(248,113,113,0.02)); border: 1px solid rgba(248,113,113,0.25); border-radius: 14px; padding: 1.4rem 1.6rem; }
.result-normal { background: linear-gradient(135deg, rgba(74,222,128,0.08), rgba(74,222,128,0.02)); border: 1px solid rgba(74,222,128,0.25); border-radius: 14px; padding: 1.4rem 1.6rem; }
.score-meta { color: #6b6f83; font-size: 0.85rem; margin-top: 0.4rem; }
.explanation-box { background: #14151c; border: 1px solid #2a2440; border-left: 3px solid #8b5cf6; border-radius: 12px; padding: 1.4rem 1.6rem; line-height: 1.65; color: #d1d3de; }
.metric-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.7rem; margin: 0.5rem 0 1.1rem; }
.metric-box { background: #11151b; border: 1px solid #263039; border-radius: 10px; padding: 0.8rem 1rem; }
.metric-value { color: #e4fbf6; font-size: 1.15rem; font-weight: 750; }
.metric-name { color: #708184; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }
</style>""", unsafe_allow_html=True)


# =====================================================
# SETTINGS
# =====================================================

def get_setting(name, default=None):
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.getenv(name, default)


# =====================================================
# LOAD ALL SAVED ARTIFACTS
# =====================================================

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

    # Day 5 final threshold — falls back to gru_threshold.pkl if
    # final_thresholds.pkl wasn't generated
    try:
        with open(DATA_DIR + "final_thresholds.pkl", "rb") as f:
            final_thresholds = pickle.load(f)
        gru_threshold = final_thresholds["GRU (main model)"]
    except FileNotFoundError:
        with open(DATA_DIR + "gru_threshold.pkl", "rb") as f:
            threshold_info = pickle.load(f)
        gru_threshold = threshold_info["best_threshold"]

    products = pd.read_csv(DATASET_PATH + "products.csv")
    product_id_to_name = products.set_index("product_id")["product_name"].to_dict()

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

    return labeled_data, item2idx, idx2item, model, model_config, gru_threshold, product_id_to_name


# =====================================================
# LOAD SAVED LLM EXPLANATIONS
# =====================================================

@st.cache_data
def load_saved_explanations():
    path = DATA_DIR + "llm_explanations.pkl"
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return {}


# =====================================================
# LOAD EVERYTHING
# =====================================================

(labeled_data, item2idx, idx2item, model, model_config,
 BEST_THRESHOLD, product_id_to_name) = load_everything()

MAX_SEQ_LEN = model_config["max_seq_len"]


# =====================================================
# HELPER — SEQUENCE → PRODUCT NAMES
# =====================================================

def sequence_to_product_names(seq_idx, max_items=16):
    names = []
    for idx in seq_idx[-max_items:]:
        product_id = idx2item.get(idx)
        name = product_id_to_name.get(product_id, f"Unknown item ({product_id})")
        names.append(name)
    return names


# =====================================================
# GRU SCORING
# =====================================================

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


def sequence_figure(seq_names, step_scores):
    """Build a simple 2D transition probability graph."""
    
    transitions = min(len(step_scores), len(seq_names) - 1)

    scores = step_scores[-transitions:]
    names_from = seq_names[-transitions - 1:-1]
    names_to = seq_names[-transitions:]

    positions = list(range(1, transitions + 1))

    # Normal vs suspicious transitions
    normal_x = []
    normal_y = []
    anomaly_x = []
    anomaly_y = []

    for position, score in zip(positions, scores):
        if score < BEST_THRESHOLD:
            anomaly_x.append(position)
            anomaly_y.append(score)
        else:
            normal_x.append(position)
            normal_y.append(score)

    figure = go.Figure()

    # Normal transitions
    figure.add_trace(go.Scatter(
        x=normal_x,
        y=normal_y,
        mode="lines+markers",
        name="Expected",
        line={
            "color": "#416b70",
            "width": 3
        },
        marker={
            "size": 8,
            "color": "#65e6d1"
        },
        hovertemplate=(
            "<b>Step %{x}</b><br>"
            "Next-item probability: %{y:.4e}"
            "<extra></extra>"
        )
    ))

    # Suspicious transitions
    figure.add_trace(go.Scatter(
        x=anomaly_x,
        y=anomaly_y,
        mode="markers",
        name="Suspicious",
        marker={
            "size": 13,
            "color": "#ff8066",
            "line": {
                "color": "#ffd6cc",
                "width": 1
            }
        },
        text=[
            f"{source} → {target}"
            for source, target in zip(
                [names_from[i - 1] for i in anomaly_x],
                [names_to[i - 1] for i in anomaly_x]
            )
        ],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Step: %{x}<br>"
            "Next-item probability: %{y:.4e}"
            "<br>⚠ Suspicious transition"
            "<extra></extra>"
        )
    ))

    # Threshold line
    figure.add_hline(
        y=BEST_THRESHOLD,
        line_dash="dash",
        line_width=2,
        line_color="#ff8066",
        annotation_text="Anomaly threshold",
        annotation_position="top right"
    )

    figure.update_layout(
        height=430,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": "#b7c8c7",
            "family": "Arial"
        },
        xaxis={
            "title": "Sequence step",
            "gridcolor": "#26383b",
            "zerolinecolor": "#26383b"
        },
        yaxis={
            "title": "GRU next-item probability",
            "gridcolor": "#26383b",
            "zerolinecolor": "#26383b",
            "type": "log"
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1
        }
    )

    return figure


# =====================================================
# BUILD GEMINI PROMPT
# =====================================================

def build_prompt(seq_names, step_scores):
    recent_names = seq_names[-16:]
    recent_scores = step_scores[-15:]

    transition_lines = []
    for i, score in enumerate(recent_scores):
        item_from = recent_names[i]
        item_to = recent_names[i + 1]
        flag = "  ← very unexpected" if score < BEST_THRESHOLD else ""
        transition_lines.append(f"{item_from} → {item_to}    confidence: {score:.4e}{flag}")

    transitions_str = "\n".join(transition_lines)

    return f"""You are analyzing a user's e-commerce purchase sequence for anomaly detection.

The GRU sequence model estimates how expected each next purchase is based on the preceding purchase history.
Lower confidence means the transition was less expected according to the model.

Purchase transitions:
{transitions_str}

In 2-3 plain-English sentences, explain to a non-technical reader:
1. What looks unusual about this sequence.
2. Which specific transition (item → item) is the most suspicious and why it stands out.

Be concise and specific. Reference the actual product names.
Do not claim fraud, theft, account takeover, or malicious behavior. Only explain why the transition is unusual according to the sequence model."""


# =====================================================
# GEMINI API
# =====================================================

def call_llm(prompt):
    api_key = get_setting("GEMINI_API_KEY") or get_setting("GOOGLE_API_KEY")
    if not api_key:
        return "[No API key configured. Add GEMINI_API_KEY to your .env file or Streamlit secrets.]"

    model_name = get_setting("GEMINI_MODEL", "gemini-3.1-flash-lite")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    try:
        response = requests.post(
            url,
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=45,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        return f"[Gemini request failed: {exc}]"
    except ValueError:
        return "[Gemini returned an invalid response. Check the API configuration.]"

    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return f"[Error getting explanation: {result}]"


# =====================================================
# SAVED EXPLANATION
# =====================================================

def get_saved_explanation(user_id):
    saved = load_saved_explanations().get(user_id, {})
    explanation = saved.get("explanation") if isinstance(saved, dict) else None
    if explanation and not explanation.startswith("[Error"):
        return explanation
    return None


# =====================================================
# UI — HEADER
# =====================================================

st.markdown("""<div class="hero">
<div class="hero-kicker">Behavior intelligence / live inspection</div>
<div class="hero-title">🛰️ Sequence-Based Anomaly Detector</div>
<p class="hero-subtitle">Flagging unusual user behavior without labeled fraud data — self-supervised sequence modeling + LLM-generated explanations.</p>
<div class="badge-row">
<span class="badge">GRU sequence model</span>
<span class="badge">Markov chain baseline</span>
<span class="badge">LLM explanation layer</span>
</div>
</div>""", unsafe_allow_html=True)


# =====================================================
# USER SELECTION
# =====================================================

user_list = list(labeled_data.keys())
selected_user = st.selectbox("Select a user to inspect", user_list, label_visibility="visible")


# =====================================================
# DISPLAY USER
# =====================================================

if selected_user:
    entry = labeled_data[selected_user]
    seq_idx = entry["sequence_idx"]
    seq_names = sequence_to_product_names(seq_idx)

    col1, col2 = st.columns([2.5, 1])

    with col1:
        chips = "".join([f'<span class="item-chip">{name}</span>' for name in seq_names])
        st.markdown(f'<div class="card"><div class="card-label">Purchase History (Most Recent)</div><div>{chips}</div></div>', unsafe_allow_html=True)

    with col2:
        pill_class = "pill-yes" if entry["is_anomalous"] else "pill-no"
        pill_text = "Yes" if entry["is_anomalous"] else "No"
        st.markdown(f'<div class="card"><div class="card-label">Ground Truth</div><span class="pill {pill_class}">Anomaly injected: {pill_text}</span></div>', unsafe_allow_html=True)

    st.write("")
    run_clicked = st.button("Run Anomaly Detection")

    if run_clicked:
        with st.spinner("Scoring sequence with GRU model..."):
            step_scores = score_sequence_gru(seq_idx)
            min_score = min(step_scores)
            is_flagged = min_score < BEST_THRESHOLD

        st.write("")

        anomaly_count = sum(score < BEST_THRESHOLD for score in step_scores)
        safe_count = len(step_scores) - anomaly_count
        st.markdown(
            f'<div class="metric-strip">'
            f'<div class="metric-box"><div class="metric-name">Transitions</div><div class="metric-value">{len(step_scores)}</div></div>'
            f'<div class="metric-box"><div class="metric-name">Low-confidence steps</div><div class="metric-value">{anomaly_count}</div></div>'
            f'<div class="metric-box"><div class="metric-name">Stable steps</div><div class="metric-value">{safe_count}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if is_flagged:
            st.markdown(f'<div class="result-flagged"><div class="card-label" style="color:#f87171;">Model Result</div><div style="color:#f87171; font-weight:700; font-size:1.2rem;">🚩 Flagged as anomalous</div><div class="score-meta">min next-item probability: {min_score:.2e} &nbsp;·&nbsp; threshold: {BEST_THRESHOLD:.2e}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-normal"><div class="card-label" style="color:#4ade80;">Model Result</div><div style="color:#4ade80; font-weight:700; font-size:1.2rem;">✅ Looks normal</div><div class="score-meta">min next-item probability: {min_score:.2e} &nbsp;·&nbsp; threshold: {BEST_THRESHOLD:.2e}</div></div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="scene-card">'
            '<div class="scene-title">Purchase Transition Analysis</div>'
            '<div class="scene-copy">'
            'Each point represents one purchase transition. Lower probability means the GRU found that transition less expected. '
            'Points below the threshold are flagged as suspicious.'
            '</div>',
            unsafe_allow_html=True
        )
        st.plotly_chart(sequence_figure(seq_names, step_scores), use_container_width=True, config={"displaylogo": False, "scrollZoom": False})
        st.markdown('</div>', unsafe_allow_html=True)

        if is_flagged:
            with st.spinner("Generating explanation..."):
                prompt = build_prompt(seq_names, step_scores)
                explanation = call_llm(prompt)
                if explanation.startswith("[No API key"):
                    explanation = get_saved_explanation(selected_user) or explanation

            st.write("")
            st.markdown(f'<div class="explanation-box"><div class="card-label" style="color:#c4b5fd;">LLM Explanation</div>{explanation}</div>', unsafe_allow_html=True)


# =====================================================
# FOOTER
# =====================================================

st.write("")
st.write("")
st.markdown('<p style="color:#4a4d5e; font-size:0.85rem;">Markov chain baseline · GRU sequence model · LLM explanation layer</p>', unsafe_allow_html=True)