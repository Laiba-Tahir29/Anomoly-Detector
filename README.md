# Sequence-Based Anomalous Behavior Detector

A proof-of-concept anomaly detection system for e-commerce/fintech platforms that flags unusual user behavior **without requiring labeled fraud data** — a common real-world constraint since fraud data is rare and privacy-restricted.

## Problem

E-commerce and banking platforms need to detect when a user's behavior deviates from their normal pattern (e.g., account takeover, fraud, bot activity). Real fraud-labeled data is hard to obtain due to privacy restrictions and the rarity of fraud events. This project explores whether a sequence-prediction deep learning model — trained only on normal behavior — can flag anomalous sequences without any labeled fraud data.

## Approach

1. **Data**: Instacart Market Basket Analysis dataset (used as a proxy for transactional sequence data — the same structure applies to banking/e-commerce transaction logs).
2. **Synthetic anomaly injection**: Since real fraud labels don't exist, we inject controlled, department-aware anomalies into ~8% of sequences to create ground truth for evaluation.
3. **Baseline model**: Markov chain — learns first-order transition probabilities from normal sequences.
4. **Main model**: GRU (Gated Recurrent Unit) — learns to predict the next item in a sequence using the full context, trained only on normal behavior (self-supervised).
5. **Anomaly scoring**: Low model confidence on an observed transition = anomalous signal.
6. **Evaluation**: Precision / Recall / F1, with threshold tuned on a held-out validation set (not the final test set, to avoid leakage).
7. **LLM explanation layer**: For sequences flagged as anomalous, an LLM (Gemini) generates a plain-English explanation of which transition is suspicious and why.
8. **Demo**: Interactive Streamlit app — select a user, run detection, see the flag and LLM explanation live.

## Results

| Model | Precision | Recall | F1-Score |
|---|---|---|---|
| Markov Chain (baseline) | *see model_comparison.csv* | | |
| GRU (main model) | *see model_comparison.csv* | | |

(Full comparison table generated in `evaluate.py`, saved as `model_comparison.csv`)

## Tech Stack

- Python, PyTorch (GRU model)
- pandas, NumPy, scikit-learn (data processing, evaluation)
- Google Gemini API (LLM explanation layer)
- Streamlit (interactive demo)

## Project Structure

```
data_prep.py          # Data cleaning, sequence construction, anomaly injection
baseline_markov.py     # Markov chain baseline model
train_gru.py           # GRU model definition and training
anomaly_scoring.py      # GRU-based anomaly scoring
evaluate.py             # Validation/test split, final comparison
llm_explanation.py      # LLM explanation layer
app.py                  # Streamlit demo app
```

## Real-World Applicability

This project uses Instacart data because real fraud-labeled transaction data (banking/e-commerce) is not publicly available due to privacy restrictions. Instacart's structure (per-user, time-ordered sequence of actions) is directly analogous to fraud-detection use cases. The same methodology — sequence learning + unsupervised anomaly scoring — is transferable to real banking or e-commerce transaction logs, with the input features swapped (e.g., transaction type/amount/merchant category instead of product IDs).

## Setup

```bash
pip install -r requirements.txt
```

Download the [Instacart Market Basket Analysis dataset](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) and place the CSVs in a `data/` folder.

Run the pipeline in order:
```bash
python data_prep.py
python baseline_markov.py
python train_gru.py
python anomaly_scoring.py
python evaluate.py
python llm_explanation.py
```

Run the demo:
```bash
streamlit run app.py
```

### Gemini API setup

The app uses `gemini-3.1-flash-lite` by default. Create `.streamlit/secrets.toml` with:

```toml
GEMINI_API_KEY = "your-api-key"
```

You can optionally set `GEMINI_MODEL` to another enabled Gemini model. The app also accepts `GEMINI_API_KEY` or `GOOGLE_API_KEY` as environment variables. Without an API key, it uses a valid saved explanation from `data/llm_explanations.pkl` when one is available.


