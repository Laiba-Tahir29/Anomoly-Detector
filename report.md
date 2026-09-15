# Sequence-Based Anomaly Detection Project

## 1. Executive summary

This project is a proof-of-concept system for detecting unusual behavior in ordered e-commerce activity. It treats a user's shopping history as a sequence of product events and learns which products are likely to follow previous products. A transition receives a low probability when the trained model considers the observed next product unexpected. Low-probability transitions are used as anomaly signals.

The project combines:

- Instacart Market Basket data used as a transaction-sequence proxy.
- Synthetic anomaly injection because the public dataset does not contain reliable fraud labels.
- A first-order Markov Chain baseline.
- A GRU neural sequence model as the main detector.
- Validation-based thresholds for converting scores into anomaly decisions.
- A Streamlit interface for selecting a user, inspecting the sequence, viewing the score graph, and reading an explanation.
- An optional Gemini explanation layer that describes suspicious product transitions in plain English.

The current saved artifacts contain 14,139 user sequences, 49,688 products, 1,081 anomalous sequences, and 13,058 normal sequences.

## 2. Problem statement

E-commerce and financial systems produce ordered events: products purchased, merchants visited, transaction types, or account actions. Normal users tend to show repeated behavioral patterns. An account or session may be considered unusual when its sequence contains transitions that do not fit those patterns.

Real fraud data is difficult to obtain because it is private, rare, and expensive to label. This project therefore tests whether a model trained on normal sequence behavior can identify deviations without depending on a large real-world fraud dataset.

The project asks whether next-item sequence prediction can identify unusual user behavior when the model mainly learns from normal behavior.

This is an anomaly-detection proof of concept, not a production fraud decision system. A flagged sequence means that its behavior is unusual according to the learned sequence model; it does not prove fraud, theft, or malicious activity.

## 3. Dataset and representation

The project uses the Instacart Market Basket Analysis dataset as a proxy for transactional behavior. The original domain is grocery shopping, but its user-item sequence structure is similar to many e-commerce and financial use cases.

The main representation is:

```text
user -> ordered product sequence
```

Each product is converted into an integer ID because the GRU consumes tensors of integer token IDs rather than product names. The saved mappings are:

- `item2idx.pkl`: product ID to model index.
- `idx2item.pkl`: model index back to product ID.
- `products.csv`: product metadata used to display product names.

The current artifacts report 49,688 products and a GRU vocabulary size of 38,852 model tokens. The difference can occur because the model vocabulary is built from products represented in the processed sequences rather than every metadata row.

## 4. Synthetic anomaly creation

The public shopping data does not provide trustworthy fraud labels. The project creates controlled labels by injecting department-aware anomalies into otherwise normal sequences.

The saved split contains:

| Group | Count |
|---|---:|
| Training users | 9,140 |
| Test users | 4,999 |
| Held-out normal users | 3,918 |
| Anomalous users | 1,081 |
| Total labeled sequences | 14,139 |

Every anomalous saved sequence has an `anomaly_position`, which identifies where the injected unusual behavior occurs. Each sequence stores:

- `sequence`: the product sequence.
- `sequence_idx`: the integer-encoded sequence used by the GRU.
- `is_anomalous`: the sequence-level ground-truth label.
- `anomaly_position`: the injected anomaly location when applicable.

These labels provide an evaluation proxy. They should not be described as real confirmed fraud labels.

## 5. Models

### 5.1 Markov Chain baseline

The Markov Chain provides a simple and interpretable baseline. It estimates the probability of a next item from a previous item or transition state. It is useful because it is transparent, computationally lighter, and gives a reference point for judging whether the GRU learns useful additional context.

The saved baseline artifacts include `transition_probs.pkl` and `markov_results.pkl`.

### 5.2 GRU main model

The main model is a Gated Recurrent Unit next-item predictor. A GRU maintains a hidden state that summarizes sequence context. Compared with a first-order Markov model, it can use a longer history when estimating the next product.

The saved configuration is:

| Parameter | Value |
|---|---:|
| Vocabulary size | 38,852 |
| Embedding dimension | 128 |
| Hidden dimension | 256 |
| Maximum sequence length | 20 |

The model architecture is:

```text
integer product IDs
        |
        v
embedding layer: 128 dimensions
        |
        v
GRU layer: 256 hidden units
        |
        v
linear output layer
        |
        v
probability for every possible next product
```

The trained weights are stored in `gru_model.pt`, and the architecture settings are stored in `gru_model_config.pkl`. The app loads the weights on CPU and switches the model to evaluation mode.

## 6. How anomaly scoring works

For a sequence containing products x_1, x_2, ..., x_n, the app scores each observed transition from the previous context to the actual next product.

For step i, the model receives a window of up to 20 earlier product IDs and returns a probability distribution over possible next products. The score for the observed item is:

$$
s_i = P(x_i | x_{i-1}, x_{i-2}, ..., x_{i-k})
$$

where k is limited by the maximum sequence length. The implementation pads shorter windows with zero values before passing them to the GRU.

The app interprets the score as follows:

- High s_i: the observed next product is expected by the model.
- Low s_i: the observed next product is unexpected according to the model.
- s_i < T: the transition is flagged as suspicious, where T is the saved threshold.

For a complete sequence, the current app uses the minimum transition probability as the sequence-level score:

$$
s_{sequence} = \min_i(s_i)
$$

The sequence is flagged when:

$$
s_{sequence} < T
$$

This focuses the decision on the most unusual observed transition.

## 7. Threshold selection

The threshold is selected using validation data rather than final test data. The final saved thresholds are:

| Model | Threshold |
|---|---:|
| Markov Chain baseline | 0.0001 |
| GRU main model | 8.297888764996486e-11 |

The GRU artifact also contains an earlier threshold record with a best validation F1 of approximately 0.2774. The app prefers the threshold in `final_thresholds.pkl` and falls back to `gru_threshold.pkl` if the final file is unavailable.

## 8. Evaluation results

The saved `model_comparison.csv` reports:

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Markov Chain baseline | 0.214 | 0.348 | 0.265 |
| GRU main model | 0.254 | 0.603 | 0.357 |

### Metric interpretation

- **Precision = 0.254 for the GRU:** approximately 25.4% of flagged sequences were anomalous according to the synthetic labels.
- **Recall = 0.603 for the GRU:** the model detected approximately 60.3% of anomalous sequences in the evaluation.
- **F1-score = 0.357 for the GRU:** this balances precision and recall and is higher than the Markov baseline.

The GRU improves on the baseline in all three reported metrics:

- Precision: 0.214 to 0.254.
- Recall: 0.348 to 0.603.
- F1-score: 0.265 to 0.357.

The strongest improvement is recall. The GRU catches more injected anomalies, although its precision remains moderate. In practical use, flagged results would need review rather than automatic rejection.

Accuracy is not included in the saved comparison table. This is reasonable for an imbalanced anomaly problem because accuracy can appear high even when a detector misses many anomalies.

## 9. Streamlit application

The main application is [app.py](app.py). It loads saved artifacts from `data/` and provides an interactive inspection workflow.

### 9.1 Startup and artifact loading

At startup, the app loads labeled sequence data, product mappings, GRU configuration, final thresholds, trained weights, product metadata, and optional saved explanations. Streamlit caching prevents repeated loading during every interaction.

### 9.2 User selection

The user selects a sequence from a dropdown. The app displays the most recent product names as visual chips and the stored synthetic ground-truth label. The ground-truth display is useful for demonstration and evaluation, but would normally be hidden from an end user in production.

### 9.3 Detection action

When **Run Anomaly Detection** is clicked:

1. The sequence is scored transition by transition by the GRU.
2. The minimum next-item probability is calculated.
3. The score is compared with the saved GRU threshold.
4. The app displays either flagged or normal.
5. It shows low-confidence and stable transition counts.
6. It renders the 2D transition graph.
7. If flagged, it generates or loads a plain-English explanation.

### 9.4 Current 2D graph

The former 3D “Sequence constellation” was replaced with a simpler 2D chart titled **Purchase Transition Analysis**.

- X-axis: sequence step.
- Y-axis: GRU next-item probability on a logarithmic scale.
- Green points: expected transitions.
- Red points: suspicious transitions below the threshold.
- Dashed red line: learned anomaly threshold.
- Hover text: sequence step, probability, and suspicious product transition where available.

The graph answers: at which step did the model find the actual next purchase least expected? The result text uses “min next-item probability” instead of “min confidence”.

## 10. LLM explanation layer

For a flagged sequence, the app constructs a prompt containing recent product-to-product transitions and their probabilities. Gemini is asked for two or three concise sentences explaining what looks unusual and which transition is most suspicious.

The prompt tells Gemini to use actual product names and not claim fraud, theft, account takeover, or malicious behavior. An anomaly score indicates unexpected behavior, not intent.

The app calls Gemini's `generateContent` endpoint using `GEMINI_API_KEY` or `GOOGLE_API_KEY`. `GEMINI_MODEL` can configure the model name. Without an API key, the app attempts to use `llm_explanations.pkl`; otherwise it displays a configuration message.

The explanation is an interpretation aid. The numerical score and original sequence remain the primary evidence.

## 11. Project files and artifacts

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard, model loading, scoring, chart, and Gemini integration. |
| `README.md` | Setup instructions and project overview. |
| `report.md` | This detailed technical document. |
| `requirements.txt` | Python dependencies. |
| `products.csv` | Product IDs and names. |
| `labeled_data.pkl` | Encoded sequences and synthetic labels. |
| `item2idx.pkl` / `idx2item.pkl` | Product/index mappings. |
| `train_test_split.pkl` | Dataset group assignments. |
| `transition_probs.pkl` | Markov transition probabilities. |
| `markov_results.pkl` | Markov results. |
| `gru_model.pt` | Trained PyTorch GRU weights. |
| `gru_model_config.pkl` | GRU architecture configuration. |
| `gru_results.pkl` | GRU results. |
| `gru_threshold.pkl` | Earlier GRU threshold information. |
| `final_thresholds.pkl` | Final thresholds used by the app. |
| `model_comparison.csv` | Precision, recall, and F1 comparison. |
| `llm_explanations.pkl` | Optional saved explanations. |

The current workspace contains trained artifacts and the dashboard, but not the original pipeline source files listed in the README, such as `data_prep.py`, `train_gru.py`, or `evaluate.py`. The current workspace can run the dashboard from saved artifacts, but reproducing the entire experiment from raw Instacart CSVs requires restoring those scripts or documenting their exact implementation.

## 12. How to install and run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the dashboard from the project directory:

```bash
streamlit run app.py
```

The app expects the saved files to remain in `data/` and should be started from the project root because it uses relative paths.

For Gemini explanations, configure an API key in `.env` or Streamlit secrets:

```text
GEMINI_API_KEY=your-api-key
```

The API key should not be committed to source control.

## 13. End-to-end workflow

```text
Instacart user-item records
          |
          v
ordered product sequences
          |
          v
integer encoding and sequence preparation
          |
          +----------------------+
          |                      |
          v                      v
Markov Chain baseline       GRU next-item model
          |                      |
          +----------+-----------+
                     v
             transition scores
                     |
                     v
          validation threshold selection
                     |
                     v
       normal or anomalous sequence decision
                     |
                     v
       Streamlit graph and optional explanation
```

## 14. Strengths

1. The sequence-prediction idea is clear and explainable.
2. A transparent Markov baseline is included.
3. The threshold is selected on validation data.
4. Precision, recall, and F1 are reported instead of accuracy alone.
5. The dashboard exposes suspicious steps through a probability graph.
6. The LLM layer translates low-probability transitions into a short explanation.
7. The approach can transfer to transaction types, merchants, amounts, or other event categories.

## 15. Limitations and risks

### Synthetic labels

The labels are injected rather than confirmed real fraud labels. The metrics apply to the chosen synthetic anomaly pattern.

### Moderate precision

The GRU recall is stronger than the baseline, but precision is 0.254. Many flags may be false positives, so the system should support investigation rather than automatically block a customer.

### Dataset mismatch

Instacart behavior is only a proxy for financial or account-security behavior. A production model needs domain-specific events and labels.

### Threshold sensitivity

The result depends on the threshold. Thresholds should be revisited when the data distribution changes.

### Sequence context

The GRU uses a maximum context length of 20. Patterns requiring longer history may not be represented fully.

### Explanation reliability

Gemini explanations can be incomplete or phrased imperfectly. They must remain secondary to the model score and transition data.

### Reproducibility

The current workspace contains output artifacts but not all original training scripts. A fully reproducible experiment should preserve preprocessing, anomaly injection, training, evaluation, random seeds, and package versions.

### Security and privacy

Real transaction data requires access control, secure storage, logging policies, and careful handling of personally identifiable information. API prompts should not expose sensitive information unnecessarily.

## 16. Recommended future improvements

1. Restore or add the complete preprocessing and training scripts.
2. Record random seeds and exact dataset versions.
3. Add confusion matrices and precision-recall curves.
4. Report support counts alongside precision, recall, and F1.
5. Compare several thresholds and show the operational trade-off.
6. Add a direct label for the most suspicious transition in the interface.
7. Add tests for artifact loading, score calculation, thresholding, and graph generation.
8. Validate artifact schemas before starting the app.
9. Add model versioning and data-drift monitoring.
10. Evaluate on real domain-specific data when privacy-approved labels become available.

## 17. Viva and examiner explanation

### One-minute project explanation

> This project detects unusual behavior in ordered shopping sequences. I use Instacart data as a proxy for transaction data and inject controlled anomalies because real fraud labels are not publicly available. A Markov Chain provides a simple baseline, while a GRU learns to predict the next product using sequence context. For each observed transition, the GRU outputs the probability of the actual next product. If the lowest probability in a sequence falls below a threshold selected on validation data, the sequence is flagged. The Streamlit application lets a user inspect the sequence, see suspicious steps in a 2D graph, and read an optional plain-English explanation.

### Graph explanation

> Each point represents a transition between two purchases. The GRU estimates the probability of the actual next item. Points below the learned threshold are shown as suspicious because that transition was less expected according to the model.

### Why use a GRU?

> A Markov Chain mainly uses local transition information. A GRU maintains a hidden representation of the sequence, so it can use longer context when predicting the next item.

### Why inject anomalies?

> The public dataset does not contain dependable fraud labels. Controlled anomaly injection creates an evaluation target while keeping the limitation clear: the results apply to the synthetic anomalies used in this experiment.

### Does a flag prove fraud?

> No. It means only that the sequence contains behavior unusual according to the learned model. It should be reviewed as an investigation signal.

## 18. Conclusion

The project has progressed from sequence-based anomaly modeling to a working artifact-driven dashboard. It contains a transparent baseline, a recurrent neural detector, validation thresholds, saved evaluation results, an interpretable 2D probability graph, and an optional natural-language explanation layer.

The saved results show that the GRU outperforms the Markov baseline on precision, recall, and F1 for the synthetic evaluation set, especially in recall. The main remaining step for a stronger research or production submission is reproducibility: preserve the original preprocessing and training code, document experiment settings, and evaluate the method on representative domain-specific data.
