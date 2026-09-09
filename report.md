# Anomaly Detection Project Report

## 1. Problem and objective

The project detects unusual behavior in sequential event data. A normal sequence is expected to follow recurring transitions, while an anomaly is a transition or event pattern that differs materially from the learned behavior. The practical objective is to compare a transparent probabilistic baseline with a recurrent neural network and expose the results in a small Streamlit dashboard.

The final application is designed for three tasks: comparing model quality, inspecting prediction outputs, and reading human-friendly explanations for detected anomalies. The training artifacts are kept separate from the dashboard code so that the app can be updated without rebuilding the modeling workflow.

## 2. Approach

Two sequence models are represented. The Markov model estimates the likelihood of the next event from the current state or recent transition history. It is easy to inspect and provides a useful baseline for sparse or short sequences. The GRU model learns a richer representation of ordered events and can capture longer context through its recurrent hidden state. Its anomaly score is compared with a saved threshold, producing the final normal/anomalous decision.

The workflow exports the encoded data mappings, trained GRU weights and configuration, the selected threshold, model predictions, a comparison table, and optional LLM explanations. The dashboard reads these files from `data/` at runtime. Missing artifacts are handled gracefully so the interface remains available while the modeling pipeline is being completed.

## 3. Evaluation

The primary comparison should include accuracy, precision, recall, and F1 score. Accuracy gives an overall view, but it can be misleading when anomalies are rare. Precision measures how many flagged events are truly anomalous, while recall measures how many real anomalies are found. F1 balances these two concerns and is therefore a useful headline metric for this task.

The included dashboard uses demo values only when `model_comparison.csv` is absent. Once the real export is copied into `data/`, the table and chart use those measured values. The GRU threshold is also displayed when `gru_threshold.pkl` is available. This keeps the reported numbers tied to the artifacts produced by the experiment rather than hard-coding them into the UI.

## 4. Limitations and future work

The quality of the detector depends on representative labels, consistent preprocessing, and a threshold selected on validation data. Class imbalance can make accuracy look stronger than the operational performance. Future work should add a confusion matrix, precision-recall curves, threshold tuning, time-based validation, and monitoring for data drift.

The explanations tab is an interpretation aid, not a replacement for the model score. Explanations should be checked against the original event sequence and should avoid exposing sensitive input data. For production use, model loading should be versioned, artifact schemas should be validated, and access to uploaded data should be controlled.

## 5. Conclusion

The project provides a compact path from exported sequence-model artifacts to an inspectable anomaly detection interface. The Markov model establishes a transparent baseline, while the GRU offers additional context capacity. The Streamlit app makes the comparison and investigation workflow accessible without requiring users to open notebooks or understand the training code.
