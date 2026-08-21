# VERITAS Evaluation Methodology & Benchmark Results

This document records the evaluation metrics, validation methodology, and explainability mechanisms for VERITAS.

---

## 1. Verified Validation Benchmark Results

The following metrics are extracted directly from the verified model checkpoint (`models/best_model.pt`) saved at **Epoch 42**:

| Metric | Checkpoint Value | Percentage | Description |
|---|---|---|---|
| **Validation ROC-AUC** | `0.880473` | **88.05%** | Area under the Receiver Operating Characteristic curve |
| **Validation Accuracy** | `0.783821` | **78.38%** | Fraction of correctly classified video windows at threshold $\tau = 0.5$ |
| **Validation F1-Score** | `0.780632` | **78.06%** | Harmonic mean of precision and recall for the positive (`Fake`) class |

*Note: Metrics represent window-level evaluation across the validation split (`Validation/` folder).*

---

## 2. Evaluation Protocol

1. **Window-Level Scoring**: For each validation sample $(ff, ec, nc, tm)$, the model computes:
   $$\hat{p} = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}$$
2. **Thresholding**:
   $$\text{Verdict} = \begin{cases} \text{FAKE}, & \text{if } \hat{p} \ge 0.5 \\ \text{REAL}, & \text{if } \hat{p} < 0.5 \end{cases}$$
3. **Metric Calculation**:
   - `roc_auc_score(y_true, y_prob)` via scikit-learn
   - `f1_score(y_true, y_pred)` via scikit-learn
   - `accuracy_score(y_true, y_pred)` via scikit-learn

---

## 3. Stream Attribution & Interpretability

In addition to scalar classification, VERITAS provides per-stream contribution percentages indicating which biometric region triggered the detection:

### Computation
For the first linear layer in the fusion head $\mathbf{W} \in \mathbb{R}^{512 \times 1280}$, the weights are sliced according to the concatenated feature order:
- $\mathbf{W}_{\text{eye}} = \mathbf{W}[:, 0:256]$
- $\mathbf{W}_{\text{nose}} = \mathbf{W}[:, 256:512]$
- $\mathbf{W}_{\text{face}} = \mathbf{W}[:, 512:1024]$
- $\mathbf{W}_{\text{temporal}} = \mathbf{W}[:, 1024:1280]$

For each stream $s$ with extracted feature matrix $\mathbf{F}_s$:
$$\text{Activation}_s = \text{mean}\left(\left| \mathbf{W}_s \mathbf{F}_s^\top \right|\right)$$
$$\text{Contribution}_s (\%) = 100 \times \frac{\text{Activation}_s}{\sum_{k} \text{Activation}_k}$$

This score is returned in the API response under `streams` and rendered in the web UI breakdown bar charts.
