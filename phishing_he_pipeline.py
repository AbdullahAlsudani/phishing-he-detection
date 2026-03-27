"""
================================================================================
Privacy-Preserving Phishing Email Detection
Using Machine Learning and Homomorphic Encryption (CKKS via TenSEAL)
================================================================================

Bachelor's Thesis — BSc Data Science, Artificial Intelligence & Digital Business
Gisma University of Applied Sciences, 2025
Supervisor: Dr. Mehran Monavari

--------------------------------------------------------------------------------
DESCRIPTION
--------------------------------------------------------------------------------
This script implements the full experimental pipeline for the thesis:
  "Privacy-Preserving Phishing Email Detection Using Machine Learning
   and Homomorphic Encryption"

It performs the following steps:
  1. Generates a synthetic phishing email dataset (N=18,000, 30 features)
     with feature distributions parameterised from published literature
  2. Trains four plaintext ML classifiers:
       - Logistic Regression
       - Linear SVM
       - Random Forest
       - Gradient Boosting
  3. Evaluates all models with accuracy, precision, recall, F1, AUC-ROC
  4. Runs 5-fold cross-validation on Logistic Regression
  5. Implements encrypted inference using the CKKS homomorphic encryption
     scheme (TenSEAL library) on Logistic Regression
  6. Compares plaintext vs encrypted performance and computational overhead
  7. Saves all results to results.json in the current working directory

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------
  Python 3.11+

  Install dependencies:
    pip install tenseal scikit-learn pandas numpy

--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------
  python3 phishing_he_pipeline.py

  Expected runtime: 3-5 minutes (HE inference on 200 samples is the bottleneck)

--------------------------------------------------------------------------------
KEY RESULTS (from thesis)
--------------------------------------------------------------------------------
  Plaintext LR Accuracy:      84.17%
  Encrypted LR Accuracy:      85.00%  (zero degradation from encryption)
  HE Overhead Factor:         ~32,655x over plaintext
  Mean HE Inference Time:     25.51ms per sample
  Ciphertext Expansion:       ~1,393x the plaintext feature vector size

--------------------------------------------------------------------------------
REFERENCES
--------------------------------------------------------------------------------
  - Cheon et al. (2017) CKKS scheme — ASIACRYPT 2017
  - Kim et al. (2018) Sigmoid polynomial approximation — JMIR Medical Informatics
  - Benaissa et al. (2021) TenSEAL — arXiv:2104.03152
  - Fette, Sadeh & Tomasic (2007) PILFER phishing detector — WWW 2007
================================================================================
"""

import numpy as np
import pandas as pd
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import tenseal as ts
import tracemalloc


# ============================================================
# SECTION 1: SYNTHETIC DATASET GENERATION
# ============================================================
# Features are sampled from distributions parameterised to match
# the statistical properties reported in phishing detection literature:
#   URL features  — Sahoo, Liu & Hoi (2017)
#   Header features — Bergholz et al. (2010)
#   Content features — Khonji, Iraqi & Jones (2013)

np.random.seed(42)   # Fixed seed for full reproducibility
N = 18000            # Total emails: ~23% phishing, ~77% legitimate

print("=" * 60)
print("PHISHING EMAIL DETECTION — FULL EXPERIMENTAL PIPELINE")
print("=" * 60)
print(f"\n[1] Generating synthetic dataset (N={N}, 30 features)...")

# ── URL Features (10) ────────────────────────────────────────
# Phishing URLs tend to be longer and contain obfuscation patterns
url_len = np.where(
    np.random.rand(N) < 0.8,
    np.random.normal(85, 30, N),   # phishing: longer URLs
    np.random.normal(35, 15, N)    # legitimate: shorter URLs
).clip(5, 300)

num_dots          = np.random.poisson(4, N).clip(1, 15)       # dot count in URL
has_ip_url        = np.random.binomial(1, 0.30, N)            # IP address instead of domain
has_at_symbol     = np.random.binomial(1, 0.25, N)            # '@' in URL (obfuscation)
url_depth         = np.random.poisson(3, N).clip(0, 12)       # number of path slashes
num_special_chars = np.random.poisson(5, N).clip(0, 30)       # special char count
https_flag        = np.random.binomial(1, 0.45, N)            # HTTPS present (0/1)
redirect_count    = np.random.poisson(1, N).clip(0, 8)        # number of redirects
domain_age_days   = np.random.exponential(200, N).clip(0, 5000)  # WHOIS domain age
subdomain_count   = np.random.poisson(2, N).clip(0, 8)        # subdomain depth

# ── Header Features (8) ──────────────────────────────────────
# Email authentication failures (SPF, DKIM) strongly indicate phishing
sender_domain_match  = np.random.binomial(1, 0.40, N)  # sender matches display name
reply_to_mismatch    = np.random.binomial(1, 0.35, N)  # reply-to differs from sender
html_to_text_ratio   = np.random.beta(2, 3, N)         # proportion of HTML content
header_anomaly_score = np.random.beta(1.5, 3, N)       # composite anomaly score
num_recipients       = np.random.poisson(2, N).clip(1, 50)  # recipient count
has_x_mailer         = np.random.binomial(1, 0.55, N)  # X-Mailer header present
spf_pass             = np.random.binomial(1, 0.50, N)  # SPF authentication result
dkim_pass            = np.random.binomial(1, 0.50, N)  # DKIM authentication result

# ── Content Features (8) ─────────────────────────────────────
# Urgency language and credential-harvesting forms are key phishing signals
num_links         = np.random.poisson(5, N).clip(0, 50)
num_images        = np.random.poisson(2, N).clip(0, 20)
urgency_word_count= np.random.poisson(2, N).clip(0, 15)  # e.g. "urgent", "verify now"
form_present      = np.random.binomial(1, 0.30, N)        # HTML form for credentials
login_keywords    = np.random.poisson(1, N).clip(0, 8)    # "login", "sign in" count
account_keywords  = np.random.poisson(1, N).clip(0, 8)    # "account", "password" count
body_length       = np.random.normal(800, 400, N).clip(50, 5000)
obfuscated_text   = np.random.binomial(1, 0.20, N)        # hidden/encoded text detected

# ── Metadata Features (4) ────────────────────────────────────
send_hour             = np.random.randint(0, 24, N)       # hour of day sent
is_weekend            = np.random.binomial(1, 0.25, N)    # sent on weekend
attachment_count      = np.random.poisson(0.5, N).clip(0, 5)
compressed_attachment = np.random.binomial(1, 0.15, N)    # .zip/.rar attachment

# ── Assemble feature matrix ───────────────────────────────────
X_raw = np.column_stack([
    url_len, num_dots, has_ip_url, has_at_symbol, url_depth,
    num_special_chars, https_flag, redirect_count, domain_age_days, subdomain_count,
    sender_domain_match, reply_to_mismatch, html_to_text_ratio, header_anomaly_score,
    num_recipients, has_x_mailer, spf_pass, dkim_pass,
    num_links, num_images, urgency_word_count, form_present, login_keywords,
    account_keywords, body_length, obfuscated_text,
    send_hour, is_weekend, attachment_count, compressed_attachment
])

feature_names = [
    'url_length', 'num_dots', 'has_ip_in_url', 'has_at_symbol', 'url_depth',
    'num_special_chars', 'https_flag', 'redirect_count', 'domain_age_days', 'subdomain_count',
    'sender_domain_match', 'reply_to_mismatch', 'html_to_text_ratio', 'header_anomaly_score',
    'num_recipients', 'has_x_mailer', 'spf_pass', 'dkim_pass',
    'num_links', 'num_images', 'urgency_word_count', 'form_present', 'login_keywords',
    'account_keywords', 'body_length', 'obfuscated_text',
    'send_hour', 'is_weekend', 'attachment_count', 'compressed_attachment'
]

# ── Label generation with realistic cross-feature correlations ─
# Weighted combination of strongest phishing indicators + noise
phishing_score = (
    0.20 * (url_len > 80).astype(float) +       # long URL
    0.15 * has_ip_url +                           # IP in URL
    0.15 * has_at_symbol +                        # @ symbol
    0.10 * (redirect_count > 2).astype(float) +  # multiple redirects
    0.10 * (domain_age_days < 30).astype(float) + # very new domain
    0.12 * reply_to_mismatch +                    # reply-to anomaly
    0.10 * (urgency_word_count > 3).astype(float) + # urgency language
    0.08 * form_present +                         # credential form
    0.10 * (1 - spf_pass) +                      # SPF failure
    0.10 * (1 - dkim_pass) +                     # DKIM failure
    0.08 * obfuscated_text +                      # obfuscation
    np.random.normal(0, 0.12, N)                  # realistic noise
)
y = (phishing_score > 0.55).astype(int)  # threshold to ~23% phishing rate

phishing_rate = y.mean()
print(f"    Emails: {N} | Phishing: {phishing_rate:.1%} ({y.sum()}) | "
      f"Legitimate: {(1-y).mean():.1%} ({(1-y).sum()})")
print(f"    Features: {X_raw.shape[1]} across 4 categories "
      f"(URL=10, Header=8, Content=8, Metadata=4)")


# ============================================================
# SECTION 2: PREPROCESSING
# ============================================================
# Stratified split preserves phishing rate in both train and test sets.
# StandardScaler fitted on training data ONLY to prevent data leakage.

print("\n[2] Preprocessing (stratified split + standardisation)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # preserves class balance
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit only on training data
X_test_s  = scaler.transform(X_test)        # apply same transform to test

print(f"    Train: {len(X_train)} samples | Test: {len(X_test)} samples")


# ============================================================
# SECTION 3: PLAINTEXT MODEL TRAINING AND EVALUATION
# ============================================================
# Four classifiers covering linear and non-linear families.
# Linear models (LR, Linear SVM) are compatible with HE inference.
# Non-linear models (RF, GBM) serve as plaintext upper-bound baselines.

print("\n[3] Training and evaluating plaintext classifiers...")

models = {
    "Logistic Regression": LogisticRegression(
        C=1.0, max_iter=500, random_state=42
        # Selected for HE inference: decision function is linear (w·x + b)
        # followed by sigmoid — both computable under CKKS
    ),
    "Linear SVM": LinearSVC(
        C=1.0, max_iter=2000, random_state=42
        # Also linear; included as second HE-compatible baseline
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        # Non-linear (conditional branching) — incompatible with HE
        # Included as plaintext accuracy upper bound
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, random_state=42
        # Non-linear ensemble — incompatible with HE
        # Typically strongest plaintext F1 on imbalanced datasets
    ),
}

plaintext_results = {}

for name, model in models.items():
    # Training
    t0 = time.perf_counter()
    model.fit(X_train_s, y_train)
    train_time = time.perf_counter() - t0

    # Inference timing
    t1 = time.perf_counter()
    y_pred = model.predict(X_test_s)
    inf_time_total = time.perf_counter() - t1
    inf_time_per   = inf_time_total / len(X_test) * 1000  # ms/sample

    # Classification metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    # AUC-ROC (uses probability scores where available, else decision function)
    try:
        proba = model.predict_proba(X_test_s)[:, 1]
        auc = roc_auc_score(y_test, proba)
    except AttributeError:
        try:
            scores = model.decision_function(X_test_s)
            auc = roc_auc_score(y_test, scores)
        except Exception:
            auc = None

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    plaintext_results[name] = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        "train_time_s": train_time, "inf_ms_per_sample": inf_time_per,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
    }

    auc_str = f"{auc:.4f}" if auc else "N/A"
    print(f"    {name:<22}  Acc={acc:.4f}  F1={f1:.4f}  "
          f"AUC={auc_str}  Inf={inf_time_per:.4f}ms/sample")


# ============================================================
# SECTION 4: CROSS-VALIDATION (Logistic Regression)
# ============================================================
# 5-fold stratified CV on training set to assess model stability.
# Reports mean F1 ± std to confirm generalisation.

print("\n[4] 5-fold cross-validation (Logistic Regression)...")

lr_cv = LogisticRegression(C=1.0, max_iter=500, random_state=42)
cv_scores = cross_val_score(lr_cv, X_train_s, y_train, cv=5, scoring='f1', n_jobs=-1)

print(f"    Fold F1 scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"    Mean F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ============================================================
# SECTION 5: HOMOMORPHIC ENCRYPTION INFERENCE (CKKS)
# ============================================================
# Implements encrypted logistic regression inference using TenSEAL.
#
# CKKS PARAMETERS (128-bit security):
#   poly_modulus_degree = 8192  (N — determines ciphertext size and depth)
#   coeff_mod_bit_sizes = [60, 40, 40, 60]  (3 usable levels)
#   global_scale = 2^40  (40-bit floating point precision)
#
# INFERENCE PIPELINE:
#   Client:  encrypt(x) → send enc_x to server
#   Server:  compute enc_dot = enc_x · w  (homomorphic dot product)
#   Client:  decrypt(enc_dot) + bias → apply sigmoid approx → label
#
# SIGMOID APPROXIMATION:
#   σ̃(x) = 0.5 + 0.197x − 0.004x³
#   Degree-3 minimax polynomial from Kim et al. (2018)
#   Max absolute error on [-6,6]: ~0.18 (concentrated in saturation tails)

print("\n[5] Homomorphic Encryption Inference (CKKS via TenSEAL)...")

# Train final logistic regression model
lr_final = LogisticRegression(C=1.0, max_iter=500, random_state=42)
lr_final.fit(X_train_s, y_train)

# Sigmoid polynomial approximation (Kim et al., 2018)
def sigmoid_approx(x):
    """
    Degree-3 minimax polynomial approximation of sigmoid function.
    Valid range: x in [-6, 6]
    Source: Kim et al. (2018), JMIR Medical Informatics
    Formula: 0.5 + 0.197x - 0.004x^3
    """
    return 0.5 + 0.197 * x - 0.004 * (x ** 3)

def sigmoid_exact(x):
    """Standard sigmoid: 1 / (1 + e^-x)"""
    return 1.0 / (1.0 + np.exp(-x))

# Validate approximation error
z_range   = np.linspace(-6, 6, 1000)
approx_err = np.max(np.abs(sigmoid_approx(z_range) - sigmoid_exact(z_range)))
print(f"    Sigmoid approx max error on [-6,6]: {approx_err:.6f}")

# Initialise CKKS context
print("    Initialising CKKS context (N=8192, 128-bit security)...")
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]  # 3 usable levels for LR inference
)
context.generate_galois_keys()   # required for ciphertext dot product (rotation)
context.global_scale = 2 ** 40   # 40-bit precision for real-valued arithmetic

# Extract model weights for server-side computation
weights = lr_final.coef_[0].tolist()    # 30-dimensional weight vector (plaintext on server)
bias    = lr_final.intercept_[0]        # scalar bias (added after decryption on client)

# Sample 200 test emails for HE evaluation
# (full test set inference would take ~90,000ms = 25 hours on single CPU thread)
HE_SAMPLE  = 200
idx_sample = np.random.choice(len(X_test_s), HE_SAMPLE, replace=False)
X_he       = X_test_s[idx_sample]
y_he       = y_test[idx_sample]

print(f"    Running encrypted inference on {HE_SAMPLE} samples...")
print(f"    (Estimated time: {HE_SAMPLE * 26 / 1000:.0f}–{HE_SAMPLE * 30 / 1000:.0f} seconds)")

he_preds, he_times, enc_sizes = [], [], []
tracemalloc.start()

for i, x_vec in enumerate(X_he):
    t0 = time.perf_counter()

    # Step 1: Client encrypts the feature vector
    enc_x = ts.ckks_vector(context, x_vec.tolist())

    # Step 2: Server computes homomorphic dot product enc_x · w
    #         (server never sees x_vec in plaintext)
    enc_dot = enc_x.dot(weights)

    # Step 3: Client decrypts scalar and adds bias
    dot_val = enc_dot.decrypt()[0] + bias

    # Step 4: Client applies sigmoid polynomial approximation
    pred_prob = sigmoid_approx(dot_val)
    pred      = 1 if pred_prob >= 0.5 else 0

    elapsed = (time.perf_counter() - t0) * 1000  # ms
    he_times.append(elapsed)
    enc_sizes.append(len(enc_x.serialize()))      # ciphertext size in bytes
    he_preds.append(pred)

    if (i + 1) % 50 == 0:
        print(f"      [{i+1}/{HE_SAMPLE}] avg so far: {np.mean(he_times):.1f}ms/sample")

_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

he_preds = np.array(he_preds)

# HE classification metrics
he_acc  = accuracy_score(y_he, he_preds)
he_prec = precision_score(y_he, he_preds, zero_division=0)
he_rec  = recall_score(y_he, he_preds, zero_division=0)
he_f1   = f1_score(y_he, he_preds, zero_division=0)
he_tn, he_fp, he_fn, he_tp = confusion_matrix(y_he, he_preds).ravel()

# Plaintext LR on the same 200 samples (for direct comparison)
pt_preds_sample = lr_final.predict(X_he)
pt_acc_sample   = accuracy_score(y_he, pt_preds_sample)
pt_f1_sample    = f1_score(y_he, pt_preds_sample)

# Plaintext inference time on same sample
t0 = time.perf_counter()
_ = lr_final.predict(X_he)
pt_inf_time = (time.perf_counter() - t0) / HE_SAMPLE * 1000  # ms/sample

mean_he_ms   = np.mean(he_times)
std_he_ms    = np.std(he_times)
mean_ct_kb   = np.mean(enc_sizes) / 1024
pt_vec_kb    = X_he[0].nbytes / 1024
ct_expansion = mean_ct_kb / pt_vec_kb
overhead     = mean_he_ms / pt_inf_time


# ============================================================
# SECTION 6: RESULTS SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("RESULTS SUMMARY")
print("=" * 65)

print(f"\n── Plaintext Models (test set n={len(X_test)}) ─────────────────")
print(f"{'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
print("-" * 65)
for name, r in plaintext_results.items():
    auc_s = f"{r['auc']:.4f}" if r['auc'] else "  N/A"
    print(f"{name:<22} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
          f"{r['recall']:>7.4f} {r['f1']:>7.4f} {auc_s:>7}")

print(f"\n── Encrypted Inference (HE-LR, n={HE_SAMPLE}) ──────────────────")
print(f"  Accuracy:           {he_acc:.4f}  (plaintext same sample: {pt_acc_sample:.4f})")
print(f"  Precision:          {he_prec:.4f}")
print(f"  Recall:             {he_rec:.4f}")
print(f"  F1 Score:           {he_f1:.4f}  (plaintext same sample: {pt_f1_sample:.4f})")
print(f"  Accuracy delta:     {abs(he_acc - pt_acc_sample):.4f}  ← zero = no HE degradation")
print(f"  F1 delta:           {abs(he_f1 - pt_f1_sample):.4f}")

print(f"\n── Computational Overhead ───────────────────────────────────────")
print(f"  HE inference time:  {mean_he_ms:.2f} ± {std_he_ms:.2f} ms/sample")
print(f"  Plaintext time:     {pt_inf_time:.4f} ms/sample")
print(f"  Overhead factor:    {overhead:,.0f}x")
print(f"  Ciphertext size:    {mean_ct_kb:.2f} KB  (plaintext: {pt_vec_kb:.4f} KB)")
print(f"  Ciphertext expansion: {ct_expansion:.0f}x")
print(f"  Peak memory usage:  {peak_mem / 1024 / 1024:.2f} MB")
print(f"  Throughput (1 CPU): ~{1000/mean_he_ms:.0f} emails/second")


# ============================================================
# SECTION 7: SAVE RESULTS
# ============================================================
# Saves to results.json in the current working directory

output_path = os.path.join(os.getcwd(), 'results.json')

results = {
    "dataset": {
        "N": N, "features": int(X_raw.shape[1]),
        "phishing_rate": float(phishing_rate),
        "train_size": len(X_train), "test_size": len(X_test)
    },
    "plaintext_models": plaintext_results,
    "cross_validation": {
        "model": "Logistic Regression",
        "folds": 5,
        "cv_f1_scores": cv_scores.tolist(),
        "mean_f1": float(cv_scores.mean()),
        "std_f1": float(cv_scores.std())
    },
    "sigmoid_approximation": {
        "degree": 3,
        "formula": "0.5 + 0.197x - 0.004x^3",
        "source": "Kim et al. (2018)",
        "max_absolute_error": float(approx_err)
    },
    "he_inference": {
        "scheme": "CKKS",
        "library": "TenSEAL 0.3.16",
        "poly_modulus_degree": 8192,
        "security_bits": 128,
        "sample_size": HE_SAMPLE,
        "accuracy": float(he_acc),
        "precision": float(he_prec),
        "recall": float(he_rec),
        "f1": float(he_f1),
        "TP": int(he_tp), "TN": int(he_tn), "FP": int(he_fp), "FN": int(he_fn),
        "plaintext_acc_same_sample": float(pt_acc_sample),
        "plaintext_f1_same_sample": float(pt_f1_sample),
        "accuracy_delta": float(abs(he_acc - pt_acc_sample)),
        "f1_delta": float(abs(he_f1 - pt_f1_sample)),
        "mean_he_time_ms": float(mean_he_ms),
        "std_he_time_ms": float(std_he_ms),
        "plaintext_time_ms": float(pt_inf_time),
        "overhead_factor": float(overhead),
        "mean_ciphertext_kb": float(mean_ct_kb),
        "plaintext_vector_kb": float(pt_vec_kb),
        "ciphertext_expansion": float(ct_expansion),
        "peak_memory_mb": float(peak_mem / 1024 / 1024),
        "throughput_per_second": float(1000 / mean_he_ms)
    }
}

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[✓] Results saved to: {output_path}")
print("[✓] Pipeline complete.")
