# phishing-he-detection
Bachelor's thesis implementation — Privacy-Preserving Phishing Email Detection Using ML and Homomorphic Encryption.

# Privacy-Preserving Phishing Email Detection
## BSc Data Science, AI & Digital Business — Bachelor's Thesis
### Gisma University of Applied Sciences, 2026

## Overview
This repository contains the full implementation for the thesis:
"Privacy-Preserving Phishing Email Detection Using Machine Learning 
and Homomorphic Encryption"

The pipeline trains four plaintext ML classifiers on a synthetic 
phishing email dataset and implements encrypted inference using the 
CKKS homomorphic encryption scheme via TenSEAL.

## Requirements
Python 3.11+

Install dependencies:
pip install tenseal scikit-learn pandas numpy

## How to Run
python3 phishing_pipeline.py

## What It Does
1. Generates 18,000 synthetic emails with 30 features
2. Trains Logistic Regression, Linear SVM, Random Forest, 
   Gradient Boosting
3. Runs encrypted inference using CKKS (TenSEAL)
4. Outputs full comparison: plaintext vs encrypted performance

## Key Results
- HE Accuracy: 85.00% (identical to plaintext LR)
- Encrypted inference time: 25.51ms per sample
- Overhead factor: ~32,655x over plaintext
- Ciphertext expansion: ~1,393x

## Thesis
Supervised by Dr. Mehran Monavari
