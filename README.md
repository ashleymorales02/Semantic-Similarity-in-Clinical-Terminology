# Semantic-Similarity-in-Clinical-Terminology

A Python-based tool designed to quantify the semantic relationship between complex clinical terms and medical concepts.

## 🎯 Overview
This tool uses two different scripts to evaluate how closely related two clinical terms given a decriptive file of clinical terms and concepts.  

## 🚀 Features
- **Semantic Comparison:** Calculates similarity scores between pairs of clinical terms.
- **Weighted Scoring System:** Implemented multiple models to mitigate scoring bias and improve matching precision.

## 🛠 Tech Stack
- **Language:** Python 3.x
- **Libraries:** NumPy, Gensim
- **Logic:** Weighted Vector Similarity / String Matching Algorithms

## 🧠 How It Works
The tool takes two input strings (clinical terms) and performs the following:
1. **Preprocessing:** Cleaning and tokenizing the medical text.
2. **Analysis:** Comparing terms against a reference set or using a vector-based approach.
3. **Scoring:** Applying a weighted coefficient to find similar medical terms and/or concepts.
