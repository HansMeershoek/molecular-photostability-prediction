# Molecular Photostability Prediction

This repository contains machine learning models to identify important molecular features for predicting photostability from limited experimental data.

## Project Overview

A common challenge in experimental chemistry is trying to infer causality from small datasets. Chemists often synthesize a limited number of molecules (~100), measure important properties, and try to determine which molecular features influence those properties.

This project addresses a specific challenge in predicting photostability lifetimes (T80) of organic molecules using a data-driven machine learning approach. The goal is to:

1. Identify the most informative molecular features that predict photostability
2. Build regression models that can accurately predict T80 values for new molecules
3. Provide insights into the physical/chemical mechanisms behind photostability

## Key Findings

Through comprehensive machine learning analysis, we identified several critical molecular features that determine photostability:

### 1. Electronic Structure Features
- **TDOS3.9, TDOS4.0, TDOS3.8**: Density of States features in the 3.8-4.0 eV range are the strongest predictors
- **SDOS4.0, SDOS4.1**: Singlet density of states in this energy range also show strong predictive power

### 2. Molecular Properties
- **Mass**: Heavier molecules tend to show better photostability
- **NumHeteroatoms**: The number of non-carbon, non-hydrogen atoms affects stability

### 3. Electronic Energies
- **HOMO(eV), HOMOm1(eV)**: Energy levels relate to oxidation potential and stability

### 4. Excited State Properties
- **S3, S6, S11**: Specific singlet excited states influence energy dissipation mechanisms
- **T3**: Triplet excited states affect energy transfer pathways

## Model Architecture

The final solution uses an ensemble approach that combines multiple machine learning algorithms:

1. Random Forest Regressor
2. Gradient Boosting Regressor
3. XGBoost Regressor
4. Support Vector Regression (SVR)
5. Lasso Regression
6. Ridge Regression
7. ElasticNet Regression

The predictions from these models are combined using a weighted average, with weights inversely proportional to each model's cross-validation error.

## Files

- `final_model.py`: The main script that performs feature selection, model training, and prediction generation
- `submissionfinal.csv`: Final predictions for the test set molecules

## Usage

To run the model:

```bash
python final_model.py
```

The script requires the following data files in the same directory:
- `train.csv`: Training data with molecular features and T80 values
- `test.csv`: Test data with molecular features

## Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn

## Scientific Impact

This work demonstrates how machine learning can identify important molecular features from limited data, providing insights that may not have been previously recognized by experts. The findings align with research published in Nature, showing that data-driven approaches can uncover quantum mechanical properties that determine photostability in organic molecules.

The methodology could be valuable for designing more stable organic molecules for applications like solar cells, reducing the need for extensive experimental testing.