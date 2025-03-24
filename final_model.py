import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import mean_squared_error
import xgboost as XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Set random state for reproducibility
RANDOM_STATE = 42

# Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Extract target variable and features
X_train = train_df.drop(['Batch_ID', 'T80', 'Smiles'], axis=1)
y_train = train_df['T80']
X_test = test_df.drop(['Batch_ID', 'T80', 'Smiles'], axis=1)

# Log-transform the target to normalize its distribution
y_train_log = np.log1p(y_train)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection
print("\nPerforming feature selection...")

# 1. Correlation with target
correlation_scores = []
for col in X_train.columns:
    corr = np.abs(np.corrcoef(X_train[col], y_train)[0, 1])
    correlation_scores.append((col, corr))

correlation_features = pd.DataFrame(correlation_scores, columns=['Feature', 'Correlation'])
correlation_features = correlation_features.sort_values('Correlation', ascending=False)
top_corr_features = correlation_features.head(10)['Feature'].tolist()

# 2. F-regression
f_selector = SelectKBest(f_regression, k=10)
f_selector.fit(X_train_scaled, y_train_log)
f_support = f_selector.get_support()
f_selected_features = [X_train.columns[i] for i in range(len(X_train.columns)) if f_support[i]]

# 3. Mutual Information
mi_selector = SelectKBest(mutual_info_regression, k=10)
mi_selector.fit(X_train_scaled, y_train_log)
mi_support = mi_selector.get_support()
mi_selected_features = [X_train.columns[i] for i in range(len(X_train.columns)) if mi_support[i]]

# 4. Random Forest Importance
rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(X_train_scaled, y_train_log)
rf_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
})
rf_importances = rf_importances.sort_values('Importance', ascending=False)
top_rf_features = rf_importances.head(10)['Feature'].tolist()

# Define key molecular features based on previous analysis findings
# These are the top features identified through multiple methods
molecular_features = [
    'TDOS4.0', 'TDOS3.9', 'TDOS3.8', 'Mass', 'NumHeteroatoms', 
    'HOMO(eV)', 'DipoleMoment(Debye)', 'S6', 'SDOS4.0', 'S3', 
    'T3', 'HOMOm1(eV)', 'S11', 'TDOS3.0', 'SDOS4.1'
]

# Create datasets with selected features
X_train_selected = X_train[molecular_features]
X_test_selected = X_test[molecular_features]

# Standardize selected features
X_train_selected_scaled = scaler.fit_transform(X_train_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)

print(f"Selected {len(molecular_features)} features for modeling:")
for i, feature in enumerate(molecular_features):
    print(f"{i+1}. {feature}")

# Define base models for ensemble
base_models = [
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=2, random_state=RANDOM_STATE)),
    ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE)),
    ('xgb', XGBRegressor.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE)),
    ('svr', SVR(kernel='rbf', C=10, gamma='scale')),
    ('lasso', Lasso(alpha=0.01, random_state=RANDOM_STATE)),
    ('ridge', Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ('elastic', ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE))
]

# Create K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Evaluate each base model
print("\nEvaluating base models...")
base_model_scores = {}
base_model_preds = {}

for name, model in base_models:
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_selected_scaled, y_train_log, 
                               cv=kf, scoring='neg_mean_squared_error')
    mse = -np.mean(cv_scores)
    base_model_scores[name] = mse
    
    # Cross-validation predictions
    cv_preds = cross_val_predict(model, X_train_selected_scaled, y_train_log, cv=kf)
    base_model_preds[name] = cv_preds
    
    print(f"{name}: MSE = {mse:.4f}")

# Train each model on the full dataset
print("\nTraining individual models on full dataset...")
trained_models = {}
for name, model in base_models:
    model.fit(X_train_selected_scaled, y_train_log)
    trained_models[name] = model

# Get test predictions from all base models
test_preds = {}
for name, model in trained_models.items():
    test_preds[name] = model.predict(X_test_selected_scaled)

# Weight models inversely proportional to their MSE
total_inverse_mse = sum(1/score for score in base_model_scores.values())
weights = {name: (1/score)/total_inverse_mse for name, score in base_model_scores.items()}

# Create weighted average prediction
weighted_pred_log = np.zeros(X_test_selected_scaled.shape[0])
for name, preds in test_preds.items():
    weighted_pred_log += weights[name] * preds

print("\nModel weights in weighted ensemble:")
for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {weight:.4f}")

# Transform predictions back to original scale
weighted_pred = np.expm1(weighted_pred_log)

# Create and save final submission file
submission = pd.DataFrame({
    'Batch_ID': test_df['Batch_ID'],
    'T80': weighted_pred
})
submission.to_csv('submissionfinal.csv', index=False)

print("\nFinal predictions:")
for i, row in submission.iterrows():
    print(f"{row['Batch_ID']}: {row['T80']:.2f}")

print("\nSaved predictions to 'submissionfinal.csv'")
