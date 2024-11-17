# This py file is not a part of the project.
# It is a demo (an example code) of xgboost


import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Assume you have your data loaded as X (features) and y (labels)
# X should be shape (n_samples, n_features) and y should be binary (0 for curve, 1 for power)
# For demonstration purposes, we’ll simulate random data
np.random.seed(0)
X = np.random.rand(100, 75)  # 100 samples, 75 features (25 body points, each with x, y, confidence)
y = np.random.randint(0, 2, 100)  # Binary labels: 0 for curve, 1 for power

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define XGBoost parameters
params = {
    'objective': 'binary:logistic',  # Binary classification
    'eval_metric': 'logloss',        # Logarithmic loss for evaluation
    'learning_rate': 0.1,            # Learning rate
    'max_depth': 5,                  # Max depth of trees
    'subsample': 0.8,                # Row sampling ratio
    'colsample_bytree': 0.8,         # Feature sampling ratio
    'n_estimators': 100              # Number of trees
}

# Initialize the XGBoost classifier with parameters
model = xgb.XGBClassifier(**params)

# Train the model
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)

# Predictions and error evaluation
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate Mean Squared Error (MSE) for both train and test sets
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

# Calculate accuracy for better interpretability in a classification setting
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Output results
print("Training MSE:", train_mse)
print("Test MSE:", test_mse)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)