import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

def run_regression_modeling(df, feature_cols, target_col, model_type='linear', test_size=0.2, random_state=42):
    """
    Train and evaluate a regression model.
    model_type: 'linear', 'ridge', 'decision_tree', 'random_forest'
    """
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(df[target_col].mean() if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]) else 0)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Initialize model
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'decision_tree':
        model = DecisionTreeRegressor(max_depth=5, random_state=random_state)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state)
    else:
        raise ValueError(f"Unknown regression model type: {model_type}")
        
    # Fit and Predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Feature Importances (if applicable)
    importances = {}
    if hasattr(model, 'coef_'):
        for col, coef in zip(feature_cols, model.coef_):
            importances[col] = float(coef)
    elif hasattr(model, 'feature_importances_'):
        for col, imp in zip(feature_cols, model.feature_importances_):
            importances[col] = float(imp)
            
    # Sorted features
    sorted_importances = dict(sorted(importances.items(), key=lambda item: abs(item[1]), reverse=True))
    
    # Test predictions vs actuals comparison
    results_comparison = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    }).reset_index(drop=True)
    
    return {
        "model": model,
        "metrics": {
            "MAE": mae,
            "RMSE": rmse,
            "R2_Score": r2
        },
        "feature_importances": sorted_importances,
        "comparison_df": results_comparison
    }

def run_classification_modeling(df, feature_cols, target_col, model_type='logistic', test_size=0.2, random_state=42):
    """
    Train and evaluate a classification model.
    model_type: 'logistic', 'decision_tree', 'random_forest'
    """
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)
    
    # Ensure classification labels are integer or string categories
    if pd.api.types.is_numeric_dtype(y):
        # Fill nulls and cast
        y = y.fillna(0).astype(int)
    else:
        y = y.astype(str)
        
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Initialize model
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)
    else:
        raise ValueError(f"Unknown classification model type: {model_type}")
        
    # Fit and Predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate
    # For multiclass, average precision/recall/f1 can be set to 'weighted'
    is_binary = len(np.unique(y)) <= 2
    avg_method = 'binary' if is_binary else 'weighted'
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
    recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
    
    # Feature Importances (if applicable)
    importances = {}
    if hasattr(model, 'coef_'):
        # In multi-class, coef_ is shape (n_classes, n_features). Let's take mean absolute coefficient across classes.
        coef_vals = np.mean(np.abs(model.coef_), axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        for col, coef in zip(feature_cols, coef_vals):
            importances[col] = float(coef)
    elif hasattr(model, 'feature_importances_'):
        for col, imp in zip(feature_cols, model.feature_importances_):
            importances[col] = float(imp)
            
    # Sorted features
    sorted_importances = dict(sorted(importances.items(), key=lambda item: abs(item[1]), reverse=True))
    
    # Test predictions vs actuals comparison
    results_comparison = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    }).reset_index(drop=True)
    
    return {
        "model": model,
        "metrics": {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1
        },
        "feature_importances": sorted_importances,
        "comparison_df": results_comparison
    }
