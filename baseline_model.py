"""
Baseline Model Training with scikit-learn
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
from pathlib import Path
import time

def load_data():
    """Load preprocessed data"""
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').squeeze()
    y_test = pd.read_csv('data/y_test.csv').squeeze()
    return X_train, X_test, y_train, y_test

def train_baseline_models(X_train, X_test, y_train, y_test):
    """Train baseline models"""
    models = {}
    results = {}
    
    # Logistic Regression
    print("Training Logistic Regression...")
    start_time = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = lr.predict(X_test)
    y_pred_proba = lr.predict_proba(X_test)[:, 1]
    
    results['Logistic Regression'] = {
        'model': lr,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'train_time': train_time
    }
    models['Logistic Regression'] = lr
    
    print(f"  Accuracy: {results['Logistic Regression']['accuracy']:.4f}")
    print(f"  F1 Score: {results['Logistic Regression']['f1']:.4f}")
    print(f"  ROC-AUC: {results['Logistic Regression']['roc_auc']:.4f}")
    print(f"  Training Time: {train_time:.2f}s")
    
    # Random Forest
    print("\nTraining Random Forest...")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    results['Random Forest'] = {
        'model': rf,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'train_time': train_time
    }
    models['Random Forest'] = rf
    
    print(f"  Accuracy: {results['Random Forest']['accuracy']:.4f}")
    print(f"  F1 Score: {results['Random Forest']['f1']:.4f}")
    print(f"  ROC-AUC: {results['Random Forest']['roc_auc']:.4f}")
    print(f"  Training Time: {train_time:.2f}s")
    
    # Feature importance for Random Forest
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return models, results, feature_importance

def save_baseline_model(model, model_name, filepath):
    """Save baseline model"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"\n{model_name} saved to {filepath}")

if __name__ == "__main__":
    print("=" * 80)
    print("BASELINE MODEL TRAINING")
    print("=" * 80)
    
    # Load data
    print("\nLoading preprocessed data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train models
    models, results, feature_importance = train_baseline_models(X_train, X_test, y_train, y_test)
    
    # Print summary
    print("\n" + "=" * 80)
    print("BASELINE MODEL RESULTS SUMMARY")
    print("=" * 80)
    results_df = pd.DataFrame(results).T
    print(results_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'train_time']])
    
    # Save best baseline model (based on F1 score)
    best_model_name = results_df['f1'].idxmax()
    best_model = models[best_model_name]
    save_baseline_model(best_model, best_model_name, f'models/baseline_{best_model_name.lower().replace(" ", "_")}.joblib')
    
    # Save feature importance
    feature_importance.to_csv('outputs/baseline_feature_importance.csv', index=False)
    print(f"\nTop 5 Features:")
    print(feature_importance.head())
    
    # Save results
    results_df.to_csv('outputs/baseline_results.csv')
    print(f"\nResults saved to 'outputs/baseline_results.csv'")

