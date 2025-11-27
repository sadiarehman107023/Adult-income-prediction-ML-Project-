"""
Simplified Automated Machine Learning (No TPOT required)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path
import time
import warnings
import os
warnings.filterwarnings('ignore')

def load_data():
    """Load preprocessed data with proper error handling"""
    try:
        print("Loading data from:", os.path.abspath('data'))
        X_train = pd.read_csv('data/X_train.csv')
        X_test = pd.read_csv('data/X_test.csv')
        y_train = pd.read_csv('data/y_train.csv').squeeze()
        y_test = pd.read_csv('data/y_test.csv').squeeze()
        
        print(f"Data loaded successfully!")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Creating sample data...")
        return create_sample_data()

def create_sample_data():
    """Create sample data if files don't exist"""
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Create features
    X = np.random.randn(n_samples, n_features)
    
    # Create target variable (binary classification)
    y = (
        (X[:, 0] > 0.5) | 
        (X[:, 1] < -0.5) | 
        (X[:, 2] * X[:, 3] > 0.2)
    ).astype(int)
    
    # Add some noise
    y = y ^ (np.random.random(n_samples) > 0.95)
    
    # Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create column names
    feature_columns = [f'feature_{i+1}' for i in range(n_features)]
    
    # Create directories
    Path('data').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    
    # Save the data
    pd.DataFrame(X_train, columns=feature_columns).to_csv('data/X_train.csv', index=False)
    pd.DataFrame(X_test, columns=feature_columns).to_csv('data/X_test.csv', index=False)
    pd.DataFrame(y_train, columns=['target']).to_csv('data/y_train.csv', index=False)
    pd.DataFrame(y_test, columns=['target']).to_csv('data/y_test.csv', index=False)
    
    print("Sample data created successfully!")
    return X_train, X_test, y_train, y_test

def train_multiple_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare their performance"""
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results_dict = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"TRAINING {name}")
        print(f"{'='*50}")
        
        try:
            start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            results = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'train_time': train_time
            }
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            results['cv_f1_mean'] = cv_scores.mean()
            results['cv_f1_std'] = cv_scores.std()
            
            print(f"  Accuracy:  {results['accuracy']:.4f}")
            print(f"  F1 Score:  {results['f1']:.4f}")
            print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
            print(f"  CV F1:     {results['cv_f1_mean']:.4f} (+/- {results['cv_f1_std']*2:.4f})")
            print(f"  Time:      {train_time:.2f}s")
            
            results_dict[name] = results
            trained_models[name] = model
            
        except Exception as e:
            print(f"  Error training {name}: {e}")
    
    return results_dict, trained_models

def save_best_model(results_dict, models_dict):
    """Save the best performing model"""
    if not results_dict:
        print("No models were successfully trained.")
        return None
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_dict).T
    
    # Find best model based on F1 score
    best_model_name = results_df['f1'].idxmax()
    best_model = models_dict[best_model_name]
    best_results = results_dict[best_model_name]
    
    # Create directories
    Path('models').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    
    # Save best model
    model_path = 'models/best_model.joblib'
    joblib.dump(best_model, model_path)
    
    # Save results
    results_df.to_csv('outputs/model_results.csv')
    
    print(f"\n🎉 Best Model: {best_model_name}")
    print(f"   F1 Score: {best_results['f1']:.4f}")
    print(f"   Accuracy: {best_results['accuracy']:.4f}")
    print(f"   Model saved to: {model_path}")
    print(f"   Results saved to: outputs/model_results.csv")
    
    return best_model_name, results_df

if __name__ == "__main__":
    print("=" * 80)
    print("SIMPLIFIED AUTOMATED MACHINE LEARNING")
    print("=" * 80)
    
    try:
        # Load data
        print("\nLoading data...")
        X_train, X_test, y_train, y_test = load_data()
        
        # Train multiple models
        print("\nTraining multiple models...")
        results_dict, models_dict = train_multiple_models(X_train, X_test, y_train, y_test)
        
        # Save best model and results
        best_model_name, results_df = save_best_model(results_dict, models_dict)
        
        # Print summary
        if results_df is not None:
            print("\n" + "=" * 80)
            print("FINAL RESULTS SUMMARY")
            print("=" * 80)
            print(results_df[['accuracy', 'f1', 'roc_auc', 'train_time']].round(4))
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your setup and try again.")