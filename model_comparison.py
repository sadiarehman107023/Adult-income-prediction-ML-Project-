"""
Model Comparison: Baseline vs AutoML
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') # To suppress warnings during plot generation

def load_results():
    """Load results from baseline and AutoML"""
    # Note: If 'outputs/automl_results.csv' doesn't exist, this will fail. 
    # The output confirms it exists, so we proceed.
    baseline_results = pd.read_csv('outputs/baseline_results.csv', index_col=0)
    automl_results = pd.read_csv('outputs/model_results.csv', index_col=0) # Corrected file name based on your output
    return baseline_results, automl_results

def compare_models():
    """Compare baseline and AutoML models"""
    print("=" * 80)
    print("MODEL COMPARISON: BASELINE vs AUTOML")
    print("=" * 80)
    
    # Load results
    try:
        baseline_results, automl_results = load_results()
    except FileNotFoundError as e:
        print(f"❌ Error loading results: {e}. Ensure 'outputs/baseline_results.csv' and 'outputs/model_results.csv' exist.")
        return
    
    # Combine results
    all_results = pd.concat([baseline_results, automl_results])
    
    print("\nAll Models Performance:")
    print(all_results[['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'train_time']])
    
    # Find best model
    best_model_name = all_results['f1'].idxmax()
    best_model_metrics = all_results.loc[best_model_name]
    
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"{'='*80}")
    print(f"Accuracy:  {best_model_metrics['accuracy']:.4f}")
    print(f"Precision: {best_model_metrics['precision']:.4f}")
    print(f"Recall:    {best_model_metrics['recall']:.4f}")
    print(f"F1 Score:  {best_model_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {best_model_metrics['roc_auc']:.4f}")
    print(f"Train Time: {best_model_metrics['train_time']:.2f}s")
    
    # --- ERROR FIXING LOGIC ---
    
    # 1. Define the possible paths for the best model
    best_model_name_safe = best_model_name.lower().replace(" ", "_")
    
    # Path 1: Generic AutoML path (where GradientBoosting was saved)
    generic_automl_path = Path('models/best_model.joblib')
    
    # Path 2: Baseline/Specific naming convention path
    if 'Auto-sklearn' in best_model_name or 'TPOT' in best_model_name:
        specific_path = Path(f'models/automl_{best_model_name_safe}.joblib')
    else:
        specific_path = Path(f'models/baseline_{best_model_name_safe}.joblib')
    
    model_path = None
    
    # Prioritize the generic path if it exists and the best model is one of the AutoML candidates
    if best_model_name in ['GradientBoosting', 'SVM'] and generic_automl_path.exists():
        model_path = generic_automl_path
    elif specific_path.exists():
        model_path = specific_path
    elif generic_automl_path.exists(): # Fallback check, just in case
        model_path = generic_automl_path
    
    if not model_path:
        print(f"\n❌ Error: Could not find model file for {best_model_name}. Tried: {generic_automl_path} and {specific_path}")
        return # Stop execution if model cannot be loaded
        
    try:
        best_model = joblib.load(model_path)
    except Exception as e:
        print(f"\n❌ Error loading model from {model_path}: {e}")
        return
    
    # --- END ERROR FIXING LOGIC ---
    
    # Save best model with standard name
    joblib.dump(best_model, 'models/best_model.joblib')
    print(f"\nBest model saved to 'models/best_model.joblib'")
    
    # Create comparison visualization
    Path('outputs').mkdir(exist_ok=True)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        all_results[metric].plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_ylabel(metric.upper())
        ax.set_xlabel('Model')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    # Training time comparison
    ax = axes[5]
    all_results['train_time'].plot(kind='bar', ax=ax, color='coral')
    ax.set_title('Training Time Comparison')
    ax.set_ylabel('Time (seconds)')
    ax.set_xlabel('Model')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nComparison visualization saved to 'outputs/model_comparison.png'")
    
    # Save comparison results
    all_results.to_csv('outputs/full_comparison.csv')
    print("Full comparison results saved to 'outputs/full_comparison.csv'")
    
    # Discussion
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    # ... (Rest of the analysis code remains the same)
    baseline_best = baseline_results['f1'].max()
    automl_best = automl_results['f1'].max() if len(automl_results) > 0 else 0
    
    print(f"\nBaseline Best F1: {baseline_best:.4f}")
    if automl_best > 0:
        print(f"AutoML Best F1:   {automl_best:.4f}")
        improvement = ((automl_best - baseline_best) / baseline_best) * 100
        print(f"Improvement:      {improvement:.2f}%")
    
    baseline_avg_time = baseline_results['train_time'].mean()
    automl_avg_time = automl_results['train_time'].mean() if len(automl_results) > 0 else 0
    
    print(f"\nBaseline Avg Training Time: {baseline_avg_time:.2f}s")
    if automl_avg_time > 0:
        print(f"AutoML Avg Training Time:   {automl_avg_time:.2f}s")
        time_ratio = automl_avg_time / baseline_avg_time
        print(f"Time Ratio:                  {time_ratio:.2f}x")
    
    print("\nAdvantages of Baseline Models:")
    print("  - Fast training time")
    print("  - Interpretable (especially Logistic Regression)")
    print("  - Easy to understand and debug")
    print("  - Lower computational requirements")
    
    print("\nAdvantages of AutoML:")
    print("  - Potentially better performance through automated search")
    print("  - Discovers complex feature interactions")
    print("  - Reduces manual hyperparameter tuning")
    print("  - Can find non-obvious model combinations")
    
    print("\nLimitations:")
    print("  - AutoML requires significantly more time and resources")
    print("  - Less interpretable models")
    print("  - May overfit if not properly configured")
    print("  - Higher computational cost")
    
if __name__ == "__main__":
    compare_models()
