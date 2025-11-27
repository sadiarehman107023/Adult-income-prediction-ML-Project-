"""
Main execution script for the ML pipeline
Run this script to execute the complete pipeline from EDA to model deployment
"""
import subprocess
import sys
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Script {script_name} not found")
        return False

def main():
    """Main execution pipeline"""
    print("="*80)
    print("ADULT INCOME CLASSIFICATION - COMPLETE ML PIPELINE")
    print("="*80)
    
    # Create necessary directories
    Path('data').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    
    steps = [
        ('eda.py', 'Exploratory Data Analysis'),
        ('preprocessing.py', 'Data Preprocessing & Feature Engineering'),
        ('baseline_model.py', 'Baseline Model Training'),
        ('automl_model.py', 'AutoML Model Training'),
        ('model_comparison.py', 'Model Comparison & Selection'),
    ]
    
    success_count = 0
    for script, description in steps:
        if run_script(script, description):
            success_count += 1
        else:
            print(f"\n⚠️  Pipeline stopped at: {description}")
            print("You can continue manually by running the remaining scripts.")
            break
    
    print(f"\n{'='*80}")
    print(f"PIPELINE SUMMARY: {success_count}/{len(steps)} steps completed")
    print(f"{'='*80}")
    
    if success_count == len(steps):
        print("\n✅ All steps completed successfully!")
        print("\nNext steps:")
        print("1. Start the FastAPI server: python api.py")
        print("2. Run the Streamlit app: streamlit run streamlit_app.py")
        print("3. Access the web interface at http://localhost:8501")
    else:
        print("\n⚠️  Some steps failed. Please check the errors above.")

if __name__ == "__main__":
    main()

