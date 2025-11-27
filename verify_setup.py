"""
Verification script to check if the project is set up correctly
"""
import sys
from pathlib import Path

def check_file(filepath, description):
    """Check if a file exists"""
    path = Path(filepath)
    if path.exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (NOT FOUND)")
        return False

def check_directory(dirpath, description):
    """Check if a directory exists"""
    path = Path(dirpath)
    if path.exists():
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"⚠️  {description}: {dirpath} (will be created when needed)")
        return False

def check_import(module_name, description):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError:
        print(f"❌ {description}: {module_name} (NOT INSTALLED)")
        return False

def main():
    print("=" * 80)
    print("PROJECT SETUP VERIFICATION")
    print("=" * 80)
    
    # Check required files
    print("\n📁 Required Files:")
    files_ok = True
    files_ok &= check_file("requirements.txt", "Requirements file")
    files_ok &= check_file("main.py", "Main execution script")
    files_ok &= check_file("eda.py", "EDA script")
    files_ok &= check_file("preprocessing.py", "Preprocessing script")
    files_ok &= check_file("baseline_model.py", "Baseline model script")
    files_ok &= check_file("automl_model.py", "AutoML script")
    files_ok &= check_file("model_comparison.py", "Model comparison script")
    files_ok &= check_file("api.py", "FastAPI script")
    files_ok &= check_file("streamlit_app.py", "Streamlit app")
    files_ok &= check_file("README.md", "README")
    files_ok &= check_file("REPORT.md", "Report")
    
    # Check directories
    print("\n📂 Directories:")
    check_directory("data", "Data directory")
    check_directory("models", "Models directory")
    check_directory("outputs", "Outputs directory")
    
    # Check Python packages
    print("\n📦 Python Packages:")
    packages_ok = True
    packages_ok &= check_import("pandas", "Pandas")
    packages_ok &= check_import("numpy", "NumPy")
    packages_ok &= check_import("sklearn", "scikit-learn")
    packages_ok &= check_import("matplotlib", "Matplotlib")
    packages_ok &= check_import("seaborn", "Seaborn")
    packages_ok &= check_import("fastapi", "FastAPI")
    packages_ok &= check_import("streamlit", "Streamlit")
    packages_ok &= check_import("joblib", "Joblib")
    
    # Optional packages
    print("\n📦 Optional Packages (for AutoML):")
    try:
        import autosklearn
        print("✅ Auto-sklearn: autosklearn")
    except ImportError:
        print("⚠️  Auto-sklearn: autosklearn (OPTIONAL - not installed)")
    
    try:
        import tpot
        print("✅ TPOT: tpot")
    except ImportError:
        print("⚠️  TPOT: tpot (OPTIONAL - not installed)")
    
    # Check if models are trained
    print("\n🤖 Trained Models:")
    model_files = [
        ("models/best_model.joblib", "Best model"),
        ("models/preprocessor.joblib", "Preprocessor"),
        ("models/baseline_logistic_regression.joblib", "Baseline Logistic Regression"),
        ("models/baseline_random_forest.joblib", "Baseline Random Forest"),
    ]
    
    models_exist = False
    for filepath, description in model_files:
        if check_file(filepath, description):
            models_exist = True
    
    if not models_exist:
        print("\n⚠️  No trained models found. Run 'python main.py' to train models.")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if files_ok and packages_ok:
        print("✅ Project structure is correct!")
        print("✅ Required packages are installed!")
        
        if models_exist:
            print("✅ Models are trained and ready!")
            print("\n🚀 You can now:")
            print("   1. Start the API: python api.py")
            print("   2. Run Streamlit: streamlit run streamlit_app.py")
        else:
            print("\n📝 Next steps:")
            print("   1. Run the pipeline: python main.py")
            print("   2. Start the API: python api.py")
            print("   3. Run Streamlit: streamlit run streamlit_app.py")
    else:
        print("❌ Some issues found. Please fix them before proceeding.")
        if not files_ok:
            print("   - Missing required files")
        if not packages_ok:
            print("   - Missing required packages. Run: pip install -r requirements.txt")
    
    print("=" * 80)

if __name__ == "__main__":
    main()

