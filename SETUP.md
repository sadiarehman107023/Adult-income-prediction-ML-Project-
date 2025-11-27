# Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Auto-sklearn requires additional system dependencies on Linux:
```bash
sudo apt-get install build-essential swig
```

On Windows, you may need to install Visual C++ Build Tools.

### 2. Run the Complete Pipeline

Execute the main script to run all steps:

```bash
python main.py
```

This will:
1. Download/load the dataset
2. Perform EDA and save visualizations
3. Preprocess data and engineer features
4. Train baseline models
5. Train AutoML models (this may take 30-60 minutes)
6. Compare models and save the best one

### 3. Start the API Server

In one terminal:

```bash
python api.py
```

The API will be available at `http://localhost:8000`

### 4. Launch the Streamlit App

In another terminal:

```bash
streamlit run streamlit_app.py
```

Or for Hugging Face Spaces:

```bash
streamlit run app.py
```

The web interface will be available at `http://localhost:8501`

## Manual Step-by-Step Execution

If you prefer to run each step individually:

### Step 1: Exploratory Data Analysis

```bash
python eda.py
```

This creates visualizations in the `outputs/` directory.

### Step 2: Data Preprocessing

```bash
python preprocessing.py
```

This creates preprocessed datasets in the `data/` directory.

### Step 3: Baseline Models

```bash
python baseline_model.py
```

This trains Logistic Regression and Random Forest models.

### Step 4: AutoML Models

```bash
python automl_model.py
```

**Note**: This step can take 30-60 minutes depending on your system.

### Step 5: Model Comparison

```bash
python model_comparison.py
```

This compares all models and saves the best one.

## Testing the API

After starting the API server, test it with:

```bash
python test_api.py
```

## Troubleshooting

### Auto-sklearn Installation Issues

If you encounter issues installing Auto-sklearn:

1. **Windows**: Install Visual C++ Build Tools
2. **Linux**: Install build-essential and swig
3. **macOS**: Install Xcode Command Line Tools

Alternative: Skip Auto-sklearn and use only TPOT by modifying `automl_model.py`.

### Dataset Not Found

The script will automatically download the dataset from UCI. If download fails:

1. Manually download from: https://archive.ics.uci.edu/ml/datasets/adult
2. Place `adult.data` in the project root or `data/` directory

### Model Files Not Found

If the API shows "Model not loaded":

1. Ensure you've run the training pipeline (`python main.py`)
2. Check that `models/best_model.joblib` exists
3. Verify `models/preprocessor.joblib` exists

### Port Already in Use

If port 8000 or 8501 is already in use:

1. **API**: Modify the port in `api.py` (last line)
2. **Streamlit**: Use `streamlit run streamlit_app.py --server.port 8502`

## Directory Structure After Setup

```
.
├── data/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── models/
│   ├── preprocessor.joblib
│   ├── best_model.joblib
│   ├── baseline_*.joblib
│   └── automl_*.joblib
└── outputs/
    ├── *.png (visualizations)
    ├── baseline_results.csv
    ├── automl_results.csv
    └── full_comparison.csv
```

## Next Steps

1. Review the EDA visualizations in `outputs/`
2. Check model performance in `outputs/full_comparison.csv`
3. Test predictions via the Streamlit interface
4. Deploy to Hugging Face Spaces (see README.md)

## Support

For issues or questions:
1. Check the error messages carefully
2. Review the logs in the terminal
3. Ensure all dependencies are installed correctly
4. Verify the dataset is available

