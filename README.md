# Adult Income Classification - End-to-End ML Project

A complete machine learning solution demonstrating end-to-end ML engineering, from model development to deployment. This project predicts whether an individual's income exceeds $50,000 per year using the Adult Income Classification dataset.

## 🎯 Project Overview

This project implements:
- **Baseline models** using scikit-learn (Logistic Regression, Random Forest)
- **Automated Machine Learning** with Auto-sklearn and TPOT
- **REST API** using FastAPI
- **Web Interface** using Streamlit
- **Deployment** ready for Hugging Face Spaces

## 📋 Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Run the Complete Pipeline

Execute all steps from EDA to model comparison:

```bash
python main.py
```

This will:
1. Perform Exploratory Data Analysis (EDA)
2. Preprocess data and engineer features
3. Train baseline models
4. Train AutoML models
5. Compare models and select the best one

### 2. Start the API Server

```bash
python api.py
```

The API will be available at `http://localhost:8000`

### 3. Launch the Streamlit App

```bash
streamlit run streamlit_app.py
```

Or for Hugging Face Spaces:

```bash
streamlit run app.py
```

The web interface will be available at `http://localhost:8501`

## 📁 Project Structure

```
.
├── main.py                 # Main execution script
├── eda.py                  # Exploratory Data Analysis
├── preprocessing.py        # Data preprocessing and feature engineering
├── baseline_model.py       # Baseline model training
├── automl_model.py         # AutoML model training
├── model_comparison.py     # Model comparison and selection
├── api.py                  # FastAPI REST API
├── streamlit_app.py        # Streamlit web interface
├── app.py                  # Hugging Face Spaces entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── REPORT.md              # Project report
├── data/                  # Data directory
├── models/                # Saved models
└── outputs/               # Results and visualizations
```

## 🔧 Individual Scripts

You can also run each script individually:

```bash
# 1. Exploratory Data Analysis
python eda.py

# 2. Data Preprocessing
python preprocessing.py

# 3. Baseline Models
python baseline_model.py

# 4. AutoML Models
python automl_model.py

# 5. Model Comparison
python model_comparison.py
```

## 🌐 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Make a Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 39,
       "workclass": "State-gov",
       "fnlwgt": 77516,
       "education": "Bachelors",
       "education_num": 13,
       "marital_status": "Never-married",
       "occupation": "Adm-clerical",
       "relationship": "Not-in-family",
       "race": "White",
       "sex": "Male",
       "capital_gain": 2174,
       "capital_loss": 0,
       "hours_per_week": 40,
       "native_country": "United-States"
     }'
```

## 📊 Model Performance

The project compares multiple models:
- **Baseline Models**: Logistic Regression, Random Forest
- **AutoML Models**: Auto-sklearn, TPOT

Performance metrics include:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Training Time

## 🚢 Deployment

### Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Upload all project files
3. Set the Space SDK to Streamlit
4. The app will automatically deploy

### Local Deployment

1. Install dependencies: `pip install -r requirements.txt`
2. Run the pipeline: `python main.py`
3. Start API: `python api.py` (in one terminal)
4. Start Streamlit: `streamlit run streamlit_app.py` (in another terminal)

## 📝 Dataset

The Adult Income Classification dataset is from the UCI Machine Learning Repository:
- **Source**: UCI ML Repository
- **Task**: Binary Classification
- **Target**: Income > $50K or <= $50K
- **Features**: 14 features (age, workclass, education, etc.)

## 🛠️ Technologies Used

- **Python 3.8+**
- **scikit-learn**: Baseline models
- **Auto-sklearn / TPOT**: Automated ML
- **FastAPI**: REST API framework
- **Streamlit**: Web interface
- **Pandas / NumPy**: Data manipulation
- **Matplotlib / Seaborn**: Visualizations

## 📈 Key Features

1. **Comprehensive EDA**: Visualizations and statistical analysis
2. **Feature Engineering**: Derived features and encoding
3. **Model Comparison**: Side-by-side performance evaluation
4. **REST API**: Production-ready API endpoint
5. **Interactive UI**: User-friendly web interface
6. **Feature Importance**: Top 5 most important features displayed

## 📄 License

This project is for educational purposes.

## 👤 Author

ML Engineering Project - Adult Income Classification

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- scikit-learn, Auto-sklearn, and TPOT communities
- FastAPI and Streamlit developers

"# Adult-income-prediction-ML-Project-" 
