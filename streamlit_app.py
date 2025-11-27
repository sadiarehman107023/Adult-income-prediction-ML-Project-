"""
Streamlit Web Interface for Income Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💰",
    layout="wide"
)

# Title
st.title("💰 Adult Income Prediction")
st.markdown("Predict whether an individual's income exceeds $50,000 per year")

# Load model performance metrics
@st.cache_data
def load_model_metrics():
    """Load model performance metrics"""
    try:
        if Path('outputs/full_comparison.csv').exists():
            results = pd.read_csv('outputs/full_comparison.csv', index_col=0)
            return results
        elif Path('outputs/baseline_results.csv').exists():
            results = pd.read_csv('outputs/baseline_results.csv', index_col=0)
            return results
    except:
        pass
    return None

@st.cache_data
def load_feature_importance():
    """Load feature importance"""
    try:
        if Path('outputs/automl_feature_importance.csv').exists():
            return pd.read_csv('outputs/automl_feature_importance.csv')
        elif Path('outputs/baseline_feature_importance.csv').exists():
            return pd.read_csv('outputs/baseline_feature_importance.csv')
    except:
        pass
    return None

# Sidebar for API configuration
st.sidebar.header("⚙️ Configuration")
api_url = st.sidebar.text_input(
    "API URL",
    value="http://localhost:8000",
    help="URL of the FastAPI backend"
)

# Check API connection
try:
    response = requests.get(f"{api_url}/health", timeout=2)
    if response.status_code == 200:
        st.sidebar.success("✅ API Connected")
        health_data = response.json()
        if not health_data.get('model_loaded', False):
            st.sidebar.warning("⚠️ Model not loaded")
    else:
        st.sidebar.error("❌ API Error")
except:
    st.sidebar.error("❌ Cannot connect to API")
    st.sidebar.info("Make sure the FastAPI server is running:\n```bash\npython api.py\n```")

# Main content
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📈 Feature Importance"])

with tab1:
    st.header("Make a Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.number_input("Age", min_value=17, max_value=90, value=35, step=1)
        sex = st.selectbox("Sex", ["Male", "Female"])
        race = st.selectbox("Race", [
            "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
        ])
        native_country = st.selectbox("Native Country", [
            "United-States", "Mexico", "Philippines", "Germany", "Canada",
            "Puerto-Rico", "El-Salvador", "India", "Cuba", "England",
            "Jamaica", "South", "China", "Italy", "Dominican-Republic",
            "Vietnam", "Guatemala", "Japan", "Columbia", "Taiwan",
            "Haiti", "Iran", "Portugal", "Nicaragua", "Peru",
            "France", "Greece", "Ecuador", "Ireland", "Hong",
            "Trinadad&Tobago", "Cambodia", "Thailand", "Laos", "Yugoslavia",
            "Outlying-US(Guam-USVI-etc)", "Scotland", "Poland", "Hungary"
        ])
    
    with col2:
        st.subheader("Work & Education")
        workclass = st.selectbox("Workclass", [
            "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
            "Local-gov", "State-gov", "Without-pay", "Never-worked"
        ])
        education = st.selectbox("Education", [
            "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
            "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th",
            "Masters", "1st-4th", "10th", "Doctorate", "5th-6th",
            "Preschool"
        ])
        education_num = st.number_input("Education Number", min_value=1, max_value=16, value=13, step=1)
        occupation = st.selectbox("Occupation", [
            "Tech-support", "Craft-repair", "Other-service", "Sales",
            "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
            "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
            "Transport-moving", "Priv-house-serv", "Protective-serv",
            "Armed-Forces"
        ])
        hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40, step=1)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Family & Relationships")
        marital_status = st.selectbox("Marital Status", [
            "Married-civ-spouse", "Divorced", "Never-married", "Separated",
            "Widowed", "Married-spouse-absent", "Married-AF-spouse"
        ])
        relationship = st.selectbox("Relationship", [
            "Wife", "Own-child", "Husband", "Not-in-family",
            "Other-relative", "Unmarried"
        ])
    
    with col4:
        st.subheader("Financial")
        capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0, step=100)
        capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0, step=100)
        fnlwgt = st.number_input("Final Weight", min_value=10000, max_value=1500000, value=200000, step=1000)
    
    # Prediction button
    if st.button("🔮 Predict Income", type="primary", use_container_width=True):
        # Prepare request
        request_data = {
            "age": int(age),
            "workclass": workclass,
            "fnlwgt": int(fnlwgt),
            "education": education,
            "education_num": int(education_num),
            "marital_status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "capital_gain": int(capital_gain),
            "capital_loss": int(capital_loss),
            "hours_per_week": int(hours_per_week),
            "native_country": native_country
        }
        
        try:
            # Make API request
            response = requests.post(
                f"{api_url}/predict",
                json=request_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display result
                st.success("✅ Prediction Complete!")
                
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    st.metric(
                        "Predicted Income",
                        result['income_class'],
                        delta=f"{result['probability']*100:.1f}% confidence"
                    )
                
                with col_result2:
                    # Probability gauge
                    prob = result['probability']
                    st.progress(prob)
                    st.caption(f"Probability of income >$50K: {prob*100:.2f}%")
                
                # Interpretation
                if result['income_class'] == ">50K":
                    st.info("🎉 This individual is predicted to have an income exceeding $50,000 per year.")
                else:
                    st.info("📊 This individual is predicted to have an income of $50,000 or less per year.")
                
            else:
                st.error(f"❌ API Error: {response.status_code}\n{response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection Error: {str(e)}\n\nMake sure the API is running at {api_url}")

with tab2:
    st.header("Model Performance Metrics")
    
    metrics_df = load_model_metrics()
    
    if metrics_df is not None:
        # Display metrics table
        st.subheader("Performance Comparison")
        
        # Select metrics to display
        metric_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'train_time']
        available_cols = [col for col in metric_cols if col in metrics_df.columns]
        
        display_df = metrics_df[available_cols].copy()
        display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
        display_df.index.name = 'Model'
        
        st.dataframe(display_df.style.format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1': '{:.4f}',
            'Roc Auc': '{:.4f}',
            'Train Time': '{:.2f}'
        }))
        
        # Best model
        if 'f1' in metrics_df.columns:
            best_model = metrics_df['f1'].idxmax()
            st.success(f"🏆 Best Model (by F1 Score): **{best_model}**")
            st.metric("Best F1 Score", f"{metrics_df.loc[best_model, 'f1']:.4f}")
        
        # Visualizations
        if len(metrics_df) > 0:
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                if 'accuracy' in metrics_df.columns:
                    st.bar_chart(metrics_df['accuracy'])
                    st.caption("Accuracy by Model")
            
            with col_viz2:
                if 'f1' in metrics_df.columns:
                    st.bar_chart(metrics_df['f1'])
                    st.caption("F1 Score by Model")
    else:
        st.warning("⚠️ Model performance metrics not available. Please run model training first.")

with tab3:
    st.header("Feature Importance")
    
    feature_importance_df = load_feature_importance()
    
    if feature_importance_df is not None:
        st.subheader("Top 5 Most Important Features")
        
        top_5 = feature_importance_df.head(5)
        
        # Display as table
        st.dataframe(top_5.style.format({'importance': '{:.4f}'}))
        
        # Display as bar chart
        st.bar_chart(top_5.set_index('feature')['importance'])
        
        # Feature importance insights
        st.subheader("Insights")
        st.info(f"""
        The most important feature is **{top_5.iloc[0]['feature']}** with an importance score of {top_5.iloc[0]['importance']:.4f}.
        
        These top 5 features account for a significant portion of the model's decision-making process.
        """)
    else:
        st.warning("⚠️ Feature importance data not available. Please run model training first.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Adult Income Classification - Machine Learning Project</p>
    <p>Built with FastAPI, Streamlit, and scikit-learn</p>
</div>
""", unsafe_allow_html=True)

