"""
Exploratory Data Analysis for Adult Income Classification Dataset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data():
    """Load the Adult Income dataset"""
    # Try to load from local file first, otherwise download
    try:
        # Common column names for Adult dataset
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                   'marital-status', 'occupation', 'relationship', 'race', 
                   'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 
                   'native-country', 'income']
        
        # Try loading from common paths
        data_paths = [
            'adult.data',
            'adult.csv',
            'data/adult.data',
            'data/adult.csv'
        ]
        
        for path in data_paths:
            if Path(path).exists():
                df = pd.read_csv(path, names=columns, na_values=' ?', skipinitialspace=True)
                return df
        
        # If not found, try to download from UCI
        print("Local file not found. Downloading from UCI...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        df = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create a sample dataset structure for demonstration
        print("Creating sample data structure...")
        return None

def perform_eda(df):
    """Perform comprehensive EDA"""
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # Basic info
    print("\n1. Dataset Shape:")
    print(f"   Rows: {df.shape[0]:,}")
    print(f"   Columns: {df.shape[1]}")
    
    print("\n2. Column Names and Types:")
    print(df.dtypes)
    
    print("\n3. Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    print("\n4. Dataset Info:")
    print(df.info())
    
    print("\n5. First Few Rows:")
    print(df.head())
    
    print("\n6. Statistical Summary:")
    print(df.describe())
    
    # Target variable analysis
    print("\n7. Target Variable Distribution:")
    target_counts = df['income'].value_counts()
    target_pct = df['income'].value_counts(normalize=True) * 100
    print(pd.DataFrame({
        'Count': target_counts,
        'Percentage': target_pct
    }))
    
    # Visualizations
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Target distribution
    plt.figure(figsize=(8, 6))
    df['income'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Target Variable Distribution (Income)')
    plt.xlabel('Income')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Age distribution
    plt.figure(figsize=(10, 6))
    df['age'].hist(bins=30, edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_dir / 'age_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'income' in categorical_cols:
        categorical_cols.remove('income')
    
    # Plot top categories for each categorical feature
    for col in categorical_cols[:6]:  # Limit to first 6 to avoid too many plots
        plt.figure(figsize=(12, 6))
        top_values = df[col].value_counts().head(10)
        top_values.plot(kind='bar')
        plt.title(f'Top 10 Values in {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / f'{col}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Correlation matrix for numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) > 0:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Income by categorical features
    for col in categorical_cols[:4]:  # Limit to first 4
        plt.figure(figsize=(12, 6))
        income_by_cat = pd.crosstab(df[col], df['income'], normalize='index') * 100
        income_by_cat.plot(kind='bar', stacked=False)
        plt.title(f'Income Distribution by {col}')
        plt.xlabel(col)
        plt.ylabel('Percentage')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Income')
        plt.tight_layout()
        plt.savefig(output_dir / f'income_by_{col}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\n8. Key Insights:")
    print(f"   - Class balance: {target_pct.iloc[0]:.2f}% vs {target_pct.iloc[1]:.2f}%")
    print(f"   - Missing values in: {', '.join(missing_df[missing_df['Missing Count'] > 0].index.tolist())}")
    print(f"   - Numerical features: {len(numerical_cols)}")
    print(f"   - Categorical features: {len(categorical_cols)}")
    
    print("\nEDA complete! Visualizations saved to 'outputs/' directory.")
    print("=" * 80)
    
    return df

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        df = perform_eda(df)
        # Save processed data for next steps
        df.to_csv('data/adult_processed.csv', index=False)
        print("\nData saved to 'data/adult_processed.csv'")
    else:
        print("Could not load data. Please ensure the dataset is available.")

