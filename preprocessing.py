"""
Data Preprocessing and Feature Engineering for Adult Income Classification
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, filepath='data/adult_processed.csv'):
        """Load the dataset"""
        if Path(filepath).exists():
            df = pd.read_csv(filepath)
        else:
            # Try to load from original source
            columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                       'marital-status', 'occupation', 'relationship', 'race', 
                       'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 
                       'native-country', 'income']
            try:
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
                df = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)
            except:
                # Try local file
                df = pd.read_csv('adult.data', names=columns, na_values=' ?', skipinitialspace=True)
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        df_clean = df.copy()
        
        # For categorical columns, fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        if 'income' in categorical_cols:
            categorical_cols.remove('income')
        
        for col in categorical_cols:
            mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col] = df_clean[col].fillna(mode_value)
        
        # For numerical columns, fill with median
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        for col in numerical_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        return df_clean
    
    def feature_engineering(self, df):
        """Create derived features"""
        # Age groups - use numerical codes instead of text labels
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], 
                                  labels=[0, 1, 2, 3])  # 0=Young, 1=Middle, 2=Senior, 3=Elderly
        
        # Capital features
        df['capital-total'] = df['capital-gain'] - df['capital-loss']
        df['has_capital'] = (df['capital-total'] > 0).astype(int)
        
        # Hours per week groups - use numerical codes
        df['hours_group'] = pd.cut(df['hours-per-week'], bins=[0, 35, 40, 50, 100],
                                    labels=[0, 1, 2, 3])  # 0=Part-time, 1=Full-time, 2=Overtime, 3=Excessive
        
        # Education level grouping - use numerical codes
        education_mapping = {
            'Preschool': 0, '1st-4th': 0, '5th-6th': 0, '7th-8th': 0, '9th': 0,
            '10th': 0, '11th': 0, '12th': 0, 'HS-grad': 1, 'Prof-school': 2,
            'Assoc-acdm': 1, 'Assoc-voc': 1, 'Some-college': 1, 'Bachelors': 2,
            'Masters': 2, 'Doctorate': 2
        }
        df['education_level'] = df['education'].map(education_mapping)
        
        return df
    
    def encode_categorical(self, df, fit=True):
        """Encode categorical variables"""
        # Include both object and category types
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'income' in categorical_cols:
            categorical_cols.remove('income')
        
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    unique_values = set(df[col].astype(str).unique())
                    known_values = set(self.label_encoders[col].classes_)
                    unknown_values = unique_values - known_values
                    
                    if unknown_values:
                        # Replace unknown with most common
                        df[col] = df[col].replace(list(unknown_values), df[col].mode()[0])
                    
                    df_encoded[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df_encoded
    
    def scale_features(self, X, fit=True):
        """Scale numerical features"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def preprocess(self, df, fit=True):
        """Complete preprocessing pipeline"""
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Encode categorical
        df = self.encode_categorical(df, fit=fit)
        
        # Separate features and target
        X = df.drop('income', axis=1)
        y = df['income']
        
        # Encode target
        if fit:
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y)
        else:
            y = self.target_encoder.transform(y)
        
        # Get numerical columns for scaling
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Scale features
        X_scaled = X.copy()
        if len(numerical_cols) > 0:
            X_scaled[numerical_cols] = self.scale_features(X[numerical_cols], fit=fit)
        
        # Store feature names
        if fit:
            self.feature_names = X_scaled.columns.tolist()
        
        return X_scaled, y
    
    def save_preprocessor(self, filepath='models/preprocessor.joblib'):
        """Save the preprocessor"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'target_encoder': self.target_encoder,
            'feature_names': self.feature_names
        }, filepath)
    
    def load_preprocessor(self, filepath='models/preprocessor.joblib'):
        """Load the preprocessor"""
        preprocessor_dict = joblib.load(filepath)
        self.scaler = preprocessor_dict['scaler']
        self.label_encoders = preprocessor_dict['label_encoders']
        self.target_encoder = preprocessor_dict['target_encoder']
        self.feature_names = preprocessor_dict['feature_names']

if __name__ == "__main__":
    # Create directories
    Path('data').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    print("Loading data...")
    df = preprocessor.load_data()
    
    # Preprocess
    print("Preprocessing data...")
    X, y = preprocessor.preprocess(df, fit=True)
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    pd.DataFrame(X_train).to_csv('data/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('data/X_test.csv', index=False)
    pd.Series(y_train).to_csv('data/y_train.csv', index=False)
    pd.Series(y_test).to_csv('data/y_test.csv', index=False)
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    print(f"\nPreprocessing complete!")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {len(preprocessor.feature_names)}")