"""
CRC Clinical Data Analysis Toolkit
=================================

A Python module for processing colorectal cancer (CRC) clinical data from 
RSUD Abdoel Wahab Sjahranie, Indonesia. Includes tumor volume calculation, 
data anonymization, and machine learning-based staging classification.

Key Features:
- Automated data cleaning and normalization
- Tumor volume calculation from dimensional measurements
- Patient data anonymization via cryptographic hashing
- Random Forest classifier for cancer staging prediction
- Comprehensive data validation checks

Author: Ahmad Ilham
Institution: Universitas Muhammadiyah Semarang
License: MIT
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import hashlib
import warnings

class CRCDataProcessor:
    """
    Main class for processing colorectal cancer clinical data.
    
    Parameters
    ----------
    data_path : str
        Path to clinical data file (TSV/CSV)
    anonymize : bool, optional
        Whether to automatically anonymize sensitive fields (default: True)
    """
    
    def __init__(self, data_path, anonymize=True):
        self.data = self._load_data(data_path)
        self._clean_data()
        
        if anonymize:
            self.anonymize_data()
            
    def _load_data(self, data_path):
        """Load and validate input data"""
        try:
            df = pd.read_csv(data_path, delimiter='\t', na_values=['', 'NA', 'N/A'])
            
            # Validate required columns
            required_cols = ['SEX', 'AGE', 'CEA RESULTS (ng/ml)', 
                            'HISTOPATHOLOGICAL TYPE', 'Length', 
                            'Width', 'Height', 'TYPE']
            if not all(col in df.columns for col in required_cols):
                raise ValueError("Missing required columns in input data")
                
            return df
            
        except Exception as e:
            raise ValueError(f"Data loading failed: {str(e)}")
    
    def _clean_data(self):
        """Clean and preprocess clinical data"""
        # Standardize decimal formats
        numeric_cols = ['CEA RESULTS (ng/ml)', 'Length', 'Width', 'Height']
        for col in numeric_cols:
            self.data[col] = (
                self.data[col]
                .astype(str)
                .str.replace(',', '.')
                .astype(float)
            )
        
        # Handle special CEA values
        self.data['CEA RESULTS (ng/ml)'] = (
            self.data['CEA RESULTS (ng/ml)']
            .replace({'< 0.50': 0.49, '> 200.00': 200.01})
        )
        
        # Clean histopathological types
        self.data['HISTOPATHOLOGICAL TYPE'] = (
            self.data['HISTOPATHOLOGICAL TYPE']
            .str.strip()
            .str.upper()
            .str.replace(r'[^A-Z ]', '', regex=True)
        )
    
    def calculate_tumor_volume(self, inplace=True):
        """
        Calculate tumor volume in mm³ from dimensional measurements.
        
        Volume = Length (mm) × Width (mm) × Height (mm)
        
        Parameters
        ----------
        inplace : bool
            Whether to modify the DataFrame in place (default: True)
            
        Returns
        -------
        pd.Series or None
            Returns Series if inplace=False, None otherwise
        """
        volume = (
            self.data['Length'] * 
            self.data['Width'] * 
            self.data['Height']
        )
        
        if inplace:
            self.data['Tumor_Volume_mm3'] = volume
            return None
        return volume
    
    def anonymize_data(self, columns=None):
        """
        Anonymize sensitive patient data using SHA-256 hashing.
        
        Parameters
        ----------
        columns : list, optional
            Columns to anonymize (default: ['SEX', 'AGE', 'ROOM'])
        """
        if columns is None:
            columns = ['SEX', 'AGE', 'ROOM']
            
        for col in columns:
            if col in self.data.columns:
                self.data[col] = (
                    self.data[col]
                    .astype(str)
                    .apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
                )
    
    def prepare_staging_model(self, test_size=0.2, random_state=42):
        """
        Train Random Forest classifier for cancer staging prediction.
        
        Parameters
        ----------
        test_size : float
            Proportion of data for testing (default: 0.2)
        random_state : int
            Random seed for reproducibility (default: 42)
            
        Returns
        -------
        dict
            Model performance metrics and trained model
        """
        # Feature engineering
        features = pd.DataFrame()
        features['CEA_level'] = self.data['CEA RESULTS (ng/ml)']
        features['Tumor_Volume'] = self.calculate_tumor_volume(inplace=False)
        features['Grade'] = self.data['Grade']
        
        # Encode histopathological types
        le = LabelEncoder()
        features['Histology_Encoded'] = le.fit_transform(
            self.data['HISTOPATHOLOGICAL TYPE']
        )
        
        # Target variable
        target = self.data['TYPE']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': report,
            'feature_importance': dict(zip(
                features.columns,
                model.feature_importances_
            ))
        }

# Example usage
if __name__ == "__main__":
    # Initialize processor with example data
    processor = CRCDataProcessor('sample_data.tsv')
    
    # Calculate tumor volumes
    processor.calculate_tumor_volume()
    
    # Get staging prediction model
    results = processor.prepare_staging_model()
    
    print(f"Model Accuracy: {results['accuracy']:.2f}")
    print("Feature Importance:")
    for feat, imp in results['feature_importance'].items():
        print(f"- {feat}: {imp:.3f}")
