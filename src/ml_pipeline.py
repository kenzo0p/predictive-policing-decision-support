"""
ML-Based Crime Risk Prediction Pipeline
========================================
This module implements an interpretable ML pipeline for crime risk categorization.

⚠️ ETHICAL DISCLAIMER:
- This is a DECISION SUPPORT tool, NOT a law enforcement automation system
- Higher reported crime ≠ higher actual crime
- Predictions reflect reporting practices, not actual crime rates
- State-level analysis prevents individual profiling

Dataset Limitation:
- Current data is from 2013 only
- For time-series prediction, multi-year data (2001-2013) is required
- This implementation uses cross-sectional analysis as a demonstration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class CrimeRiskPredictor:
    """
    Interpretable Crime Risk Prediction Model
    
    Risk Categories:
    - LOW: Bottom 33% of crime rates
    - MEDIUM: Middle 33% of crime rates  
    - HIGH: Top 33% of crime rates
    
    Features used:
    - Crime category distributions (not absolute counts)
    - This reduces reporting bias impact
    """
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        os.makedirs(model_dir, exist_ok=True)
        
    def load_and_prepare_data(self, filepath="data/raw/dstrIPC_2013.csv"):
        """Load raw data and prepare features for ML"""
        print("📊 Loading and preparing data...")
        
        # Load raw data with all crime categories
        df = pd.read_csv(filepath)
        
        # Define crime categories for feature engineering
        crime_categories = [
            'MURDER', 'ATTEMPT TO MURDER', 'RAPE', 
            'KIDNAPPING & ABDUCTION', 'DACOITY', 'ROBBERY', 
            'BURGLARY', 'THEFT', 'RIOTS', 'CHEATING',
            'HURT/GREVIOUS HURT', 'DOWRY DEATHS',
            'CRUELTY BY HUSBAND OR HIS RELATIVES'
        ]
        
        # Filter columns that exist in the dataset
        available_cols = [col for col in crime_categories if col in df.columns]
        
        # Prepare features
        features_df = df[['STATE/UT', 'DISTRICT', 'YEAR']].copy()
        
        # Add crime category columns
        for col in available_cols:
            features_df[col] = df[col]
        
        # Add total crimes
        features_df['TOTAL_CRIMES'] = df['TOTAL IPC CRIMES']
        
        # Remove aggregate rows (ZZ TOTAL, etc.)
        features_df = features_df[~features_df['DISTRICT'].str.contains('TOTAL|RLY|G.R.P|CID|STF|BIEO|R.P.O', case=False, na=False)]
        features_df = features_df.reset_index(drop=True)  # Reset index to avoid alignment issues
        
        # Create risk category based on total crimes (tertiles)
        features_df['RISK_CATEGORY'] = pd.qcut(
            features_df['TOTAL_CRIMES'], 
            q=3, 
            labels=['LOW', 'MEDIUM', 'HIGH']
        )
        
        # Create proportional features (reduces reporting bias)
        for col in available_cols:
            features_df[f'{col}_PROP'] = features_df[col] / features_df['TOTAL_CRIMES'].replace(0, 1)
        
        print(f"   ✅ Loaded {len(features_df)} districts")
        print(f"   ✅ Features: {len(available_cols)} crime categories")
        print(f"   ✅ Risk distribution:")
        print(features_df['RISK_CATEGORY'].value_counts().to_string())
        
        self.feature_columns = [f'{col}_PROP' for col in available_cols]
        self.raw_data = features_df
        
        return features_df
    
    def prepare_features(self, df):
        """Extract feature matrix and target variable"""
        X = df[self.feature_columns].fillna(0)
        y = df['RISK_CATEGORY']
        return X, y
    
    def train(self, df, model_type='random_forest'):
        """
        Train the risk prediction model
        
        Models available:
        - 'random_forest': Best for feature importance interpretation
        - 'logistic': Best for probability interpretation
        - 'decision_tree': Best for rule extraction
        """
        print(f"\n🤖 Training {model_type} model...")
        
        X, y = self.prepare_features(df)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data (stratified to maintain class balance)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Select model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(
                max_depth=5,
                random_state=42,
                class_weight='balanced'
            )
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   ✅ Training complete!")
        print(f"   📈 Test Accuracy: {accuracy:.2%}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, self.scaler.transform(X), y_encoded, cv=5)
        print(f"   📊 Cross-validation: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
        
        # Store for later
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        
        return accuracy, cv_scores
    
    def evaluate(self):
        """Generate detailed evaluation report"""
        print("\n📋 DETAILED EVALUATION REPORT")
        print("=" * 50)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            self.y_test, 
            self.y_pred,
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        print("\nConfusion Matrix:")
        print(pd.DataFrame(
            cm,
            index=self.label_encoder.classes_,
            columns=self.label_encoder.classes_
        ))
        
        return cm
    
    def get_feature_importance(self):
        """Extract and visualize feature importance"""
        if not hasattr(self.model, 'feature_importances_'):
            print("⚠️ Feature importance not available for this model type")
            return None
        
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 FEATURE IMPORTANCE (What drives risk predictions)")
        print("=" * 50)
        print(importance.head(10).to_string(index=False))
        
        # Visualize
        plt.figure(figsize=(10, 6))
        top_features = importance.head(10)
        plt.barh(top_features['feature'], top_features['importance'], color='steelblue')
        plt.xlabel('Importance')
        plt.title('Top 10 Features for Crime Risk Prediction')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{self.model_dir}/feature_importance.png', dpi=150)
        plt.close()  # Close instead of show to prevent blocking
        print(f"   📊 Plot saved to {self.model_dir}/feature_importance.png")
        
        return importance
    
    def predict_risk(self, state=None, district=None):
        """Predict risk category for a specific location"""
        if state is None and district is None:
            # Return all predictions
            X, _ = self.prepare_features(self.raw_data)
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            results = self.raw_data[['STATE/UT', 'DISTRICT', 'TOTAL_CRIMES']].copy()
            results['PREDICTED_RISK'] = self.label_encoder.inverse_transform(predictions)
            results['CONFIDENCE'] = np.max(probabilities, axis=1)
            
            return results
        else:
            # Filter and predict
            if state:
                mask = self.raw_data['STATE/UT'].str.upper() == state.upper()
            else:
                mask = pd.Series([True] * len(self.raw_data), index=self.raw_data.index)
            if district:
                mask = mask & (self.raw_data['DISTRICT'].str.upper() == district.upper())
            
            subset = self.raw_data[mask].reset_index(drop=True)
            if len(subset) == 0:
                return None
            
            X = subset[self.feature_columns].fillna(0)
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            results = subset[['STATE/UT', 'DISTRICT', 'TOTAL_CRIMES']].copy()
            results['PREDICTED_RISK'] = self.label_encoder.inverse_transform(predictions)
            results['CONFIDENCE'] = np.max(probabilities, axis=1)
            
            return results
    
    def save_model(self, filename="crime_risk_model"):
        """Save trained model and preprocessing objects"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }, f'{self.model_dir}/{filename}.pkl')
        print(f"✅ Model saved to {self.model_dir}/{filename}.pkl")
    
    def load_model(self, filename="crime_risk_model"):
        """Load trained model"""
        data = joblib.load(f'{self.model_dir}/{filename}.pkl')
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']
        self.feature_columns = data['feature_columns']
        print(f"✅ Model loaded from {self.model_dir}/{filename}.pkl")


def main():
    """Main pipeline execution"""
    print("=" * 60)
    print("🚔 PREDICTIVE POLICING DECISION SUPPORT SYSTEM")
    print("   Crime Risk Prediction Pipeline")
    print("=" * 60)
    
    # Initialize predictor
    predictor = CrimeRiskPredictor()
    
    # Load and prepare data
    df = predictor.load_and_prepare_data()
    
    # Train model
    accuracy, cv_scores = predictor.train(df, model_type='random_forest')
    
    # Evaluate
    predictor.evaluate()
    
    # Feature importance
    importance = predictor.get_feature_importance()
    
    # Save model
    predictor.save_model()
    
    # Sample predictions
    print("\n🔮 SAMPLE PREDICTIONS")
    print("=" * 50)
    
    # Get predictions for a specific state
    results = predictor.predict_risk(state="Maharashtra")
    if results is not None:
        print("\nMaharashtra Districts - Risk Assessment:")
        print(results.head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("⚠️ ETHICAL REMINDER:")
    print("   - These predictions are for DECISION SUPPORT only")
    print("   - They reflect REPORTED crime, not actual crime")
    print("   - State-level aggregation prevents individual profiling")
    print("   - Always consider socioeconomic context")
    print("=" * 60)
    
    return predictor


if __name__ == "__main__":
    predictor = main()
