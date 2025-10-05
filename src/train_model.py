import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import time

def train_bias_model():
    print("Loading full dataset...")
    start_time = time.time()
    
    # Load entire dataset
    df = pd.read_csv('../data/bead_full_dataset.csv')
    print(f"Dataset loaded: {len(df):,} samples")
    
    # Clean data
    df = df.dropna(subset=['text', 'label'])
    
    # Create binary labels and bias scores
    def create_bias_score(label):
        if label == 'Neutral':
            return 0, 0  # Not biased, 0% bias
        elif label == 'Slightly Biased':
            return 1, 30  # Biased, 30% bias
        elif label == 'Highly Biased':
            return 1, 80  # Biased, 80% bias
        else:
            return 0, 0
    
    # Apply the function
    df[['is_biased', 'bias_percentage']] = df['label'].apply(
        lambda x: pd.Series(create_bias_score(x))
    )
    
    # Create simplified labels
    df['simple_label'] = df['is_biased'].map({0: 'Neutral', 1: 'Biased'})
    
    print(f"After cleaning: {len(df):,} samples")
    print(f"Label distribution:")
    print(df['simple_label'].value_counts())
    print(f"Average bias percentage: {df['bias_percentage'].mean():.1f}%")
    
    # Text preprocessing
    print("Creating text features...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95
    )
    
    X = vectorizer.fit_transform(df['text'])
    y_class = df['simple_label']  # For classification
    y_score = df['bias_percentage']  # For bias percentage
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X, y_class, y_score, test_size=0.1, random_state=42, stratify=y_class
    )
    
    # Train classification model (Neutral vs Biased)
    print("Training classification model...")
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    classifier.fit(X_train, y_class_train)
    
    # Train regression model (Bias percentage)
    print("Training bias percentage model...")
    from sklearn.ensemble import RandomForestRegressor
    
    regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    regressor.fit(X_train, y_score_train)
    
    # Evaluate classification
    class_score = classifier.score(X_test, y_class_test)
    print(f"\nClassification accuracy: {class_score:.4f}")
    
    y_class_pred = classifier.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_class_test, y_class_pred))
    
    # Evaluate regression
    from sklearn.metrics import mean_absolute_error, r2_score
    y_score_pred = regressor.predict(X_test)
    mae = mean_absolute_error(y_score_test, y_score_pred)
    r2 = r2_score(y_score_test, y_score_pred)
    
    print(f"\nBias Percentage Model:")
    print(f"Mean Absolute Error: {mae:.2f}%")
    print(f"R² Score: {r2:.4f}")
    
    # Save models
    print("Saving models...")
    with open('../models/bias_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    with open('../models/bias_regressor.pkl', 'wb') as f:
        pickle.dump(regressor, f)
    with open('../models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds")
    print("Models saved as '../models/bias_classifier.pkl', '../models/bias_regressor.pkl' and '../models/vectorizer.pkl'")

if __name__ == "__main__":
    train_bias_model()