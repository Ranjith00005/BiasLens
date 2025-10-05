from flask import Flask, render_template, request, jsonify
import pickle
import re
import numpy as np

app = Flask(__name__)

class BiasDetector:
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.regressor = None
        self.load_models()
    
    def load_models(self):
        try:
            with open('../models/bias_classifier.pkl', 'rb') as f:
                self.classifier = pickle.load(f)
            with open('../models/bias_regressor.pkl', 'rb') as f:
                self.regressor = pickle.load(f)
            with open('../models/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("Models loaded successfully!")
        except FileNotFoundError:
            print("ERROR: Model files not found!")
            print("Please run 'python train_model.py' first to train the models.")
            exit(1)
    
    def predict(self, text):
        # Clean text
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Vectorize
        X = self.vectorizer.transform([clean_text])
        
        # Get classification (Neutral vs Biased)
        prediction = self.classifier.predict(X)[0]
        class_probabilities = self.classifier.predict_proba(X)[0]
        
        # Get bias percentage (0-100%)
        bias_percentage = max(0, min(100, self.regressor.predict(X)[0]))
        
        # Get confidence for classification
        confidence = max(class_probabilities)
        
        # Create probability dictionary
        classes = self.classifier.classes_
        prob_dict = {cls: prob for cls, prob in zip(classes, class_probabilities)}
        
        return {
            'prediction': prediction,
            'bias_percentage': round(bias_percentage, 1),
            'confidence': confidence,
            'probabilities': prob_dict
        }
    
    def summarize(self, text):
        # Simple extractive summarization
        sentences = text.split('.')
        if len(sentences) <= 3:
            return text
        
        # Take first and last sentences as summary
        summary = sentences[0] + '. ' + sentences[-2] + '.'
        return summary.strip()

# Initialize detector
detector = BiasDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'error': 'Please provide text to analyze'})
    
    # Get bias prediction
    bias_result = detector.predict(text)
    
    # Get summary
    summary = detector.summarize(text)
    
    # Combine results
    result = {
        'bias': bias_result,
        'summary': summary
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)