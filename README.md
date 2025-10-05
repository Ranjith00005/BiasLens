# 🔍 BiasLens

An AI-powered web application that detects bias in news articles using machine learning. Built with Random Forest models trained on 3.6M samples from the BEAD dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🚀 Features

- **Real-time Bias Detection**: Instant analysis of news articles
- **Dual-Model System**: 
  - Classification (Neutral vs Biased)
  - Regression (Bias Percentage 0-100%)
- **High Accuracy**: 71.1% accuracy on 3.6M sample dataset
- **Confidence Scoring**: Model certainty metrics
- **Web Interface**: User-friendly Flask web application
- **Production Ready**: 17+ hour trained models

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 71.1% |
| **Biased Precision** | 81% |
| **Neutral Precision** | 66% |
| **Training Dataset** | 3.6M samples (BEAD) |
| **Features** | 15K TF-IDF features |
| **Training Time** | 17+ hours |

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **ML Models**: Scikit-learn Random Forest
- **Text Processing**: TF-IDF Vectorizer (15K features, bigrams)
- **Dataset**: BEAD (Bias and Emotion Analysis Dataset)
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: Pandas, NumPy

## 📁 Project Structure

```
BiasLens/
├── src/                    # Core application
│   ├── app.py             # Flask web server
│   ├── train_model.py     # Model training script
│   ├── load_bead_datasets.py  # Dataset loader
│   └── templates/         # Web UI templates
├── models/                # Trained ML models
│   ├── bias_classifier.pkl   # Classification model
│   ├── bias_regressor.pkl    # Regression model
│   └── vectorizer.pkl        # TF-IDF vectorizer
├── data/                  # Dataset files
├── analysis/              # Jupyter notebooks
├── docs/                  # Documentation
└── requirements.txt       # Dependencies
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/BiasLens.git
cd BiasLens
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
cd src
python3 app.py
```

### 4. Open Browser
Navigate to `http://localhost:8080`

## 📖 Usage

### Web Interface
1. Open the web application
2. Paste your news article text
3. Click "Analyze Bias"
4. Get instant results:
   - **Bias Level**: Neutral or Biased
   - **Bias Percentage**: 0-100% intensity
   - **Confidence**: Model certainty
   - **Probabilities**: Detailed breakdown

### API Usage
```python
import requests

response = requests.post('http://localhost:8080/predict', 
                        json={'text': 'Your news article here'})
result = response.json()
print(result['bias'])
```

## 🎯 Model Details

### Classification Model
- **Algorithm**: Random Forest (200 trees)
- **Purpose**: Neutral vs Biased classification
- **Accuracy**: 71.1%
- **Features**: 15K TF-IDF features with bigrams

### Regression Model
- **Algorithm**: Random Forest Regressor (200 trees)
- **Purpose**: Bias intensity (0-100%)
- **MAE**: 21.84%
- **R² Score**: 0.2489

### Training Configuration
```python
# Classification Model
RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# TF-IDF Vectorizer
TfidfVectorizer(
    max_features=15000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95
)
```

## 📊 Dataset

**BEAD (Bias and Emotion Analysis Dataset)**
- **Size**: 3,674,925 samples
- **Source**: Hugging Face (`shainar/BEAD`)
- **Labels**: Neutral, Slightly Biased, Highly Biased
- **Distribution**: 51.3% Neutral, 48.7% Biased

## 🔧 Development

### Retrain Models
```bash
cd src
python3 train_model.py
```

### Load New Dataset
```bash
cd src
python3 load_bead_datasets.py
```

### Run Analysis
```bash
jupyter notebook analysis/data_visualization.ipynb
```

## 📈 Results Examples

```json
{
  "bias": {
    "prediction": "Biased",
    "bias_percentage": 47.8,
    "confidence": 0.642,
    "probabilities": {
      "Neutral": 0.358,
      "Biased": 0.642
    }
  },
  "summary": "Article summary..."
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BEAD Dataset**: Shainar et al. for the comprehensive bias dataset
- **Hugging Face**: For dataset hosting and tools
- **Scikit-learn**: For robust machine learning algorithms

## 📞 Contact

- **GitHub**: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- **Email**: your.email@example.com

---

⭐ **Star this repository if you found it helpful!**