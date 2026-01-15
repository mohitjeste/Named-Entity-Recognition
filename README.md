# Named Entity Recognition (NER)

A comprehensive machine learning project implementing **Named Entity Recognition** using both deep learning and traditional machine learning approaches on CONLL-formatted data.

## 📋 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approaches](#approaches)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)

## 🎯 Overview

This project develops and compares multiple machine learning models for identifying and classifying named entities (like person names, organizations, locations) in text. The implementation includes both state-of-the-art deep learning models (LSTM) and classical machine learning algorithms, providing insights into the trade-offs between model complexity and performance.

## 📖 Problem Statement

Named Entity Recognition is a fundamental NLP task where the goal is to:
1. Identify spans of text that represent entities
2. Classify each entity into predefined categories (e.g., PERSON, ORG, LOCATION)

This project tackles this using the standard CONLL format for training and evaluation.

## 📊 Dataset

- **Format**: CONLL (Conference on Computational Natural Language Learning)
- **Files**:
  - `train.txt`: Training data with annotated entities
  - `test.txt`: Test data for evaluation
- **Processed Data**: Pre-processed numpy arrays (`.npy`) for efficient training
  - `processed_sents.npy`: Tokenized sentences
  - `processed_tags.npy`: Entity tags
  - `pos.npy`: Part-of-speech tags
  - Deep learning variants (`_dl` suffix) with specialized preprocessing

## 🤖 Approaches

### 1. Deep Learning (LSTM)
- **Model**: Bidirectional LSTM with embedding layer
- **Framework**: TensorFlow/Keras
- **Features**:
  - Captures sequential dependencies in text
  - Leverages pre-trained embeddings
  - Best checkpoint saved: `best-lstm-v8`
- **Notebook**: `lstm-approach.ipynb`

### 2. Traditional Machine Learning
Multiple classical algorithms compared:
- **KNN**: k-Nearest Neighbors classifier
- **Random Forest**: Ensemble decision tree method
- **Naive Bayes**: Probabilistic classifier
- **Perceptron**: Linear classifier
- **SGD**: Stochastic Gradient Descent
- **Notebook**: `traditional-ml.ipynb`

### 3. Ensemble Learning
- Combined approaches for improved performance
- **Notebook**: `ensemble-learning.ipynb`

## 📁 Project Structure

```
Named-Entity-Recognition/
├── README.md                              # This file
├── data/
│   ├── train.txt                         # Training data (CONLL format)
│   ├── test.txt                          # Test data (CONLL format)
│   ├── processed_sents.npy               # Preprocessed sentences
│   ├── processed_tags.npy                # Entity tags
│   ├── pos.npy                           # Part-of-speech tags
│   ├── processed_sents_dl.npy            # DL-specific sentence preprocessing
│   ├── processed_tags_dl.npy             # DL-specific tag preprocessing
│   └── pos_dl.npy                        # DL-specific POS tags
├── models/
│   ├── best-lstm-v8.*                    # Trained LSTM model (TensorFlow)
│   ├── knn.sav                           # Trained KNN model
│   ├── random-forest.sav                 # Trained Random Forest
│   ├── nb.sav                            # Trained Naive Bayes
│   ├── perceptron.sav                    # Trained Perceptron
│   └── sgd.sav                           # Trained SGD classifier
├── scripts/
│   ├── EDA.ipynb                         # Exploratory Data Analysis
│   ├── pre-processing.ipynb              # Data preprocessing pipeline
│   ├── lstm-approach.ipynb               # LSTM model implementation
│   ├── traditional-ml.ipynb              # Traditional ML models
│   ├── knn-randomforest.ipynb            # KNN & Random Forest deep dive
│   ├── ensemble-learning.ipynb           # Ensemble methods
│   └── lstm-testdata-predictions.ipynb   # Inference on test data
└── results/
    └── test_predictions.txt              # Model predictions on test set
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Named-Entity-Recognition
   ```

2. **Create virtual environment** (optional but recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Key dependencies**:
   - TensorFlow/Keras
   - scikit-learn
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - jupyter

## 📖 Usage

### Running Notebooks

1. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

2. **Execute notebooks in order**:
   1. `EDA.ipynb` - Understand data distribution
   2. `pre-processing.ipynb` - Prepare data
   3. `lstm-approach.ipynb` - Train LSTM model
   4. `traditional-ml.ipynb` - Train classical models
   5. `ensemble-learning.ipynb` - Combine models
   6. `lstm-testdata-predictions.ipynb` - Generate predictions

### Loading Pre-trained Models

```python
import tensorflow as tf
from sklearn.externals import joblib

# Load LSTM model
lstm_model = tf.keras.models.load_model('models/best-lstm-v8')

# Load traditional ML models
knn_model = joblib.load('models/knn.sav')
rf_model = joblib.load('models/random-forest.sav')
```

### Making Predictions

```python
# Using pre-trained LSTM
predictions = lstm_model.predict(test_data)

# Using traditional ML
predictions = knn_model.predict(test_features)
```

## 🧠 Models

| Model | Type | Use Case | Pros | Cons |
|-------|------|----------|------|------|
| LSTM | Deep Learning | Sequential data | Captures long-range dependencies | Requires more data, computationally expensive |
| Random Forest | Ensemble | Feature importance | Robust, handles non-linear relationships | Less interpretable |
| KNN | Instance-based | Simple baseline | Fast, interpretable | Sensitive to feature scaling |
| Naive Bayes | Probabilistic | Text classification | Fast, works well with sparse data | Assumes feature independence |
| Perceptron | Linear | Simple separator | Fast, interpretable | Limited to linearly separable data |
| SGD | Gradient-based | Online learning | Efficient, scalable | Requires hyperparameter tuning |

## 📈 Results

Predictions on test data are saved to `results/test_predictions.txt` in CONLL format with entity classifications.

**Evaluation Metrics**:
- Precision, Recall, F1-Score (per entity type)
- Overall accuracy
- Confusion matrices

See individual notebooks for detailed results and visualizations.

## 📝 Notes

- LSTM model significantly outperforms traditional ML approaches on this task
- Consider using pre-trained embeddings (Word2Vec, GloVe, BERT) for improved performance
- Data preprocessing is critical for model performance
- Hyperparameter tuning can further improve results

## 📚 References

- CONLL 2003 Shared Task: https://www.clips.uantwerpen.be/conll2003/
- LSTM for NER: Huang et al. (2015)
- TensorFlow/Keras Documentation: https://tensorflow.org
- Scikit-learn Documentation: https://scikit-learn.org
