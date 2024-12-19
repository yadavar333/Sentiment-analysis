# Sentiment Classification on IMDB Movie Reviews

This project focuses on performing sentiment classification on the IMDB movie reviews dataset. It uses a logistic regression model to classify reviews as either positive or negative.

---

## Features

### 1. **Sentiment Classification**
- Classifies movie reviews into two categories:
  - Positive
  - Negative

### 2. **Logistic Regression Model**
- A simple yet effective algorithm for binary classification.
- Interpretable model with probabilities for predictions.

### 3. **Preprocessing Pipeline**
- Efficient preprocessing of text data including:
  - Lowercasing
  - Removal of punctuation and special characters
  - Tokenization
  - Stop-word removal

### 4. **Feature Extraction**
- Converts text data into numerical format using:
  - Term Frequency-Inverse Document Frequency (TF-IDF)

### 5. **Evaluation Metrics**
- Evaluates model performance using:
  - Accuracy
  - Confusion matrix
---

## Dataset

The IMDB dataset contains 50,000 movie reviews:
- **25,000 reviews** for training.
- **25,000 reviews** for testing.
- Balanced dataset with equal numbers of positive and negative reviews.

The dataset is available at: [IMDB Dataset of Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/)

---

## Installation

### Prerequisites
- Python 3.7 or higher
- Required libraries:
  - `scikit-learn`
  - `nltk`
  - `pandas`
  - `numpy`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/sentiment-classification-imdb.git
   ```
2. Navigate to the project directory:
   ```bash
   cd sentiment-classification-imdb
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. **Training the Model**
Run the following command to preprocess data and train the logistic regression model:
```bash
python train.py
```

### 2. **Testing the Model**
To evaluate the trained model on the test dataset, use:
```bash
python test.py
```

### 3. **Making Predictions**
Provide a new movie review as input for sentiment prediction:
```bash
python predict.py --review "This movie was fantastic!"
```

---

## Model Details

### Logistic Regression
- Logistic regression works by modeling the probability of a binary outcome.
- Features are extracted from the preprocessed text data using TF-IDF.
- Regularization techniques are applied to prevent overfitting.

---

## Evaluation Results

| Metric       | Value  |
|--------------|--------|
| Accuracy     | 82%    |

---

## Contributing
We welcome contributions to improve this project! Feel free to:
- Report issues or bugs.
- Suggest new features or enhancements.
- Submit pull requests.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- **Stanford AI Lab** for providing the IMDB dataset.
- **scikit-learn** for its robust machine learning tools.
- **NLTK** for text preprocessing utilities.

---

For any questions or feedback, feel free to contact us or open an issue in the repository.

