# 🎬 IMDB Sentiment Analysis using NLP and Logistic Regression

This project performs **sentiment analysis** on IMDB movie reviews using **Natural Language Processing (NLP)** techniques and a **Logistic Regression** classifier. The model learns to classify reviews as either **positive** or **negative** based on textual features extracted with **TF-IDF vectorization**.

## 📌 Project Highlights

- **Dataset**: IMDB movie reviews
- **Preprocessing**: Text cleaning, tokenization, lowercasing, stopword removal
- **Feature Extraction**: TF-IDF Vectorizer
- **Model**: Logistic Regression (scikit-learn)
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Custom Prediction Function**: Allows user input for sentiment prediction

## 🧰 Technologies Used

- Python 3.x
- Jupyter Notebook
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- NLTK

  ## 📊 Model Evaluation

Below are the actual performance metrics of the model based on the test set:

| Metric     | Class 0 (Negative) | Class 1 (Positive) | Macro Avg | Accuracy |
|------------|-------------------|--------------------|-----------|----------|
| Precision  | 0.90              | 0.86               | 0.88      | 0.88     |
| Recall     | 0.86              | 0.90               | 0.88      | 0.88     |
| F1-Score   | 0.88              | 0.88               | 0.88      | 0.88     |
| Support    | 5125              | 4875               | —         | 10000    |


## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/imdb-sentiment-analysis.git
   cd imdb-sentiment-analysis
