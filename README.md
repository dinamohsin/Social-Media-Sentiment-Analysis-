![NLP Sentiment Analysis](https://github.com/dinamohsin/Social-Media-Sentiment-Analysis-/blob/main/img/NLP.png?raw=true)


# Social Media Sentiment Analysis (Python & NLP)

This project focuses on analyzing public sentiment from social media data using Natural Language Processing (NLP) techniques. It involves cleaning and preprocessing raw text, performing sentiment classification using TextBlob, balancing the dataset, training a machine learning model, and visualizing sentiment trends.

---

## Table of Contents  
- [1. Project Overview](#1-project-overview)  
- [2. Project Goal](#2-project-goal)  
- [3. Dataset Review & Cleaning](#3-dataset-review--cleaning)  
- [4. Sentiment Analysis & Label Encoding](#4-sentiment-analysis--label-encoding)  
- [5. Text Preprocessing (NLTK)](#5-text-preprocessing-nltk)  
- [6. Data Balancing](#6-data-balancing)  
- [7. Model Building & Evaluation](#7-model-building--evaluation)  
- [8. Sentiment Trend Visualization](#8-sentiment-trend-visualization)  
- [9. Technologies Used](#9-technologies-used)  
- [10. Final Thoughts](#10-final-thoughts)  

---

## 1. Project Overview  
The project uses a dataset containing user-generated social media text and their corresponding sentiment labels. The goal is to understand public sentiment (positive or negative) and how it changes over time.

---

## 2. Project Goal  
The main objective is to apply NLP techniques to classify sentiments and visualize trends. The process includes text cleaning, sentiment classification, training a machine learning model, and using data visualizations for insights.

---

## 3. Dataset Review & Cleaning  
We began by inspecting the dataset's shape and checking for missing values. Unnecessary columns were dropped to streamline the analysis.

---

## 4. Sentiment Analysis & Label Encoding  
We used the `Sentiment` column (which includes labels like “sad,” “happy,” “positive,” “negative,” etc.) to create a binary target column using **TextBlob** polarity scoring:

```python
# Classify sentiment using polarity score
def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    return 'positive' if polarity > 0 else 'negative'

# Apply sentiment function and encode as 0 (negative) or 1 (positive)
df['Target'] = df['Sentiment'].apply(get_sentiment)
df['Target'] = df['Target'].map({'positive': 1, 'negative': 0})
```

---

## 5. Text Preprocessing (NLTK)  
We used **NLTK (Natural Language Toolkit)** for cleaning and preparing the text for machine learning:

### Key Steps:
- **Removing Noise:**  
  - URLs, hashtags, and special characters using **Regular Expressions**  
- **Tokenization:**  
  - Breaking text into words  
- **Lemmatization:**  
  - Reducing words to their base form (“running” → “run”)  
- **Removing Stop Words:**  
  - Filtering out common words like “the”, “is”, and”

> **Example:**  
> Before: `"Just finished an amazing workout! 💪"`  
> After: `finished amazing workout`

---

## 6. Data Balancing  
To avoid bias during model training, we balanced the dataset by oversampling the minority class:

```python
positive_df = df[df['Target'] == 1]
negative_df = df[df['Target'] == 0]

# Oversampling
if len(positive_df) > len(negative_df):
    diff = len(positive_df) - len(negative_df)
    negative_oversampled = negative_df.sample(diff, replace=True, random_state=42)
    balanced_df = pd.concat([positive_df, negative_df, negative_oversampled])
elif len(negative_df) > len(positive_df):
    diff = len(negative_df) - len(positive_df)
    positive_oversampled = positive_df.sample(diff, replace=True, random_state=42)
    balanced_df = pd.concat([positive_df, negative_df, positive_oversampled])
else:
    balanced_df = pd.concat([positive_df, negative_df])

# Shuffle dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
```

---

## 7. Model Building & Evaluation  
We vectorized the cleaned text using **TfidfVectorizer** and trained a **Logistic Regression** model.

### Feature & Target:
- `X` = Cleaned text (`'clean_text'` column)  
- `y` = Sentiment label (`'Target'` column)

### Train-Test Split:
- Split the dataset: 80% training, 20% testing  
- Evaluated the model using accuracy score

---

## 8. Sentiment Trend Visualization  
We visualized the trend of positive vs. negative sentiments over time using line chart to understand public opinion patterns and shifts.


![Positive vs. Negative Sentiments](https://github.com/dinamohsin/Social-Media-Sentiment-Analysis-/blob/main/img/positive%20vs.%20negative%20sentiments.png?raw=true)

---

## 9. Technologies Used  
- **Python**  
- **Pandas & NumPy** – Data manipulation  
- **NLTK** – Text preprocessing  
- **TextBlob** – Sentiment analysis  
- **scikit-learn** – Machine learning  
- **Matplotlib & Seaborn** – Data visualization  

---

## 10. Final Thoughts  
This project demonstrates the power of NLP in extracting meaningful insights from social media data. The combination of preprocessing, sentiment classification, and visualization provides valuable tools for monitoring public opinion and trends.
