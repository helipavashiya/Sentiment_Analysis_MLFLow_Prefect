# import libraries
from prefect import task, flow
import pandas as pd
import re

# For splitting train & test
from sklearn.model_selection import train_test_split 

# Data preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

# models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
@task
def load_data(file_path):
    return pd.read_csv(file_path)

@task
def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = text.lower()
    return text

@task
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

@task
def get_feedback(sentiment_score):
    return 'Positive' if sentiment_score > 0 else 'Negative'

@task
def split_train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

@task
def preprocess_data(X_train, X_test, y_train, y_test):
    vectorization = CountVectorizer()
    X_train_vec = vectorization.fit_transform(X_train)
    X_test_vec = vectorization.transform(X_test)
    return X_train_vec, X_test_vec, y_train, y_test

@task
def train_model(X_train_vec, y_train, hyperparameters):
    clf = LogisticRegression(**hyperparameters)
    clf.fit(X_train_vec, y_train)
    return clf

@task
def evaluate_model(model, X_train_vec, y_train, X_test_vec, y_test):
    y_train_pred = model.predict(X_train_vec)
    y_test_pred = model.predict(X_test_vec)
    train_score = f1_score(y_train, y_train_pred, average='micro')
    test_score = f1_score(y_test, y_test_pred, average='micro')
    return train_score, test_score


@flow(name="Logistic Regression Flow")
def workflow(data_path): 
    
    HYPERPARAMETERS = {
        'C': 10.0,
        'penalty': 'elasticnet',
        'l1_ratio': 0.6,
        'solver': 'saga',
        'class_weight': 'balanced'
    }
    
    data = load_data(data_path="reviews_badminton.csv")
    preprocessed_text = preprocess_text.map(data['Review text'])
    sentiment_scores = get_sentiment.map(preprocessed_text)
    feedbacks = get_feedback.map(sentiment_scores)
    
    X = preprocessed_text
    y = feedbacks
    
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    X_train_vec, X_test_vec, y_train, y_test = preprocess_data(X_train, X_test, y_train, y_test)
    
    model = train_model(X_train_vec, y_train, HYPERPARAMETERS)
    
    train_score, test_score = evaluate_model(model, X_train_vec, y_train, X_test_vec, y_test)
    

if __name__ == "__main__":
    workflow.serve(
        name="hella-first-deployment",
        cron="0 0 1 5 *"
    )
