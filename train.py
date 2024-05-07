import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def preprocess_data(data):
    df = pd.DataFrame(data)
    df.dropna(subset=['headline', 'short_description', 'category'], inplace=True)
    return df[['headline', 'short_description', 'category']]


def train_model(train_df):
    # Combine headline & short_description into one text column
    train_df['text'] = train_df['headline'] + ' ' + train_df['short_description']
    X_train, X_val, y_train, y_val = train_test_split(train_df['text'], train_df['category'], test_size=0.2, random_state=42)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=1)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_val_tfidf = tfidf_vectorizer.transform(X_val)

    
    clf = SVC(kernel='linear')
    clf.fit(X_train_tfidf, y_train)

    
    y_pred = clf.predict(X_val_tfidf)
    report = classification_report(y_val, y_pred)
    print("Validation Classification Report:")
    print(report)

    return clf, tfidf_vectorizer


def save_model(clf, vectorizer, model_path):
    joblib.dump((clf, vectorizer), model_path)


def main():
    print("Reading and preprocessing data...")
    train_data = read_jsonl('data/train.jsonl')
    train_df = preprocess_data(train_data)

    print("Training model...")
    clf, vectorizer = train_model(train_df)

    print("Saving model...")
    save_model(clf, vectorizer, 'model.pkl')


if __name__ == "__main__":
    main()

