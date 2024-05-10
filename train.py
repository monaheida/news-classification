import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


def load_data(file_path):
	data = []
	with open(file_path, 'r', encoding='utf-8') as file:
		for line in file:
			data.append(json.loads(line))
	return data

def preprocess_data(data):
	X = [d['headline'] + ' ' + d['short_description'] for d in data]
	y = [d['category'] for d in data]
	return X, y

def train_model(X_train, y_train):
	model = Pipeline([
			('tfidf', TfidfVectorizer()),
			('clf', LogisticRegression(max_iter=1000))
	])
	model.fit(X_train, y_train)

	return model

def main():
	train_data = load_data('data/train.jsonl')

	X_train, y_train = preprocess_data(train_data)
	
	for data, label in zip(X_train, y_train):
		print("Data:", data)
		print("Label:", label)
		print()

	print("Training model...")

	model = train_model(X_train, y_train)
	
	print("Saving model...")

	joblib.dump(model, 'model.pkl')

	print("Training completed.")

if __name__ == "__main__":
    main()

