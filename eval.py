import json
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

def evaluate_model(model, X_eval, y_eval):
	y_pred = model.predict(X_eval)
	accuracy = accuracy_score(y_eval, y_pred)
	print("Accuracy:", accuracy)

	print("Cls Report:")
	print(classification_report(y_eval, y_pred))
    
	cm = confusion_matrix(y_eval, y_pred)

	plt.figure(figsize=(14,12))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
			xticklabels=sorted(set(y_eval)),
			yticklabels=sorted(set(y_eval)))
	plt.xlabel('Predicted labels')
	plt.ylabel('True labels')
	plt.title('Confusion Matrix')
	plt.show()


def main():
	eval_data = load_data('data/dev.jsonl')

	X_eval, y_eval = preprocess_data(eval_data)

	model = joblib.load('model.pkl')

	print("Evaluating model...")
	evaluate_model(model, X_eval, y_eval)

	print("Evaluation completed.")


if __name__ == "__main__":
    main()

