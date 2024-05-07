import json
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

def load_model(model_path):
    clf, vectorizer = joblib.load(model_path)
    return clf, vectorizer

def evaluate_model(model, vectorizer, data):
	X = vectorizer.transform(data['headline'] + ' ' + data['short_description'])
	y_true = data['category']
	y_pred = model.predict(X)

	accuracy = accuracy_score(y_true, y_pred)
	confusion = confusion_matrix(y_true, y_pred)
	report = classification_report(y_true, y_pred)

	return accuracy, confusion, report

def main():
    print("Loading model...")
    clf, vectorizer = load_model('model.pkl')

    eval_data = read_jsonl('data/dev.jsonl')
    eval_df = preprocess_data(eval_data)

    print("Evaluating model...")
    accuracy, confusion, report = evaluate_model(clf, vectorizer, eval_df)
    print(f'Accuracy: {accuracy:.4f}')
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    main()

