import json
import joblib

def load_data(file_path):
	data = []
	with open(file_path, 'r', encoding='utf-8') as file:
		for line in file:
			data.append(json.loads(line))
	return data

def preprocess_data(data):
    X = [d['headline'] + ' ' + d['short_description'] for d in data]
    return X

def classify_data(model, X):
    y_pred = model.predict(X)
    return y_pred

def main():
	data_to_classify = load_data('data/test.jsonl')

	X_to_classify = preprocess_data(data_to_classify)

	model = joblib.load('model.pkl')

	print("Classifying data...")
	y_pred = classify_data(model, X_to_classify)
	
	"""
	# uncomment to see the predicted category in terminal
	for i, item in enumerate(data_to_classify):
		item['category'] = y_pred[i]
		print(f"Item {i+1}: Predicted Category - {y_pred[i]}")
	"""

	for i in range(len(data_to_classify)):
		data_to_classify[i]['category'] = y_pred[i]

	with open('classified_data.jsonl', 'w', encoding='utf-8') as f:
		for item in data_to_classify:
			f.write(json.dumps(item) + '\n')

	print("Classification completed. Classified data saved to 'classified_data.jsonl'.")


if __name__ == "__main__":
    main()

