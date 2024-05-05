import json
import pandas as pd
import joblib

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def preprocess_data(data):
    df = pd.DataFrame(data)
    df.dropna(subset=['headline', 'short_description'], inplace=True)
    return df[['headline', 'short_description']]

def load_model(model_path):
    clf, vectorizer = joblib.load(model_path)
    return clf, vectorizer

def classify_data(model, vectorizer, data):
    X = vectorizer.transform(data['headline'] + ' ' + data['short_description'])
    data['category'] = model.predict(X)
    return data

def main():
    print("Loading model...")
    clf, vectorizer = load_model('model.pkl')

    test_data = read_jsonl('data/test.jsonl')
    test_df = preprocess_data(test_data)

    print("Classifying data...")
    classified_data = classify_data(clf, vectorizer, test_df)

    print("Predicted Categories:")
    print(classified_data['category'].value_counts())

    classified_data.to_json('classified_data.jsonl', orient='records', lines=True)
    print("Classification complete. Classified data saved to 'classified_data.jsonl'.")

if __name__ == "__main__":
    main()

