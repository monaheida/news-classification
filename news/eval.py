import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {
            "politics": 0,
            "healthyliving": 1,
            "travel": 2,
            "entertainment": 3,
            "stylebeauty": 4,
            "sports": 5,
            "arts": 6,
            "queervoices": 7,
            "business": 8,
            "media": 9,
            "latinovoices": 10,
            "worldnews": 11,
            "usnews": 12,
            "parents": 13,
            "parenting": 14,
            "blackvoices": 15,
            "wellness": 16,
            "taste": 17,
            "divorce": 18,
            "fallback_label": -1
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        headline = self.data[index]['headline']
        category = self.data[index]['category'].strip().lower().replace(" ", "_")
        category = ''.join(e for e in category if e.isalnum())

        try:
            label = self.label_map[category]
        except KeyError:
            label = self.label_map["fallback_label"]

        encoding = self.tokenizer(headline, truncation=True, padding='max_length', max_length=self.max_length,
                                  return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def evaluate(model, eval_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return all_labels, all_preds


def main(model_file, eval_file):
    with open(eval_file, 'r', encoding='utf-8') as file:
        eval_data = [json.loads(line) for line in file]

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    model = DistilBertForSequenceClassification.from_pretrained(model_file)

    eval_dataset = CustomDataset(eval_data, tokenizer, max_length=128)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    true_labels, pred_labels = evaluate(model, eval_loader, device)
    accuracy = accuracy_score(true_labels, pred_labels)
    confusion = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a DistilBERT model on a dataset.")
    parser.add_argument("--model", type=str, help="Path to the trained model directory")
    parser.add_argument("--eval_data", type=str, help="Path to the evaluation data file")
    args = parser.parse_args()

    main(args.model, args.eval_data)

