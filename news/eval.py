import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, label_map):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data[idx]['headline']) + ' ' + str(self.data[idx]['short_description'])
        label = self.label_map[self.data[idx]['category']]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_label_map(data):
    unique_labels = sorted(set(sample['category'] for sample in data))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    return label_map

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

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

def main(eval_file):
    eval_data = load_jsonl(eval_file)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    label_map_eval = create_label_map(eval_data)

    eval_dataset = CustomDataset(eval_data, tokenizer, max_length=MAX_SEQ_LENGTH, label_map=label_map_eval)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_map_eval))
    model.to(device)

    model.load_state_dict(torch.load('distilbert_news_classifier.pth'))
    model.eval()

    true_labels, pred_labels = evaluate(model, eval_loader, device)
    accuracy = accuracy_score(true_labels, pred_labels)
    confusion = confusion_matrix(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    main("data/dev.jsonl")

