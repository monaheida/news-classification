import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['headline'] + ' ' + self.data[idx]['short_description']
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
            'attention_mask': encoding['attention_mask'].flatten()
        }

def classify(model, classify_loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in classify_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    return all_preds

def main(model_file, data_file):
    with open(data_file, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    model = DistilBertForSequenceClassification.from_pretrained(model_file)

    classify_dataset = CustomDataset(data, tokenizer, max_length=128)
    classify_loader = DataLoader(classify_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    preds = classify(model, classify_loader, device)

    for i, article in enumerate(data):
        article['category'] = int(preds[i])  # Convert NumPy int64 to Python int
        print(json.dumps(article))

if __name__ == "__main__":
    model_file = input("Enter path to the trained model directory: ")
    data_file = input("Enter path to the data file to classify: ")
    main(model_file, data_file)

