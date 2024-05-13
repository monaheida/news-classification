import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score

MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16
LEARN_RATE = 2e-5
NUM_EPOCHS = 5

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

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader)

def main(train_file):
    train_data = load_jsonl(train_file)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    label_map_train = create_label_map(train_data)
    num_classes = len(label_map_train)

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)

    print("Number of classes in dataset:", num_classes)

    train_dataset = CustomDataset(train_data, tokenizer, max_length=MAX_SEQ_LENGTH, label_map=label_map_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARN_RATE)

    for epoch in range(NUM_EPOCHS):
        loss = train(model, train_loader, optimizer, device)
        print(f'Epoch {epoch + 1}, Loss: {loss}')

    # Save the trained model
    model_path = "distilbert_news_classifier.pth"
    torch.save(model.state_dict(), model_path)

    print("Trained model saved at:", model_path)

if __name__ == "__main__":
    main("data/train.jsonl")

