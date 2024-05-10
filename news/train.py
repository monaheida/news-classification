import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader

# Constants
MODEL_NAME = "distilbert-base-uncased"
NUM_EPOCHS = 40
BATCH_SIZE = 80
MAX_SEQ_LENGTH = 256
LEARN_RATE = 5e-5
ACCUM_STEPS = 4
SEED = 42

print("Model Setup:")
print(f"Model Name: {MODEL_NAME}")
print(f"Number of Epochs: {NUM_EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Max Sequence Length: {MAX_SEQ_LENGTH}")
print(f"Learning Rate: {LEARN_RATE}")
print(f"Accumulation Steps: {ACCUM_STEPS}")
print(f"Random Seed: {SEED}")

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

def load_jsonl(file_path):
	data = []
	with open(file_path, 'r', encoding='utf-8') as file:
		for line in file:
			data.append(json.loads(line))
	return data

def create_label_map(data):
    unique_labels = sorted(set(sample['category'] for sample in data))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    return label_map

def count_unique_classes(data):
    unique_classes = set(sample['category'] for sample in data)
    return len(unique_classes)

def main(train_file, model_output):
	train_data = load_jsonl(train_file)
	num_classes_dataset = count_unique_classes(train_data)

	tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
	model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes_dataset)

	print("Number of classes in dataset:", num_classes_dataset)
	print("Number of labels expected by model:", model.config.num_labels)

	label_map = create_label_map(train_data)
	train_dataset = CustomDataset(train_data, tokenizer, max_length=MAX_SEQ_LENGTH, label_map=label_map)  # Pass label_map
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)

	optimizer = AdamW(model.parameters(), lr=LEARN_RATE)

	for epoch in range(NUM_EPOCHS):
		loss = train(model, train_loader, optimizer, device)
		print(f'Epoch {epoch + 1}, Loss: {loss}')
	
	model.save_pretrained(model_output)


if __name__ == "__main__":
	train_data = 'data/train.jsonl'
    model_output = "distilbert_model"
    main(train_data, model_output)

