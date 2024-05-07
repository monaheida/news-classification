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
	def __init__(self, data, tokenizer, max_length):
		self.data = data
		self.tokenizer = tokenizer
		self.max_length = max_length
    
	def __len__(self):
		return len(self.data)
    
	def __getitem__(self, idx):
		text = self.data[idx]['headline'] + ' ' + self.data[idx]['short_description']
		label = self.data[idx]['category']
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


def main(train_file, model_output):
	with open(train_file, 'r', encoding='utf-8') as file:
		train_data = [json.loads(line) for line in file]
    
    
	tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    
	model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=20)
    
    
	train_dataset = CustomDataset(train_data, tokenizer, max_length=MAX_SEQ_LENGTH)
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
    
	optimizer = AdamW(model.parameters(), lr=LEARN_RATE)
    
	for epoch in range(NUM_EPOCHS):
		loss = train(model, train_loader, optimizer, device)
		print(f'Epoch {epoch + 1}, Loss: {loss}')
    
    
	model.save_pretrained(model_output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a text classification model using DistilBERT")
    parser.add_argument("train_file", help="Path to the training data file")
    parser.add_argument("model_output", help="Path to save the trained model")
    args = parser.parse_args()
    main(args.train_file, args.model_output)

