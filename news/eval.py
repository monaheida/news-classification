import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
    parser = argparse.ArgumentParser(description="Evaluate a text classification model using DistilBERT")
    parser.add_argument("model_file", help="Path to the trained model directory")
    parser.add_argument("eval_file", help="Path to the evaluation data file")
    args = parser.parse_args()
    main(args.model_file, args.eval_file)

