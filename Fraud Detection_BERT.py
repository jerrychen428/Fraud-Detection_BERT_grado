import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import uvicorn
from fastapi import FastAPI

# Load Pretrained BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Simulated Financial Fraud Dataset
data = {
    "text": [
        "Suspicious transfer of $5000 detected from account 1234.",
        "Regular transaction of $200 at Walmart.",
        "Fraudulent activity detected with credit card 5678.",
        "Payment of $1500 sent to offshore account.",
        "Normal transaction: purchase of groceries for $50."
    ],
    "label": [1, 0, 1, 1, 0]  # 1: Fraud, 0: Legitimate
}

# Convert dataset to DataFrame
df = pd.DataFrame(data)

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=0.2)

# Tokenization
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

class FinancialFraudDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = FinancialFraudDataset(train_encodings, train_labels)
val_dataset = FinancialFraudDataset(val_encodings, val_labels)

# Load Pretrained BERT Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Define Compute Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train Model
trainer.train()

# Save Model
model.save_pretrained("fraud_bert_model")
tokenizer.save_pretrained("fraud_bert_model")

# FastAPI Deployment
app = FastAPI()

@app.post("/predict")
def predict_transaction(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        output = model(**inputs)
        prediction = torch.argmax(output.logits).item()
    return {"text": text, "fraudulent": bool(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)