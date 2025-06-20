import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import gradio as gr
import numpy as np

# 模擬資料集
data = {
    "text": [
        "Suspicious transfer of $5000 detected from account 1234.",
        "Regular transaction of $200 at Walmart.",
        "Fraudulent activity detected with credit card 5678.",
        "Payment of $1500 sent to offshore account.",
        "Normal transaction: purchase of groceries for $50."
    ],
    "label": [1, 0, 1, 1, 0]
}
df = pd.DataFrame(data)

# 分割訓練與驗證
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Dataset class
class FinancialFraudDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = FinancialFraudDataset(train_encodings, train_labels)
val_dataset = FinancialFraudDataset(val_encodings, val_labels)

# 模型
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 計算 class weights，轉成 tensor 並放到 model 使用的 device
class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# 自訂 Trainer 讓 loss 支援 class weights
from transformers import Trainer

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # 將 class_weights 放到同一裝置
        weights = class_weights.to(logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# 訓練參數
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=70,  # 增加epoch
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"
)

# 評估指標
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 建立 trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 訓練
trainer.train()

# 儲存模型
model.save_pretrained("fraud_bert_model_1")
tokenizer.save_pretrained("fraud_bert_model_1")

# 載入模型供推論用
model = BertForSequenceClassification.from_pretrained("fraud_bert_model_1")
tokenizer = BertTokenizer.from_pretrained("fraud_bert_model_1")
model.eval()

# 預測函式
def predict_transaction(text):
    try:
        model.eval()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        return "✅ Legitimate" if prediction == 0 else "⚠️ Fraudulent"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio 介面
interface = gr.Interface(
    fn=predict_transaction,
    inputs=gr.Textbox(lines=3, placeholder="Enter transaction description..."),
    outputs="text",
    title="💳 Financial Fraud Detector",
    description="Enter a transaction message to predict whether it's fraudulent or legitimate."
)

# 啟動介面（加上 share=True 可讓外部存取）
interface.launch(share=False)
