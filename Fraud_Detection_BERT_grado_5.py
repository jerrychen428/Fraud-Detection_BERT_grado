import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import gradio as gr
import pandas as pd

# 資料集準備函式
def dataset_preparation():
    # 讀取資料集    
    df = pd.read_csv("./fraud_detection_sample.csv", encoding="utf-8")
    # 分割資料集為訓練集和驗證集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)
    return train_texts, val_texts, train_labels, val_labels  

# 使用中文 BERT tokenizer
def Bert_Tokeninzner(train_texts, val_texts):
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    return train_encodings, val_encodings, tokenizer  

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

def create_datasets(train_encodings, val_encodings, train_labels, val_labels):
    train_dataset = FinancialFraudDataset(train_encodings, train_labels)
    val_dataset = FinancialFraudDataset(val_encodings, val_labels)
    return train_dataset, val_dataset

# 載入中文 BERT 模型
def load_model():    
    model = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels=2)
    return model

# 訓練參數
def Training_args():
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=20,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none"
    )
    return training_args

# 評估指標
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 設定 Trainer
def create_trainer(model, train_dataset, val_dataset, training_args):    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()

# 儲存模型和tokenizer
def save_model(model, tokenizer):    
    model.save_pretrained("fraud_bert_model")
    tokenizer.save_pretrained("fraud_bert_model")

def load_saved_model():
    model = BertForSequenceClassification.from_pretrained("fraud_bert_model")
    tokenizer = BertTokenizer.from_pretrained("fraud_bert_model")
    model.eval()
    return model, tokenizer

# 預測函式（加上信心分數）
def predict_transaction(text):
    try:
        model.eval()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
        label = "✅ Legitimate" if prediction == 0 else "⚠️ Fraudulent"
        return f"{label}  (Confidence: {confidence:.2f})"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio 介面
def Gradio_interface():
    gr.Interface(
        fn=predict_transaction,
        inputs=gr.Textbox(lines=3, placeholder="輸入交易簡訊..."),
        outputs="text",
        title="💳 中英文詐騙簡訊判斷器",
        description="輸入交易相關訊息，判斷是否為詐騙訊息（支援中文與英文）。"
    ).launch(share=False)

# 主程式入口
if __name__ == "__main__":
    # 資料準備
    train_texts, val_texts, train_labels, val_labels = dataset_preparation()
    train_encodings, val_encodings, tokenizer = Bert_Tokeninzner(train_texts, val_texts)
    train_dataset, val_dataset = create_datasets(train_encodings, val_encodings, train_labels, val_labels)
    Load_model = load_model()
    # 訓練參數
    training_args = Training_args()
    create_trainer(Load_model, train_dataset, val_dataset, training_args)
    # 訓練模型
    save_model(Load_model, tokenizer)
    # 載入模型供推論
    model, tokenizer = load_saved_model()
    # 啟動 Gradio 介面
    Gradio_interface()
