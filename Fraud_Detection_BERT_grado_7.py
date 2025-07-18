import os
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import gradio as gr

class FinancialFraudDataset(Dataset):
    """
    自定義 Dataset 類別，用於將文本和標籤轉換為 PyTorch 能處理的格式。
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 將每筆資料轉換為 tensor，包含 token 編碼及對應的標籤
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

class FinancialFraudTrainer:
    def __init__(self):
        self.data_path = None
        self.train_texts = None
        self.val_texts = None
        self.train_labels = None
        self.val_labels = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.model = None

    def prepare_dataset(self, file_path="./fraud_detection_sample.csv"):
        # 讀取 CSV 檔案，使用 UTF-8 編碼
        self.data_path = file_path or self.data_path
        df = pd.read_csv(self.data_path, encoding="utf-8")
        # 分割為訓練集與驗證集
        self.train_texts, self.val_texts, self.train_labels, self.val_labels = train_test_split(
            df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)

    def tokenize_data(self, pretrained="hfl/chinese-roberta-wwm-ext"):
        # 載入中文 RoBERTa tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        # 對訓練與驗證文本進行編碼
        train_encodings = self.tokenizer(self.train_texts, truncation=True, padding=True, max_length=128)
        val_encodings = self.tokenizer(self.val_texts, truncation=True, padding=True, max_length=128)
        # 封裝成 Dataset
        self.train_dataset = FinancialFraudDataset(train_encodings, self.train_labels)
        self.val_dataset = FinancialFraudDataset(val_encodings, self.val_labels)

    def load_model(self, pretrained="hfl/chinese-roberta-wwm-ext"):
        # 載入中文 RoBERTa 分類模型，設定分類數為 2（合法 / 詐騙）
        self.model = BertForSequenceClassification.from_pretrained(pretrained, num_labels=2)

    def train_model(self, output_dir="./results"):
        # 設定訓練參數
        training_args = TrainingArguments(
            output_dir=output_dir,                # 訓練結果儲存位置
            num_train_epochs=20,                    # 訓練輪數
            per_device_train_batch_size=4,          # 每批訓練數量
            per_device_eval_batch_size=4,           # 每批驗證數量
            warmup_steps=10,                         # 預熱步驟數
            weight_decay=0.01,                       # 權重衰退
            logging_dir="./logs",                  # 日誌儲存位置
            logging_steps=10,                        # 日誌紀錄頻率
            report_to="none"                         # 不使用外部工具報告訓練過程
        )

        # 定義 Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics   # 計算評估指標
        )

        # 執行訓練
        trainer.train()

    def compute_metrics(self, pred):
        # 計算 accuracy、precision、recall、F1 分數
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    def save_model(self):
        # 儲存模型與 tokenizer
        self.model.save_pretrained("fraud_bert_model")
        self.tokenizer.save_pretrained("fraud_bert_model")

    def load_saved_model(self):
        # 重新載入已儲存的模型與 tokenizer，供推論使用
        self.model = BertForSequenceClassification.from_pretrained("fraud_bert_model")
        self.tokenizer = BertTokenizer.from_pretrained("fraud_bert_model")
        self.model.eval()

    def predict_transaction(self, text):
        # 單筆推論用，回傳預測結果與信心分數
        try:
            self.model.eval()
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)  # 機率分布
                prediction = torch.argmax(probs, dim=1).item()
                confidence = probs[0][prediction].item()
            label = "✅ Legitimate" if prediction == 0 else "⚠️ Fraudulent"
            return f"{label}  (Confidence: {confidence:.2f})"
        except Exception as e:
            return f"Error: {str(e)}"

    def launch_gradio(self):
        # 使用 Gradio 部署網頁介面
        gr.Interface(
            fn=self.predict_transaction,   # 指定推論函式
            inputs=gr.Textbox(lines=3, placeholder="輸入交易簡訊..."),
            outputs="text",
            title="💳 中英文詐騙簡訊判斷器",
            description="輸入交易相關訊息，判斷是否為詐騙訊息（支援中文與英文）。"
        ).launch(share=False) # 如果防毒軟體會報錯，請將share=True, debug=True改為share=False

    # 檢查模型是否已經訓練過function
    def model_already_trained(self):
        print("Checking if model already trained...")
        model_path_bin = os.path.join("fraud_bert_model", "pytorch_model.bin")
        model_path_safe = os.path.join("fraud_bert_model", "model.safetensors")

        if any([
            os.path.exists(model_path_bin),
            os.path.exists(model_path_safe),
        ]):
            print("✅ Model already trained.")
            return True

        print("❌ Model not trained yet.")
        return False

if __name__ == "__main__":
    # 建立 Trainer 實例
    trainer = FinancialFraudTrainer()
    if not trainer.model_already_trained():# 檢查模型是否已訓練
        trainer.prepare_dataset()       # 資料前處理
        trainer.tokenize_data()         # 文字編碼
        trainer.load_model()            # 載入模型
        trainer.train_model()           # 模型訓練
        trainer.save_model()            # 儲存模型

    trainer.load_saved_model()  # 載入模型供預測
    trainer.launch_gradio()     # 啟動 Gradio 網頁介面
