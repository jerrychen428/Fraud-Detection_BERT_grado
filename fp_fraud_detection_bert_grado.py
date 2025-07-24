import os
import pathlib
from collections import namedtuple
from collections.abc import Callable

import gradio
import pandas as pd
import sklearn
import sklearn.metrics
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction


CWD = pathlib.Path(__file__).parent.resolve()
_MODEL_DIR = CWD.joinpath(os.environ.get('MODEL_PATH_DIR_NAME', default="fraud_bert_model"))
_MODEL_BIN_NAME = os.environ.get('MODEL_BIN_NAME', default='pytorch_model.bin')
_MODEL_SAFE_NAME = os.environ.get('MODEL_BIN_NAME', default='model.safetensors')


PreDataSet = namedtuple('DataSetTuple', field_names=(
    'train_texts',
    'val_texts',
    'train_labels',
    'val_labels',
))


class FinancialFraudDataset(torch.utils.data.Dataset):
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


def is_model_trained(
    model_dir_name: str = 'fraud_bert_model',
    model_bin_name: str = 'pytorch_model.bin',
    model_safe_name: str = 'model.safetensors',
) -> bool:
    model_dir = CWD.joinpath(model_dir_name)

    if not model_dir.exists():
        model_dir.mkdir()

    if any((
        model_dir.joinpath(model_bin_name).exists(),
        model_dir.joinpath(model_safe_name).exists(),
    )):
        print("✅ Model already trained.")
        return True

    print("❌ Model not trained yet.")
    return False


def prepare_dataset(data_path: str = None):
    path = CWD.joinpath(data_path or 'fraud_detection_sample.csv')

    if not path.is_file():
        raise Exception(f"{path} file not exists.")

    df = pd.read_csv(str(path), encoding="utf-8")

    dataset_list = sklearn.model_selection.train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42,
    )

    return PreDataSet(
        train_texts=dataset_list[0],
        val_texts=dataset_list[0],
        train_labels=dataset_list[0],
        val_labels=dataset_list[0],
    )


def train_model(
    dataset: PreDataSet,
    tokenizer: BertTokenizer = None,
    model: BertForSequenceClassification = None,
    train_args: TrainingArguments = None,
    compute_metrics: Callable[[EvalPrediction], dict] = None,
    model_save_dir: str | pathlib.Path = 'fraud_bert_model',
) -> pathlib.Path:
    """Train model with dataset then return trained model data path"""
    def _compute_metrics(pred: EvalPrediction):
        # 計算 accuracy、precision、recall、F1 分數
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = sklearn.metrics.accuracy_score(labels, preds)
        precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
            y_true=labels,
            y_pred=preds,
            average="binary",
        )
        return dict(
            accuracy=acc,
            precision=precision,
            recall=recall,
            f1=f1,
        )

    tokenizer = tokenizer or BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    model = model or BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels=2)
    train_args = train_args or TrainingArguments(
        output_dir="./results",        # 訓練結果儲存位置
        num_train_epochs=20,           # 訓練輪數
        per_device_train_batch_size=4, # 每批訓練數量
        per_device_eval_batch_size=4,  # 每批驗證數量
        warmup_steps=10,               # 預熱步驟數
        weight_decay=0.01,             # 權重衰退
        logging_dir="./logs",          # 日誌儲存位置
        logging_steps=10,              # 日誌紀錄頻率
        report_to="none"               # 不使用外部工具報告訓練過程
    )
    compute_metrics = compute_metrics or _compute_metrics

    train_dataset = FinancialFraudDataset(
        encodings=tokenizer(dataset.train_texts, truncation=True, padding=True, max_length=128),
        labels=dataset.train_labels,
    )
    val_dataset = FinancialFraudDataset(
        encodings=tokenizer(dataset.val_texts, truncation=True, padding=True, max_length=128),
        labels=dataset.val_labels,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics, # 計算評估指標
    )
    trainer.train()

    # Save model and tokenizer
    model_dir = CWD.joinpath(model_save_dir) if isinstance(model_save_dir, str) else model_save_dir
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    return model_dir


def show_training_result_with_gradio(model_dir: pathlib.Path):
    model = BertForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))

    def _predict_transaction(inp: str):
        try:
            with torch.no_grad():
                inputs = tokenizer(inp, return_tensors="pt", truncation=True, padding=True, max_length=128)
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)  # 機率分布
                prediction = torch.argmax(probs, dim=1).item()

            confidence = probs[0][prediction].item()
            label = "✅ Legitimate" if prediction == 0 else "⚠️ Fraudulent"
            return f"{label}  (Confidence: {confidence:.2f})"

        except Exception as e:
            return f"Error: {str(e)}"
    
    gradio.Interface(
        fn=_predict_transaction,   # 指定推論函式
        inputs=gradio.Textbox(lines=3, placeholder="輸入交易簡訊..."),
        outputs="text",
        title="💳 中英文詐騙簡訊判斷器",
        description="輸入交易相關訊息，判斷是否為詐騙訊息（支援中文與英文）。"
    ).launch(share=False) # 如果防毒軟體會報錯，請將share=True, debug=True改為share=False


def main():
    model_dir_name = os.environ.get('MODEL_DIR_NAME')
    model_bin_name = os.environ.get('MODEL_BIN_NAME')
    model_safe_name = os.environ.get('MODEL_SAFE_NAME')
    dataset_path = os.environ.get('DATA_PATH')

    model_dir = CWD.joinpath(model_dir_name)

    if not is_model_trained(
        model_dir_name,
        model_bin_name,
        model_safe_name,
    ):
        # TRAIN data
        dataset = prepare_dataset(dataset_path)
        model_dir = train_model(dataset)
    
    show_training_result_with_gradio(model_dir)
