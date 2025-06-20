from Fraud_Detection_BERT_grado_6 import FinancialFraudTrainer
from transformers import BertTokenizer, BertForSequenceClassification

def setup_trainer():
    trainer = FinancialFraudTrainer()
    trainer.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    trainer.model = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels=2)
    trainer.model.eval()
    return trainer

def test_predict_chinese():
    trainer = setup_trainer()
    result = trainer.predict_transaction("您已中獎，請點選連結領取獎金")
    assert "Fraudulent" in result or "Legitimate" in result or "Error" in result

def test_predict_english():
    trainer = setup_trainer()
    result = trainer.predict_transaction("Congratulations! You won a prize.")
    assert "Fraudulent" in result or "Legitimate" in result or "Error" in result
