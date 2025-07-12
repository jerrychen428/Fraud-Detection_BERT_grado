from Fraud_Detection_BERT_grado_6 import FinancialFraudTrainer
from transformers import BertTokenizer, BertForSequenceClassification
import pytest


@pytest.fixture(scope='session')
def trainer():
    trainer = FinancialFraudTrainer()
    trainer.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    trainer.model = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels=2)
    trainer.model.eval()
    return trainer
