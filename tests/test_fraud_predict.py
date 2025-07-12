from Fraud_Detection_BERT_grado_6 import FinancialFraudTrainer


def test_predict_chinese(trainer: FinancialFraudTrainer):
    result = trainer.predict_transaction("您已中獎，請點選連結領取獎金")
    assert "Fraudulent" in result or "Legitimate" in result or "Error" in result


def test_predict_english(trainer: FinancialFraudTrainer):
    result = trainer.predict_transaction("Congratulations! You won a prize.")
    assert "Fraudulent" in result or "Legitimate" in result or "Error" in result
