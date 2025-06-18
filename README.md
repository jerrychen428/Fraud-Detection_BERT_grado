# 🛡️ Fraud-Detection_BERT_grado
> 📘 A bilingual (Traditional Chinese & English) scam message detection system powered by deep learning.

## 👨‍💻 ccClub Student Project

---

## 🔍 專案簡介
本專案旨在開發一個基於 Python 與深度學習語言模型的詐騙訊息偵測系統，能夠有效處理繁體中文與英文雙語資料。透過研究與微調現有的開源語言分類模型，最終實作出一套具備 API 介面的應用，使用者可直接輸入訊息，模型即自動辨識是否為潛在詐騙內容。

---

## 📑 計畫書連結
👉 [查看完整計畫書（Google Drive）](https://drive.google.com/file/d/1oROgam9Gi4sWE_0txIXegUWiMfho0Kko/view?usp=drive_link)

---

## 🧠 使用技術與模型

- `Python 3.10`
- `PyTorch` 深度學習框架
- `Transformers` 套件（Hugging Face）
- `Gradio` 前端介面
- `sklearn`資料集分切, 精準度分數
- 模型基底：[`hfl/chinese-roberta-wwm-ext`](https://huggingface.co/hfl/chinese-roberta-wwm-ext)

---

## 🗂️ 專案結構
```bash
.
├── Fraud Detection_BERT.py                 # 原文code轉換 (.ipynb → .py)
├── Fraud Detection_BERT_grado.py          # Gradio 第一次迭代版本
├── Fraud Detection_BERT_grado_1.py        # 第二次迭代
├── Fraud Detection_BERT_grado_2.py        # 第三次迭代
├── Fraud Detection_BERT_grado_3.py        # 第四次迭代
├── Fraud Detection_BERT_grado_4.py        # 第五次迭代
├── Fraud Detection_BERT_grado_5.py        # 第六次迭代，開始分切code，準備寫成Class，方便協做
├── Fraud Detection_BERT_grado_6.py        # 第七次迭代，寫成class歡迎協做
├── README.md                              # 說明文件
├── data_clean.py                          # 資料清洗py檔範本
├── fraud_detection_sample.csv             # 資料集CSV檔
└── requirements.txt                       # 環境需求
```
---

## 📚 Reference 資源與致謝

> 本專案靈感與部分程式碼來自以下開源項目與模型，特此致謝：

- 📌 **模型：**  
  - [`hfl/chinese-roberta-wwm-ext`](https://huggingface.co/hfl/chinese-roberta-wwm-ext) – 哈工大與科大訊飛聯合實驗室發布的全詞遮罩中文語言模型  
  - [GitHub 原始碼](https://github.com/ymcui/Chinese-BERT-wwm) – 由 Yiming Cui 團隊維護的 BERT-WWM 中文模型系列  
  - [iFLYTEK HFL-Anthology 模型總覽](https://github.com/iflytek/HFL-Anthology?tab=readme-ov-file#Pre-trained-Language-Model)

- 💻 **程式碼參考與基礎模板：**  
  - [alamin19/fraud-detection-bert](https://github.com/alamin19/fraud-detection-bert) – 英文詐騙訊息分類的 BERT 模型範例

- 📘 **Gradio 應用教學與參考：**  
  - [Gradio 官方文檔](https://www.gradio.app/guides)
  - [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers/index)

---

