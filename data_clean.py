import pandas as pd

# 讀取原始 CSV 檔案
df = pd.read_csv("./email_spam.csv")

# 刪除 'title' 欄位（如果存在）
if 'title' in df.columns:
    df.drop(columns=['title'], inplace=True)

# 將 'type' 欄位重新命名為 'label'
if 'type' in df.columns:
    df.rename(columns={'type': 'label'}, inplace=True)

# 將 'label' 欄位的值轉換為數值：'spam' → 1，'not spam' → 0
if 'label' in df.columns:
    df['label'] = df['label'].map({'spam': 1, 'not spam': 0})

# 儲存清洗後的結果為新 CSV
df.to_csv("./cleaned_email_spam.csv", index=False)

print("檔案清洗完成，已儲存為 cleaned_email_spam.csv")
