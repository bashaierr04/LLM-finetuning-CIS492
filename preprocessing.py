import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset

file_name = "Restaurant_reviews.csv"

try:
    df = pd.read_excel(file_name)
except:
    df = pd.read_csv(file_name, encoding="ISO-8859-1", on_bad_lines='skip', engine='python')

df.columns = df.columns.astype(str).str.strip()

if "7514" in df.columns:
    df.drop(columns=["7514"], inplace=True)

df.dropna(subset=["Review", "Rating"], inplace=True)
df = df[df["Rating"] != "Like"]
df["Rating"] = pd.to_numeric(df["Rating"], errors='coerce')
df = df.dropna(subset=["Rating"])
df["Rating"] = df["Rating"].round().astype(int)

def rating_to_label(rating):
    if rating <= 2:
        return 0
    else:
        return 1

df["label"] = df["Rating"].apply(rating_to_label)

def clean_text(text):
    text = re.sub(r"<.*?>", " ", str(text))
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

df["clean_review"] = df["Review"].apply(clean_text)
df = df[df["clean_review"].str.len() > 5]

min_count = df["label"].value_counts().min()
df_balanced = df.groupby("label", group_keys=False).apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)

train_df, temp_df = train_test_split(df_balanced, test_size=0.2, stratify=df_balanced["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["clean_review"], padding="max_length", truncation=True, max_length=512)

train_ds = Dataset.from_pandas(train_df[["clean_review", "label"]]).map(tokenize, batched=True)
val_ds = Dataset.from_pandas(val_df[["clean_review", "label"]]).map(tokenize, batched=True)
test_ds = Dataset.from_pandas(test_df[["clean_review", "label"]]).map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_df[["clean_review", "label"]].to_csv("train.csv", index=False)
val_df[["clean_review", "label"]].to_csv("val.csv", index=False)
test_df[["clean_review", "label"]].to_csv("test.csv", index=False)

print("✅ Preprocessing Complete! Files saved: train.csv, val.csv, test.csv")