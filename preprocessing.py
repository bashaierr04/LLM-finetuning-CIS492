import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset

nltk.download('vader_lexicon', quiet=True)

# Initialize VADER
sia = SentimentIntensityAnalyzer()


file_name = "Restaurant_reviews.csv"

try:
    df = pd.read_excel(file_name)
except:
    df = pd.read_csv(file_name, encoding="ISO-8859-1", on_bad_lines='skip', engine='python')

df.columns = df.columns.astype(str).str.strip()

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())


if "7514" in df.columns:
    df.drop(columns=["7514"], inplace=True)

df.dropna(subset=["Review", "Rating"], inplace=True)

print("\nAfter dropping nulls:", df.shape)


df = df[df["Rating"] != "Like"]
df["Rating"] = pd.to_numeric(df["Rating"], errors='coerce')
df = df.dropna(subset=["Rating"])
df["Rating"] = df["Rating"].round().astype(int)

print("\nCleaned rating distribution:\n", df["Rating"].value_counts().sort_index())


def get_vader_sentiment(text):
    score = sia.polarity_scores(str(text))['compound']
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

print("\nRunning VADER on all reviews (this takes a few seconds)...")
df["vader_sentiment"] = df["Review"].apply(get_vader_sentiment)

print("VADER sentiment distribution:")
print(df["vader_sentiment"].value_counts())


def hybrid_label(row):
    rating = row["Rating"]
    vader = row["vader_sentiment"]

    # Rating 3 is always Neutral
    if rating == 3:
        return 1  # Neutral

    # VADER detected mixed/neutral sentiment → Neutral
    if vader == "neutral":
        return 1  # Neutral

    # Rating says positive (4-5) AND VADER agrees → Positive
    if rating >= 4 and vader == "positive":
        return 2  # Positive

    # Rating says negative (1-2) AND VADER agrees → Negative
    if rating <= 2 and vader == "negative":
        return 0  # Negative

    # Rating and VADER disagree → mixed signals → Neutral
    return 1  


df["label"] = df.apply(hybrid_label, axis=1)

label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
print("\nFinal hybrid label distribution:")
print(df["label"].map(label_names).value_counts())
print()

# Show a few examples of what VADER caught that rating alone would have missed
mixed = df[(df["Rating"] >= 4) & (df["label"] == 1)][["Review", "Rating", "vader_sentiment", "label"]].head(3)
print("Examples of high-rated reviews VADER flagged as Neutral (mixed):")
for _, row in mixed.iterrows():
    print(f"  Rating {row['Rating']} | VADER: {row['vader_sentiment']} | {str(row['Review'])[:120]}")


def clean_text(text):
    text = re.sub(r"<.*?>", " ", str(text))
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.lower()
    return text

df["clean_review"] = df["Review"].apply(clean_text)
df = df[df["clean_review"].str.len() > 5]

print("\nSample cleaned reviews per label:")
for label_id, label_name in label_names.items():
    subset = df[df["label"] == label_id]
    if len(subset) > 0:
        sample = subset["clean_review"].iloc[0]
        print(f"  [{label_name}] {sample[:120]}")


min_count = df["label"].value_counts().min()
print(f"\nBalancing all 3 classes to {min_count} samples each...")

df_balanced = (
    df.groupby("label", group_keys=False)
      .apply(lambda x: x.sample(min_count, random_state=42))
      .reset_index(drop=True)
)

print("Balanced label distribution:")
print(df_balanced["label"].map(label_names).value_counts())


train_df, temp_df = train_test_split(
    df_balanced, test_size=0.2, stratify=df_balanced["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

print(f"\nSplit sizes → Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


MODEL_NAME = "bert-base-uncased"

print(f"\nLoading tokenizer: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["clean_review"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

train_ds = Dataset.from_pandas(train_df[["clean_review", "label"]]).map(tokenize, batched=True)
val_ds   = Dataset.from_pandas(val_df[["clean_review", "label"]]).map(tokenize,   batched=True)
test_ds  = Dataset.from_pandas(test_df[["clean_review", "label"]]).map(tokenize,  batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(  type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format( type="torch", columns=["input_ids", "attention_mask", "label"])

print("Tokenization complete!")


train_df[["clean_review", "label"]].to_csv("train.csv", index=False)
val_df[["clean_review", "label"]].to_csv("val.csv",     index=False)
test_df[["clean_review", "label"]].to_csv("test.csv",   index=False)

print("\n✅ Preprocessing Complete! Files saved: train.csv | val.csv | test.csv")
print(f"   train_ds → {len(train_ds)} samples")
print(f"   val_ds   → {len(val_ds)} samples")
print(f"   test_ds  → {len(test_ds)} samples")
print("\n── Notes for the fine-tuning teammate ────────────────────")
print(f"   Model       : {MODEL_NAME}")
print("   num_labels  : 3  ← IMPORTANT: must be 3 not 2")
print("   Labels      : 0 = Negative | 1 = Neutral | 2 = Positive")
print("   Input column: clean_review")
print("   Label column: label")
print("   Labeling    : Hybrid (VADER compound score + star rating)")
