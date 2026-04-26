

import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline

train_df = pd.read_csv('/content/testfinal.csv')
test_df = pd.read_csv('/content/testfinal.csv')
val_df = pd.read_csv('/content/valfinal.csv')


model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["clean_review"], padding="max_length", truncation=True, max_length=128)

train_ds = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
val_ds = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
test_ds = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
)

# 6. تشغيل الـ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

print("train time")
trainer.train()


#final result
print("final result")
print(trainer.evaluate(test_ds))

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)
#test any sentence
result = classifier(" the food was very good delivery was bad but i will give the resturant another chance ")
print(result)
