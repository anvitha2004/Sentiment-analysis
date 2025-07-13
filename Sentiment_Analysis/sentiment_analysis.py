# === SETUP ===
import os
os.environ["WANDB_DISABLED"] = "true"
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
import io

# === LOAD DATA ===
df = pd.read_csv('/content/drive/MyDrive/reviews.csv')
df = df.dropna(subset=["rating", "review_text", "place_name"])

def label_sentiment(rating):
    return 2 if rating >= 4 else 1 if rating == 3 else 0

df["label"] = df["rating"].apply(label_sentiment).astype(int)

# === OVERSAMPLING TO BALANCE CLASSES ===
df_minority = df[df['label'] != 2]
df_majority = df[df['label'] == 2]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# === TOKENIZATION ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_balanced["review_text"].tolist(),
    df_balanced["label"].tolist(),
    test_size=0.2,
    stratify=df_balanced["label"].tolist(),
    random_state=42
)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

class ReviewDataset(torch.utils.data.Dataset):
    def _init_(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def _getitem_(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def _len_(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_encodings, train_labels)
val_dataset = ReviewDataset(val_encodings, val_labels)

# === MODEL ===
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).to(device)

# === TRAINING ARGUMENTS ===
training_args = TrainingArguments(
    output_dir="./bert_output",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=10,
)

# === METRICS ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# === TRAINER ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# === TRAIN & EVALUATE ===
trainer.train()
preds_raw = trainer.predict(val_dataset).predictions
preds = torch.argmax(torch.tensor(preds_raw), dim=1).numpy()

val_accuracy = accuracy_score(val_labels, preds)
print("\nClassification Report:\n", classification_report(val_labels, preds, target_names=["Negative", "Neutral", "Positive"]))
print(f"\n✅ Validation Accuracy: {val_accuracy:.4f}")

# === GRADIO INTERFACE ===
def analyze_place_sentiment(place_name):
    place_reviews = df[df["place_name"].str.lower() == place_name.lower()]["review_text"].tolist()
    if not place_reviews:
        return f"No reviews found for '{place_name}'.", None

    inputs = tokenizer(place_reviews[:50], return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    total = len(preds)
    pos = np.sum(preds == 2)
    neu = np.sum(preds == 1)
    neg = np.sum(preds == 0)

    result_text = (
        f"Sentiment Analysis for: {place_name}\n"
        f"Total Reviews: {total}\n\n"
        f"Positive: {pos/total*100:.2f}%\n"
        f"Neutral: {neu/total*100:.2f}%\n"
        f"Negative: {neg/total*100:.2f}%\n\n"
        f"Overall Sentiment: {'POSITIVE' if pos > max(neg, neu) else 'NEUTRAL' if neu > max(pos, neg) else 'NEGATIVE'}\n\n"
        f"(Validation Accuracy: {val_accuracy:.2%})"
    )

    # Pie Chart
    fig, ax = plt.subplots()
    ax.pie([pos, neu, neg], labels=["Positive", "Neutral", "Negative"], autopct="%1.1f%%", startangle=140,
           colors=["#A2D9CE", "#F9E79F", "#F5B7B1"])
    ax.axis("equal")
    plt.title(f"Sentiment for {place_name}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return result_text, img

iface = gr.Interface(
    fn=analyze_place_sentiment,
    inputs="text",
    outputs=["text", "image"],
    title="Touristic Place Sentiment Analyzer",
    description="Enter a place name to view sentiment analysis from user reviews."
)

try:
    iface.launch(share=True)
except Exception as e:
    print("⚠ Gradio launch error:", e)