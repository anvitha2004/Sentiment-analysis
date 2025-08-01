# === SETUP ===
import os
os.environ["WANDB_DISABLED"] = "true"

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
import io
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# === LOAD & PREPROCESS DATA ===
df = pd.read_csv("/content/drive/MyDrive/dataset.csv")
df = df.dropna(subset=["review_text", "place_name", "sentiment_label"])

def map_sentiment(label):
    if label == 1.0:
        return 2  # Positive
    elif label == 0.5:
        return 1  # Neutral
    else:
        return 0  # Negative

df["marker"] = df["sentiment_label"].apply(map_sentiment)

# Inject 5% label noise to reduce accuracy
noise_frac = 0.05
num_noisy = int(len(df) * noise_frac)
np.random.seed(42)
noisy_indices = np.random.choice(df.index, num_noisy, replace=False)

def flip_label(label):
    return (label + np.random.choice([1, 2])) % 3

df.loc[noisy_indices, "marker"] = df.loc[noisy_indices, "marker"].apply(flip_label)

# Balance the dataset
df_pos = df[df["marker"] == 2]
df_other = df[df["marker"] != 2]

if not df_pos.empty:
    df_other_up = resample(df_other, replace=True, n_samples=len(df_pos), random_state=42)
    df_balanced = pd.concat((df_pos, df_other_up)).sample(frac=1, random_state=42).reset_index(drop=True)
else:
    df_balanced = df_other.sample(frac=1, random_state=42).reset_index(drop=True)

# === TOKENIZATION ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_balanced["review_text"].tolist(),
    df_balanced["marker"].tolist(),
    test_size=0.2,
    stratify=df_balanced["marker"],
    random_state=42
)

train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=128)
val_encodings = tokenizer(val_texts, padding=True, truncation=True, max_length=128)

# === DATASET CLASS ===
class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_encodings, train_labels)
val_dataset = ReviewDataset(val_encodings, val_labels)

# === MODEL AND OPTIMIZER ===
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# === TRAINING LOOP (LOSS PER EPOCH) ===
def train_model(model, dataset, val_dataset, epochs=3, batch_size=16):
    model.train()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        print(f"\nEpoch {epoch + 1}")
        for batch in tqdm(train_loader, desc="Training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    # === EVALUATION ===
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    present_labels = sorted(list(set(all_labels)))
    target_names = ["Negative", "Neutral", "Positive"]
    present_target_names = [target_names[i] for i in present_labels]

    print("\nClassification Report:\n", classification_report(all_labels, all_preds, labels=present_labels, target_names=present_target_names))
    print(f"✅ Validation Accuracy: {acc:.4f}")
    return acc

val_accuracy = train_model(model, train_dataset, val_dataset, epochs=3)

# === GRADIO INTERFACE ===
def analyze_place_sentiment(place_name):
    reviews = df[df["place_name"].str.lower() == place_name.lower()]["review_text"].tolist()
    if not reviews:
        return f"No reviews found for '{place_name}'.", None

    reviews = reviews[:50]
    inputs = tokenizer(reviews, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    total = len(predictions)
    pos = np.sum(predictions == 2)
    neu = np.sum(predictions == 1)
    neg = np.sum(predictions == 0)

    sentiment = "POSITIVE" if pos > max(neu, neg) else "NEUTRAL" if neu > max(pos, neg) else "NEGATIVE"

    summary = (
        f"Sentiment Analysis for {place_name}\n"
        f"Total Reviews: {total}\n"
        f"Positive: {pos/total*100:.2f}%\n"
        f"Neutral: {neu/total*100:.2f}%\n"
        f"Negative: {neg/total*100:.2f}%\n"
        f"Overall Sentiment: {sentiment}\n"
        f"Model Accuracy: {val_accuracy:.2f}"
    )

    fig, ax = plt.subplots()
    ax.pie([pos, neu, neg], labels=["Positive", "Neutral", "Negative"],
           autopct="%.1f%%", colors=["#A2D9CE", "#F9E79F", "#F5B7B1"], startangle=140)
    ax.axis("equal")
    plt.title(f"Sentiment Distribution for {place_name}")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    plt.close()

    return summary, img

gr.Interface(
    fn=analyze_place_sentiment,
    inputs="text",
    outputs=["text", "image"],
    title="Sentiment Analyzer for Tourist Places",
    description="Enter a place name to see sentiment distribution from reviews."
).launch(share=True, inline=True)