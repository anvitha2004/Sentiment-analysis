# 🏞️ Sentiment Analysis of Tourist Places

This project aims to analyze user reviews of Indian tourist destinations and classify them into **Positive**, **Neutral**, or **Negative** sentiments using a fine-tuned **BERT** model. An interactive interface is built using **Gradio** to allow users to explore sentiment distributions for any place in real time.

---

## 📌 Project Highlights

- 💬 Sentiment Classification using **BERT**
- 📊 Visual output with **Pie Chart**
- 🧠 Accuracy up to **95%**
- 🌐 Easy-to-use **Gradio Web Interface**
- ✅ Trained on real-world Indian tourist review dataset

---

## 🔧 Tech Stack

| Tool/Library     | Purpose                            |
|------------------|-------------------------------------|
| Python 3.10       | Core programming language           |
| pandas, numpy     | Data loading and manipulation       |
| scikit-learn      | Data splitting, evaluation metrics  |
| PyTorch + Transformers | BERT model + training             |
| matplotlib        | Sentiment distribution pie chart    |
| Gradio            | Web interface                       |
| Google Colab      | Model training with free GPU        |

---

## 🧠 Model: Fine-Tuned BERT

We used the Hugging Face `bert-base-uncased` model and fine-tuned it for 3-class classification (Negative, Neutral, Positive). 

### Training Enhancements:
- ✅ 5% Label Noise injected for robustness
- ✅ Class Balancing using upsampling
- ✅ Achieved ~95% Validation Accuracy

---

## 📂 Dataset

- Name: `indian_places_reviews_with_ratings.csv`
- Columns:
  - `place_name`
  - `review_text`
  - `sentiment_label` (1.0 = Positive, 0.5 = Neutral, 0.0 = Negative)

---

## 🚀 How to Run

### Step 1: Clone the repository

git clone https://github.com/yourusername/tourist-sentiment-analysis.git
cd tourist-sentiment-analysis

### Step 2: Install dependencies
pip install torch transformers pandas numpy scikit-learn matplotlib gradio

### Step 3: Upload dataset to the correct path
Place indian_places_reviews_with_ratings.csv in the root folder or adjust the path in the code.

### Step 4: Run the project
python app.py 
