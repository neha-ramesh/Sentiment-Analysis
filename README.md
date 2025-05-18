# 🎙️ Speech Emotion Analyzer

A deep learning-based system for recognizing human emotions from speech 🎧.  
This project uses **LSTM** and **Variational Autoencoder (VAE)** models to classify emotions such as:

> 😄 Happy &nbsp;&nbsp; 😢 Sad &nbsp;&nbsp; 😠 Angry &nbsp;&nbsp; 😨 Fearful &nbsp;&nbsp; 😲 Surprise &nbsp;&nbsp; 🤢 Disgust

---

## 📂 Datasets Used

We use four well-known speech emotion datasets:

| Dataset      | Samples | Emotions Included                                             | Gender    |
|--------------|---------|--------------------------------------------------------------|-----------|
| **RAVDESS**  | 2452    | Happy, Sad, Angry, Fearful, Surprise, Disgust                | Male, Female |
| **SAVEE**    | 500     | Happy, Sad, Angry, Fearful, Surprise, Disgust                | Male      |
| **TESS**     | 2800    | Happy, Sad, Angry, Fearful, Surprise, Disgust, Neutral       | Female    |
| **CREMA-D**  | 7442    | Happy, Sad, Angry, Fearful, Disgust, Neutral                 | Male, Female |
---

## 🎯 Features Extracted

From each audio sample, we extract meaningful features:

- 🎼 **MFCC** – Captures timbre and tone
- 🎹 **Chroma** – Pitch-based variations
- 🔁 **ZCR** – Measures voice noisiness
- ⚡ **TEO** – Detects energy bursts
- 📈 **Pitch** – Indicates tone height
- 🔊 **Energy** – Measures loudness

> All features are normalized and stored in a dataframe for model input.
---

## 🧠 Model Architectures

### 🔁 LSTM

An LSTM model is used to capture **temporal patterns** in speech:

- `128` LSTM units → Dropout → Dense layer
- Softmax activation for multiclass emotion classification
- Trained on extracted features to learn emotion sequences over time

### 🌌 VAE + Classifier

A **Variational Autoencoder (VAE)** reduces dimensionality and learns compressed emotion representations:

- Latent vectors from the VAE encoder are passed into an MLP classifier
- Efficient at emotion classification with fewer parameters
- Helps handle noise and redundancy in features

---

## 📊 Model Evaluation

### 📈 LSTM Model
- ✅ **Training Accuracy:** `88.75%`
- ✅ **Validation Accuracy:** `91.00%`
- 📉 **Loss:** Steady decrease → Proper learning
- ✅ **Confusion Matrix:** Strong on `happy`, `sad`, `angry`
- 📋 **Classification Report:**
  - Precision, Recall, F1-Score: **> 85%**
  - Weighted F1 Score: **~91%**

### 🌌 VAE + Classifier
- ✅ **Training Accuracy:** `82.50%`
- ✅ **Validation Accuracy:** `84.75%`
- 🌐 Performs well on `neutral` and `sad`
- ⚠️ Slight confusion between `angry` and `disgust`

---

## 🚀 Technologies Used
- Python, NumPy, Pandas, Librosa
- TensorFlow / Keras
- Matplotlib & Seaborn (for visualizations)
---
