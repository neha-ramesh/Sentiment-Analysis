🎙️ Speech Emotion Analyzer 
A deep learning-based system for recognizing human emotions from speech. This project uses LSTM neural network and Variational Autoencoder (VAE) to classify emotions such as happy, sad, angry, calm, etc., from audio files.

📂 Datasets Used
This project uses well-known emotion datasets:

RAVDESS (2452 samples)

SAVEE (500 samples)

TESS (2800 samples)

Each dataset includes labeled audio clips corresponding to different emotions such as:

Happy

Sad

Angry

Fearful

Surprise

Disgust

🎯 Features Extracted
Features are extracted from each audio file and stored in a dataframe:

MFCC (Mel Frequency Cepstral Coefficients) – Tone, timbre

Chroma – Pitch variation

Zero Crossing Rate (ZCR) – Noise measurement

Teager Energy Operator (TEO) – Sudden energy bursts

Pitch – Frequency

Energy – Loudness

These features are statistically distributed and normalized to optimize learning.

🧠 Model Architectures
🔁 LSTM
An LSTM model is used to capture the temporal patterns in speech features. It includes:

128 LSTM units, followed by dropout and dense layers

Softmax output for multiclass emotion classification

Trained on MFCC and other features to learn sequential dependencies in speech

🌌 VAE + Classifier
A Variational Autoencoder (VAE) is used to reduce feature dimensions and extract compact emotional representations. These compressed features are then fed into a simple classifier (MLP) to predict emotions efficiently.

📊 Model Evaluation
LSTM Model:
Training Accuracy: 88.75%

Validation Accuracy: 91.00%

Loss: Decreases steadily, indicating proper learning

Confusion Matrix: Shows strong performance on happy, sad, and angry emotions

Classification Report:

Precision, Recall, F1-Score: All above 85% for major classes

Weighted Average F1-Score: ~91%

VAE + Classifier:
Training Accuracy: 82.50%

Validation Accuracy: 84.75%

Used compressed latent vectors from VAE for classification

Performed well on neutral and sad classes

Slight confusion between angry and disgust
