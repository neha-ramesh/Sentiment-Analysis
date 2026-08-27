# 🎙️ Speech Emotion Recognition with BiLSTM, Conformer & Ensemble Fusion

A deep learning-based **speech emotion recognition (SER)** system that combines complementary temporal and spectral representations using **BiLSTM-Xception** and **Conformer** architectures, followed by multiple ensemble fusion strategies.

The system recognizes **7 emotions** from speech:

> 😠 Angry · 🤢 Disgust · 😨 Fearful · 😄 Happy · 😐 Neutral · 😢 Sad · 😲 Surprise

---

## 📂 Datasets Used

The project combines four widely used speech emotion datasets:

| Dataset     | Samples | Emotions                                                     | Speakers     |
| ----------- | ------: | ------------------------------------------------------------ | ------------ |
| **RAVDESS** |   2,452 | Angry, Disgust, Fearful, Happy, Neutral, Sad, Surprise, Calm | Male, Female |
| **TESS**    |   2,800 | Angry, Disgust, Fearful, Happy, Neutral, Sad, Surprise       | Female       |
| **CREMA-D** |   7,442 | Angry, Disgust, Fearful, Happy, Neutral, Sad                 | Male, Female |
| **SAVEE**   |     480 | Angry, Disgust, Fearful, Happy, Neutral, Sad, Surprise       | Male         |

### Dataset Processing

* RAVDESS **Calm** samples are removed to maintain a 7-class classification setup.
* `surprised` is standardized to the label `surprise`.
* Audio samples are converted to **mono** and resampled to **16 kHz**.
* Each audio clip is fixed to **3 seconds** through trimming or zero-padding.
* Failed or unusable audio samples are excluded during preprocessing.
* Class weights are calculated to address class imbalance.

---

## 🎧 Audio Preprocessing

Each audio sample passes through a preprocessing pipeline before feature extraction.

### Processing Steps

1. **Resampling**

   * Sample rate: `16,000 Hz`
   * Mono audio

2. **Fixed Duration**

   * Duration: `3 seconds`
   * Audio is trimmed or padded to `48,000` samples.

3. **Bandpass Filtering**

   * Butterworth bandpass filter
   * Frequency range: `300–7,500 Hz`

4. **RMS Normalization**

   * Normalizes the signal based on its root mean square energy.

5. **Voice Activity Detection**

   * Removes low-energy/silent portions using an energy threshold.
   * The resulting signal is again fixed to 3 seconds.

---

## 🔄 Data Augmentation

Augmentation is applied **only to the training data** to improve model robustness.

Each training sample can receive one augmented version using randomly applied transformations:

* 🎵 **Pitch Shift**

  * Randomly shifted by `-2, -1, +1, +2` semitones.

* ⏩ **Time Stretch**

  * Random stretch factor between `0.85` and `1.15`.

* 📢 **Noise Injection**

  * Adds low-level Gaussian noise.

The training data is divided into **10 batches**, augmented separately, and then merged into the final augmented training set.

---

## 🧮 Feature Extraction

Instead of conventional handcrafted features such as Chroma, ZCR, Pitch and Energy, the current system uses **MFCC-based wavelet representations**.

### 1. MFCC

**Mel-Frequency Cepstral Coefficients (MFCCs)** capture spectral characteristics of speech.

* `40` MFCC coefficients
* Transposed into a temporal representation.

### 2. MFDWC

**MFCC + Discrete Wavelet Coefficients (MFDWC)** applies a Discrete Wavelet Transform using the `db4` wavelet to each MFCC coefficient.

### 3. MFWPC

**MFCC + Wavelet Packet Coefficients (MFWPC)** applies Wavelet Packet decomposition using:

* Wavelet: `db4`
* Maximum level: `2`

### Feature Combinations

The pipeline generates:

* `MFCC`
* `MFDWC`
* `MFWPC`
* `MFCC + MFDWC`
* `MFDWC + MFWPC`
* `ALL`

The model evaluation pipeline uses:

> **MFCC + MFDWC**

resulting in an input representation of:

> **40 time steps × 90 features**

---

# 🧠 Model Architecture

The final system consists of **two complementary deep learning branches**:

```text
                    ┌─────────────────────┐
                    │   Audio Features    │
                    │    (40 × 90)        │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
   ┌─────────────────────┐          ┌─────────────────────┐
   │   BiLSTM-Xception   │          │      Conformer      │
   │      Branch         │          │       Branch        │
   └──────────┬──────────┘          └──────────┬──────────┘
              │                                 │
              ▼                                 ▼
       128-D Features                    128-D Features
              │                                 │
              └────────────────┬────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Ensemble Fusion     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 7-Class Classifier   │
                    └─────────────────────┘
```

---

## 🔁 BiLSTM-Xception Branch

The first branch combines bidirectional temporal modeling with depthwise-separable convolution.

### Architecture

* **Bidirectional LSTM**

  * `128` LSTM units
  * `return_sequences=True`

* **Highway Connection**

  * Projects the input to `256` dimensions.
  * Uses transform and carry gates to combine the original representation with the BiLSTM output.

* **Xception Blocks**

  * Two 1D Xception-style blocks.
  * Each block uses `SeparableConv1D`, Batch Normalization and ReLU.
  * Filters:

    * Block 1: `64`
    * Block 2: `128`

* **Global Average Pooling**

* **128-dimensional feature layer**

The 128-dimensional feature representation is passed to the ensemble module.

---

## 🧩 Conformer Branch

The second branch uses a lightweight **Conformer architecture** to model both local convolutional patterns and long-range temporal dependencies.

### Architecture

* Input projection:

  * `90 → 128` dimensions
* Positional encoding
* **2 Conformer blocks**
* Multi-Head Self-Attention

  * `4` attention heads
* Feed-forward modules
* Convolution modules
* **Mish activation**
* **Squeeze-and-Excitation (SE) blocks**
* 128-dimensional feature layer
* Final 7-class classifier

The Conformer branch is trained using a **CTC-based objective** with class-weight scaling.

---

# 🔗 Ensemble Fusion

The 128-dimensional representations from the BiLSTM-Xception and Conformer branches are normalized using **Layer Normalization** before fusion.

Five fusion strategies are evaluated:

### 1. Add Fusion

```text
fused = features₁ + features₂
```

### 2. Concatenation Fusion

```text
fused = [features₁ ; features₂]
```

Produces a 256-dimensional representation.

### 3. Weighted Add Fusion

Learns weights for the two feature representations before adding them.

### 4. Weighted Concatenation Fusion

Learns separate weights for both branches and concatenates the weighted representations.

### 5. Attention Scalar Fusion

Learns a scalar attention value to dynamically balance the contribution of the two branches.

The fused representation is passed through:

* Batch Normalization
* Dense layer: `64` units
* ReLU
* Dropout
* Final `7`-class output layer

---

# 📊 Model Evaluation

The final evaluation uses:

* Accuracy
* Macro F1 Score
* Weighted F1 Score
* Matthews Correlation Coefficient (MCC)
* Cohen's Kappa
* Classification Report
* Concordance Correlation Coefficient (CCC)

### Dataset Split

The main model evaluation pipeline uses:

| Split      | Samples |
| ---------- | ------: |
| Training   |  13,788 |
| Validation |   1,533 |
| Test       |   3,831 |

---

## 🏆 Fusion Results

| Fusion Method         |   Accuracy |   Macro F1 | Weighted F1 |        MCC | Cohen's Kappa |
| --------------------- | ---------: | ---------: | ----------: | ---------: | ------------: |
| **Add**               |     70.48% |     0.7211 |      0.7038 |     0.6521 |        0.6518 |
| **Concat**            |     70.35% |     0.7201 |      0.7023 |     0.6505 |        0.6503 |
| **Weighted Add**      |     70.19% |     0.7179 |      0.7005 |     0.6491 |        0.6485 |
| **Weighted Concat** ⭐ | **70.66%** | **0.7249** |  **0.7057** | **0.6541** |    **0.6539** |
| **Attention Scalar**  |     70.48% |     0.7217 |      0.7038 |     0.6521 |        0.6518 |

### 🥇 Best Model

**Weighted Concatenation Fusion** achieves the best overall test performance:

* 🎯 **Accuracy:** `70.66%`
* 📈 **Macro F1:** `0.7249`
* 📈 **Weighted F1:** `0.7057`
* 📊 **MCC:** `0.6541`
* 📊 **Cohen's Kappa:** `0.6539`
* 🔗 **CCC:** `0.6999`

The best-performing fusion method is therefore:

> **BiLSTM-Xception + Conformer → Weighted Concatenation → Classifier**

---

# 🔍 Explainability

The project includes an explainability pipeline for understanding model predictions.

Two feature-attribution approaches are implemented:

### 🍃 Integrated Gradients / SHAP-style Analysis

A gradient-based attribution approach is used to estimate the contribution of individual input features to model predictions.

The implementation:

* Uses the mean of background samples as a baseline.
* Interpolates between the baseline and input.
* Computes gradients over multiple integration steps.
* Produces feature-level attribution scores.
* Falls back to perturbation-based attribution when gradients cannot be computed.

### 🍋 LIME

**LIME (Local Interpretable Model-agnostic Explanations)** is used to generate local explanations.

The LIME pipeline:

* Flattens the `40 × 90` input representation.
* Generates perturbed samples.
* Obtains predictions from the complete BiLSTM-Xception + Conformer + ensemble pipeline.
* Fits a local surrogate model.
* Identifies the most influential features for an individual prediction.

---

# 🔒 Trustworthy XAI Evaluation

The explainability pipeline evaluates the quality of explanations using several metrics.

### Faithfulness

Measured using **AOPC (Area Over the Perturbation Curve)** to evaluate whether important features identified by the explanation actually influence model predictions.

### Robustness

Evaluated using a **Local Lipschitz-based measure** to examine the stability of explanations under small perturbations.

### Sparsity

Measures how concentrated the explanation is among the input features.

### Consistency

Evaluates whether explanations remain consistent across similar inputs.

### Example Explainability Results

For the evaluated samples:

* **Average Faithfulness (AOPC):** `0.1167 ± 0.0116`
* **Average Sparsity:** `0.9994 ± 0.0000`

The explainability pipeline also reports **LIME surrogate fidelity** for individual samples.

---

# 🛡️ Robustness & Reliability Analysis

Beyond classification accuracy, the project evaluates whether the model's explanations are meaningful and stable.

The evaluation framework considers:

* Feature perturbation
* Local explanation stability
* Faithfulness
* Sparsity
* Consistency
* LIME surrogate fidelity

This provides an additional layer of analysis beyond conventional accuracy and F1-score evaluation.

---

# 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Librosa**
* **PyTorch**
* **TensorFlow / Keras**
* **Torchaudio**
* **PyWavelets**
* **Scikit-learn**
* **SHAP**
* **LIME**
* **Matplotlib**
* **Seaborn**

---

# 📁 Project Pipeline

```text
Speech Audio
     │
     ▼
Dataset Collection
(RAVDESS / TESS / CREMA-D / SAVEE)
     │
     ▼
Audio Preprocessing
 ├── Resampling → 16 kHz
 ├── 3-second fixing
 ├── Bandpass filtering
 ├── RMS normalization
 └── Voice Activity Detection
     │
     ▼
Training Data Augmentation
 ├── Pitch Shift
 ├── Time Stretch
 └── Noise Injection
     │
     ▼
Feature Extraction
 ├── MFCC
 ├── MFDWC
 └── MFWPC
     │
     ▼
MFCC + MFDWC (40 × 90)
     │
     ├───────────────────────────┐
     ▼                           ▼
BiLSTM + Highway + Xception   Conformer
     │                           │
     ▼                           ▼
128-D Features              128-D Features
     │                           │
     └──────────────┬────────────┘
                    ▼
             Ensemble Fusion
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
      Add        Concat     Weighted Fusion
                                  │
                                  ▼
                         Best: Weighted Concat
                                  │
                                  ▼
                         7-Class Prediction
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
              Model Metrics              Explainability
                                            │
                                      ┌─────┴─────┐
                                      ▼           ▼
                                  Integrated    LIME
                                  Gradients
                                      │
                                      ▼
                             Trustworthy XAI
                             ├── Faithfulness
                             ├── Robustness
                             ├── Sparsity
                             └── Consistency
```

---

## 🚀 Key Contributions

* 🎙️ Multi-dataset speech emotion recognition using **RAVDESS, TESS, CREMA-D and SAVEE**.
* 🎧 Robust audio preprocessing with filtering, normalization and voice activity detection.
* 🔄 Training-only augmentation using pitch shifting, time stretching and noise injection.
* 🧮 Multi-level speech representation using **MFCC, MFDWC and MFWPC**.
* 🧠 Hybrid **BiLSTM-Xception + Conformer** architecture.
* 🔗 Systematic comparison of **five ensemble fusion strategies**.
* 🏆 Best test accuracy of **70.66%** using **Weighted Concatenation Fusion**.
* 🔍 Explainability using **Integrated Gradients and LIME**.
* 🔒 Trustworthy XAI evaluation using **faithfulness, robustness, sparsity and consistency** metrics.
