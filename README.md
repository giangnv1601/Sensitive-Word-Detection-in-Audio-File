# Sensitive-Word-Detection-in-Audio-File
## In this thesis, we focus on researching and applying convolutional neural networks to the problem of detecting sensitive words in audio files. The goal of this study is to develop an automated system capable of efficiently and accurately detecting sensitive words in audio content. This system will support media platforms and online services in monitoring and managing content, contributing to creating a safe and healthy environment for users.
### Dataset
- We use the Speech Commands v2 dataset from Google, containing 35 keywords.
- Each audio file is 1 second long at 16kHz.

| Nhãn      | Backward | Bed  | Bird  | Cat   | Dog  | Down | Eight | Five  | Forward  | For  | Go    | Happy | House | Learn | Left  | Marvin | Sheila | Six   | Stop  | Three | Tree  | Two   | Up    | Visual | Off    | On    | Right | Seven | Zero  | No    | One   | Yes    | Follow | Nine  | Wow   | Silence |
|-----------|----------|------|-------|-------|------|------|-------|-------|----------|------|-------|-------|-------|-------|-------|--------|--------|-------|-------|-------|-------|-------|-------|--------|--------|-------|-------|-------|-------|-------|-------|--------|--------|-------|-------|---------|
| Số file   | 1664     | 2014 | 2064  | 2031  | 2128 | 3917 | 3787  | 4052  | 1557     | 3728 | 3880  | 2054  | 2113  | 1575  | 3801  | 2100   | 2022   | 3860  | 3872  | 3727  | 1759  | 3880  | 3723  | 1592   | 3745   | 3845  | 3778  | 3998  | 4052  | 3941  | 3890  | 4044   | 1579   | 3934  | 2123  | 385     |

### 🚀 Features
- Preprocessing pipeline for raw `.wav` audio:
  - Resample → Mono → Pad/Truncate → Mel-spectrogram.
- Custom **AudioCNN** built with PyTorch.
- Training with checkpoint saving.
- Inference on single audio files.
- Visualization of spectrogram with predicted label.

### ⚙️ Audio Preprocessing Pipeline
1. **Load audio file** – Read raw `.wav` files.  
2. **Channel conversion** – Convert all audio files to **mono (1 channel)**.  
3. **Resampling** – Convert audio to a **16 kHz** sample rate.  
4. **Length normalization** – Pad or truncate audio files to **1 second (16,000 samples)**.  
5. **Time-shifting augmentation** – Apply random temporal shifts (up to 20% of audio length) for data augmentation.  
6. **Mel Spectrogram conversion** – Transform audio signals into **64-band Mel Spectrogram**.  
7. **SpecAugment (time & frequency masking)** – Apply time and frequency masking on Mel Spectrogram for augmentation.

### 🧠 Model Architecture
Defined in `model.py`:

- 4 convolutional blocks (`Conv2d + BatchNorm + LeakyReLU + MaxPool`)  
- Fully connected layers with dropout  
- Output layer for **35 classes**  

Input: **(1, 64, 32)** Mel-spectrogram  
Output: **class probabilities**  

### 📈 Results




