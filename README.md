# 🇬🇧 English → 🇩🇪 German Neural Machine Translation  
### Sequence-to-Sequence LSTM Model (Keras | TensorFlow)

---

## 📌 Project Overview

This project implements a **Neural Machine Translation (NMT)** system that translates **English sentences into German** using a **Sequence-to-Sequence (Seq2Seq) Encoder–Decoder LSTM model**.

The model is trained on **80,000 parallel English–German sentence pairs** from the **WMT 2014 dataset**.  
It uses:

- Text cleaning (lowercasing, punctuation removal)  
- Tokenization and word–index mapping  
- Sequence padding  
- Encoder–Decoder LSTM with Embedding layers  
- Greedy decoding for translation inference  

---

## 🧠 Model Architecture

**Encoder**

- Input: padded English sentence  
- `Embedding(eng_vocab, embedding_dim, mask_zero=True)`  
- `LSTM(latent_dim, return_state=True)`  
- Outputs: encoder hidden state `(h)` and cell state `(c)`

**Decoder**

- Input: padded German sentence shifted by one token (`decoder_input_data`)  
- `Embedding(ger_vocab, embedding_dim, mask_zero=True)`  
- `LSTM(latent_dim, return_sequences=True, return_state=True, initial_state=[h, c])`  
- `Dense(ger_vocab, activation='softmax')` → predicts next German word

**Loss**

- `sparse_categorical_crossentropy`  

**Optimizer**

- `adam`

---

## 📂 Dataset

- **File path** (Kaggle):

```python
file_path = "/kaggle/input/wmt-2014-english-german/wmt14_translate_de-en_train.csv"
