# Word Embeddings & Next-Word Prediction with PyTorch

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-FA0F00?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 📌 Project Overview

This project implements a **word embedding and next-word prediction model** from scratch using **PyTorch**, trained on *The Time Machine* by H.G. Wells.  
The goal is to demonstrate how neural networks learn **semantic word representations** and use contextual information to generate text.

Unlike large pre-trained language models, this project focuses on **foundational NLP concepts**, including tokenization, embeddings, cosine similarity, and sequence-based prediction.

---

## 🧠 What This Project Covers

- Text preprocessing and vocabulary construction  
- Custom `Dataset` and `DataLoader` creation  
- Word embeddings using `nn.Embedding`  
- Cosine similarity between word vectors  
- Feedforward neural network for next-word prediction  
- Text generation using sliding context windows  
- Model size analysis and embedding visualization  
- GPU-accelerated training (when available)

---

## 📚 Dataset

- **Source**: *The Time Machine* by H.G. Wells (Project Gutenberg)  
- **Total words**: ~30,000  
- **Vocabulary size**: ~4,600 unique tokens  
- **Tokenization**: Word-level (lowercased, punctuation removed)

---

## 🏗️ Model Architecture

```
Input (context words)
        ↓
Embedding Layer (Vocab × Embedding Dim)
        ↓
Flattened Context Embeddings
        ↓
Fully Connected Layer (ReLU)
        ↓
Output Layer (Vocab Size)
        ↓
Log-Softmax (Next Word Probabilities)
```

**Key details:**
- Embedding dimension: `100`  
- Context window size: configurable (e.g. 6 words)  
- Loss function: `Negative Log Likelihood (NLLLoss)`  
- Optimizer: `AdamW`  
- Weight initialization: Xavier Normal  

---

## 🔍 Embeddings & Semantic Similarity

The project demonstrates how embeddings capture semantic relationships by:

- Visualizing embedding matrices  
- Tracking how embeddings change before vs after training  
- Computing **cosine similarity** between related words  

Example:
```
Cosine similarity between "time" and "machine"
↑ increases after training
```

---

## ✍️ Text Generation Example

After training, the model generates text by repeatedly predicting the most likely next word:

```
look the stained glass windows which
the stained glass windows which exultation
stained glass windows which exultation intense
glass windows which exultation intense gusts
```

This illustrates how learned embeddings and context influence predictions.

---

## 📊 Model Size & Performance

- **Total parameters**: ~1.13 million  
- **Trainable parameters**: 100%  
- **Estimated model size**: ~6 MB  
- **Training loss**: decreases steadily over epochs, indicating successful learning  

---

## 🚀 How to Run

1. Clone the repository
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```

2. Install dependencies
   ```bash
   pip install torch numpy matplotlib torchinfo
   ```

3. Open the notebook
   ```bash
   jupyter notebook
   ```

4. Run all cells to train the model and generate text

---

## 🔮 Future Improvements

- Add negative sampling or hierarchical softmax  
- Replace feedforward model with an RNN or Transformer  
- Train on larger corpora  
- Add temperature-based sampling for text generation  
- Save and reload trained embeddings  

---

## 🎯 Why This Project Matters

This project demonstrates a **deep understanding of NLP fundamentals**, not just usage of pre-trained models.  
It shows how modern language models are built from simple components: embeddings, context, and probability.

Perfect for:
- Machine learning portfolios  
- NLP coursework  
- Interview discussion projects  
- Foundational understanding before scaling to LLMs  

---

## 📜 License

MIT License
