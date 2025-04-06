# HW3_neural
Bhargav Sai Allam 700752883
# README - Deep Learning Tasks

## Overview
This project covers four deep learning tasks:
1. *Basic Autoencoder* - Learning to reconstruct images using an autoencoder.
2. *Denoising Autoencoder* - Removing noise from images using a modified autoencoder.
3. *RNN for Text Generation* - Training an LSTM model to generate text.
4. *Sentiment Classification using RNN* - Using an LSTM model to classify sentiment in movie reviews.

---
## Task 1: Implementing a Basic Autoencoder

### What was done:
- Loaded the MNIST dataset using tensorflow.keras.datasets.
- Defined an autoencoder with a *784-32-784* architecture using Dense layers.
- Compiled the model with *binary cross-entropy* loss.
- Trained the model to reconstruct input images.
- Compared original vs. reconstructed images.
- Experimented with different latent sizes (16, 64) to observe reconstruction quality.

### Key Learning:
- Smaller latent dimensions retain fewer details, affecting reconstruction quality.

---
## Task 2: Implementing a Denoising Autoencoder

### What was done:
- Modified the basic autoencoder to add *Gaussian noise* (mean=0, std=0.5) to input images.
- Ensured that the model learns to reconstruct *clean images* from noisy inputs.
- Trained the model and visualized noisy vs. reconstructed images.
- Compared performance between the basic and denoising autoencoder.
- Identified *medical imaging* as a real-world application where denoising autoencoders are useful.

### Key Learning:
- Denoising autoencoders can remove noise while preserving important details.

---
## Task 3: Implementing an RNN for Text Generation

### What was done:
- Loaded a text dataset (e.g., "Shakespeare Sonnets").
- Converted text into *sequences of characters* using one-hot encoding or embeddings.
- Built an *LSTM-based RNN* model to predict the next character.
- Trained the model and generated new text character-by-character.
- Explained the impact of *temperature scaling* on text randomness:
  - Low temperature (0.2): Predictable, repeats common patterns.
  - Medium temperature (1.0): Balanced creativity and coherence.
  - High temperature (1.5): More random and sometimes nonsensical.

### Key Learning:
- Temperature scaling controls the trade-off between predictability and randomness in text generation.

---
## Task 4: Sentiment Classification Using RNN

### What was done:
- Loaded the *IMDB sentiment dataset* using tensorflow.keras.datasets.imdb.
- Preprocessed text by *tokenization and padding sequences*.
- Trained an *LSTM-based model* to classify reviews as positive or negative.
- Generated a *confusion matrix* and a *classification report* (accuracy, precision, recall, F1-score).
- Explained why the *precision-recall tradeoff* is important:
  - High recall ensures that most positive reviews are identified.
  - High precision ensures fewer false positives (misclassifying negative reviews as positive).

### Key Learning:
- A balance between precision and recall is crucial for sentiment classification accuracy.

---
## Dependencies
- TensorFlow/Keras
- NumPy
- Matplotlib (for visualization)
- scikit-learn (for evaluation metrics)

Install dependencies using:
bash
pip install tensorflow numpy matplotlib scikit-learn


---
## Running the Code
Each task has its own Python script (or Jupyter Notebook). Simply run the respective script to execute a task.
bash
python task1_autoencoder.py
python task2_denoising_autoencoder.py
python task3_text_generation.py
python task4_sentiment_analysis.py


---
## Summary
This project demonstrates how deep learning can be applied to *image reconstruction, noise removal, text generation, and sentiment analysis* using autoencoders and LSTMs. The experiments provide insights into model performance and real-world applications.
