### Question 1 :
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten images
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Define latent space dimension
latent_dim = 32  # Modify this to 16, 64 for analysis

# Define encoder
input_img = Input(shape=(784,))
encoded = Dense(latent_dim, activation='relu')(input_img)

# Define decoder
decoded = Dense(784, activation='sigmoid')(encoded)

# Define autoencoder model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Generate reconstructed images
reconstructed = autoencoder.predict(x_test)

# Plot original vs reconstructed images
def plot_images(original, reconstructed, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        
        # Reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()

plot_images(x_test, reconstructed)

### Question 2 :
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten images
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Add Gaussian noise to input images
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Clip values to be in valid range
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Define latent space dimension
latent_dim = 32

# Define encoder
input_img = Input(shape=(784,))
encoded = Dense(latent_dim, activation='relu')(input_img)

# Define decoder
decoded = Dense(784, activation='sigmoid')(encoded)

# Define denoising autoencoder model
denoising_autoencoder = Model(input_img, decoded)
denoising_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
denoising_autoencoder.fit(x_train_noisy, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test_noisy, x_test))

# Generate reconstructed images
reconstructed = denoising_autoencoder.predict(x_test_noisy)

# Plot noisy vs reconstructed images
def plot_images(original, noisy, reconstructed, n=10):
    plt.figure(figsize=(20, 6))
    for i in range(n):
        # Noisy images
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(noisy[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        
        # Original images
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        
        # Reconstructed images
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()

plot_images(x_test, x_test_noisy, reconstructed)
### Question 3:
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
import numpy as np
import random
import sys

# Load text dataset (Example: Shakespeare Sonnets)
path = tf.keras.utils.get_file('shakespeare.txt', 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
text = open(path, 'r', encoding='utf-8').read().lower()

# Create character mappings
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Convert text into sequences
seq_length = 40
step = 3
sequences = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    sequences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])

x = np.zeros((len(sequences), seq_length, len(chars)), dtype=np.float32)
y = np.zeros((len(sequences), len(chars)), dtype=np.float32)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        if char in char_to_idx:
            x[i, t, char_to_idx[char]] = 1.0
    if next_chars[i] in char_to_idx:
        y[i, char_to_idx[next_chars[i]]] = 1.0

# Define LSTM-based RNN model
input_layer = Input(shape=(seq_length, len(chars)))
lstm_layer1 = LSTM(128, return_sequences=True)(input_layer)
lstm_layer2 = LSTM(128)(lstm_layer1)
output_layer = Dense(len(chars), activation='softmax')(lstm_layer2)

model = Model(input_layer, output_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Ensure x and y are not empty
if len(x) > 0 and len(y) > 0:
    model.fit(x, y, batch_size=128, epochs=10)
else:
    print("Error: Training data is empty. Check text processing.")

# Text generation function
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature  # Avoid log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(seed_text, length=400, temperature=1.0):
    generated = seed_text
    for _ in range(length):
        sampled = np.zeros((1, seq_length, len(chars)))
        for t, char in enumerate(seed_text):
            if char in char_to_idx:
                sampled[0, t, char_to_idx[char]] = 1.0
        
        preds = model.predict(sampled, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = idx_to_char[next_index]
        
        generated += next_char
        seed_text = seed_text[1:] + next_char
    
    return generated

# Generate text with different temperatures
seed_text = "shall i compare thee to a summer's day"
print("Generated Text at Temperature 0.2:")
print(generate_text(seed_text, temperature=0.2))
print("\nGenerated Text at Temperature 1.0:")
print(generate_text(seed_text, temperature=1.0))
print("\nGenerated Text at Temperature 1.5:")
print(generate_text(seed_text, temperature=1.5))
### Question 4
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 1. Load the IMDB sentiment dataset
num_words = 10000  # Consider only the top 10,000 most frequent words
maxlen = 200       # Maximum sequence length

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# 2. Preprocess the text data
# Pad sequences to ensure uniform length
x_train_padded = pad_sequences(x_train, maxlen=maxlen)
x_test_padded = pad_sequences(x_test, maxlen=maxlen)

# 3. Train an LSTM-based model
embedding_dim = 128
lstm_units = 128

model = Sequential([
    Embedding(num_words, embedding_dim, input_length=maxlen),
    LSTM(lstm_units),
    Dense(1, activation='sigmoid')  # Output layer with sigmoid for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 5
batch_size = 128

history = model.fit(x_train_padded, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# 4. Generate confusion matrix and classification report
# Make predictions on the test set
y_pred_probs = model.predict(x_test_padded)
y_pred = np.round(y_pred_probs).astype(int)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Generate classification report
cr = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(cr)

# 5. Interpret why precision-recall tradeoff is important in sentiment classification.

print("\nInterpretation of Precision-Recall Tradeoff in Sentiment Classification:")
print("""
In sentiment classification, the precision-recall tradeoff highlights the balance
between the accuracy of positive predictions (precision) and the ability to
identify all actual positive instances (recall).

Consider a scenario where we want to identify positive reviews for a product.

High Precision: If our model has high precision, it means that when it predicts
a review as positive, it is very likely to be actually positive. This is important
to avoid falsely promoting negative reviews. However, to achieve high precision,
the model might be very conservative and miss some actual positive reviews
(resulting in lower recall).

High Recall: If our model has high recall, it means that it identifies a large
proportion of all the actual positive reviews. This is important to ensure that
we don't miss many positive opinions. However, to achieve high recall, the model
might be more liberal in its positive predictions, leading to some negative
reviews being incorrectly classified as positive (resulting in lower precision).

The importance of the tradeoff depends on the specific application:

- For applications where falsely identifying a negative sentiment as positive
  has significant consequences (e.g., flagging potentially harmful content),
  high precision might be prioritized.

- For applications where missing positive sentiment is more costly (e.g.,
  identifying enthusiastic customers), high recall might be prioritized.

The F1-score, which is the harmonic mean of precision and recall, provides a
single metric to balance both aspects. Choosing the right balance depends on the
specific goals and costs associated with false positives and false negatives in
the sentiment classification task.
""")