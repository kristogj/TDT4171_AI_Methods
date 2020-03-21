import pickle
import tensorflow as tf
import numpy as np

# Load data
try:
    data = pickle.load(open("./data/keras-data.pickle", "rb"))
except FileNotFoundError:
    raise FileNotFoundError("Place data files in ./data/keras-data.pickle")

x_train, y_train = data["x_train"], np.array(data["y_train"])
x_test, y_test = data["x_test"], np.array(data["y_test"])

VOCAB_SIZE = data["vocab_size"]
MAX_LENGTH = data["max_length"] // 8

# Preprocess into equal sized sequence length
x_train = tf.keras.preprocessing.sequence.pad_sequences(
    x_train, maxlen=MAX_LENGTH, dtype='int32', padding='post', truncating='post', value=0)
x_test = tf.keras.preprocessing.sequence.pad_sequences(
    x_test, maxlen=MAX_LENGTH, dtype='int32', padding='post', truncating='post', value=0)

# Initialize the model
EMBEDDING_DIM = 512
HIDDEN_DIM = 256
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH))
model.add(tf.keras.layers.LSTM(HIDDEN_DIM))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# Compile and fit
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=3, batch_size=256, verbose=1)

# Evaluate prediction on test data
loss, acc = model.evaluate(x_test, y_test, verbose=1)
