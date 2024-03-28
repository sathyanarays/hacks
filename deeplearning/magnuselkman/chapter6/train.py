import tensorflow as tf
import keras
from keras.utils import to_categorical
import numpy as np
import logging

tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 20
BATCH_SIZE = 1

# Load training and test datasets
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Standardize the data
mean = np.mean(train_images)
stddev = np.std(train_images)
train_images = (train_images - mean) / stddev
test_images = (test_images - mean) / stddev

# One hot encode labels
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

initializer = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(25, activation='tanh', kernel_initializer=initializer, bias_initializer='zeros'),
    keras.layers.Dense(10, activation='sigmoid', kernel_initializer=initializer, bias_initializer='zeros')
])

model.summary()

#opt = keras.optimizers.SGD(learning_rate=0.01)
opt = keras.optimizers.legacy.SGD(learning_rate=0.01)

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

# Train the model for 20 epochs
# Shuffle order
# Update weights after each example (batch_size=1)
history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)

