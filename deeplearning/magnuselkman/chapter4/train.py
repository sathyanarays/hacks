## Download from https://pjreddie.com/projects/mnist-in-csv/

import numpy as np
TRAIN_FILENAME = "mnist_train.csv"
TEST_FILENAME = "mnist_test.csv"

# Function to read examples from file.
def read_file_data(filename):
    x_examples = []
    y_examples = []
    file = open(filename, 'r', encoding="utf-8")
    for line in file:
        example = line.split(',')
        y_examples.append(example[0])
        x_examples.append(example[1:])
    file.close()
    x_examples = np.array(x_examples, dtype=float)
    y_examples = np.array(y_examples, dtype=np.int32)
    return x_examples, y_examples

def read_mnist():
    # Read files.
    train_images, train_labels = read_file_data(TRAIN_FILENAME)
    test_images, test_labels = read_file_data(TEST_FILENAME)

    # Print dimensions.
    print('dimensions of train_images: ', train_images.shape)
    print('dimensions of train_labels: ', train_labels.shape)
    print('dimensions of test_images: ', test_images.shape)
    print('dimensions of test_images: ', test_labels.shape)

    # Print one training example.
    print('label for first training example: ', train_labels[0])
    print('---beginning of pattern for first training example---')
    line = train_images[0]
    for i in range(len(line)):
        if line[i] > 0:
            print('*', end = ' ')
        else:
            print(' ', end = ' ')
        if (i % 28) == 0:
            print('')
    print('')
    print('---end of pattern for first training example---')

    x_train = train_images.reshape(60000, 784)
    mean = np.mean(x_train)
    stddev = np.std(x_train)
    x_train = (x_train - mean) / stddev
    x_test = test_images.reshape(10000, 784)
    x_test = (x_test - mean) / stddev

    # One-hot encoded output
    y_train = np.zeros((60000,10))
    y_test = np.zeros((10000,10))

    for i,y in enumerate(train_labels):
        y_train[i][y] = 1
    for i,y in enumerate(test_labels):
        y_test[i][y] = 1
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = read_mnist()
index_list = list(range(len(x_train)))

def layer_w(neuron_count, input_count):
    weights = np.zeros((neuron_count, input_count+1))
    for i in range(neuron_count):
        for j in range(1, (input_count+1)):
            weights[i][j] = np.random.uniform(-0.1, 0.1)
    return weights

# Declare matrics and vectors representing the neurons
hidden_layer_w = layer_w(25,784)
hidden_layer_y = np.zeros(25)
hidden_layer_error = np.zeros(25)

output_layer_w = layer_w(10,25)
output_layer_y = np.zeros(10)
output_layer_error = np.zeros(10)