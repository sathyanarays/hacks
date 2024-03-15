## Download from https://pjreddie.com/projects/mnist-in-csv/

import numpy as np
TRAIN_FILENAME = "mnist_train.csv"
TEST_FILENAME = "mnist_test.csv"
LEARNING_RATE = 0.01
EPOCHS = 20

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

def show_learning(epoch_no, train_acc, test_acc):
    print('epoch no:', epoch_no, ', train_acc: ', train_acc, ', test_acc', test_acc)

def forward_pass(x):
    global hidden_layer_y
    global output_layer_y
    # Activation function for hidden layer
    for i,w in enumerate(hidden_layer_w):
        z = np.dot(w, x)
        hidden_layer_y[i] = np.tanh(z)
    hidden_output_array = np.concatenate(
        (np.array([1.0]), hidden_layer_y)
    )
    # Activation function for output layer
    for i, w in enumerate(output_layer_w):
        z = np.dot(w, hidden_output_array)
        output_layer_y[i] = 1.0 / (1.0 + np.exp(-z))

def backward_pass(y_truth):
    global hidden_layer_error
    global output_layer_error
    # Backpropogate for each  output neuron
    # and create array of all output neuron errors
    for i, y in enumerate(output_layer_y):
        error_prime = -(y_truth[i] - y)
        derivative = y * (1.0 - y)
        output_layer_error[i] = error_prime * derivative
    for i, y in enumerate(hidden_layer_y):
        # Create array weights connecting the output of
        # hidden neuron i to neurons in the output layer
        error_weights = []
        for w in output_layer_w:
            error_weights.append(w[i+1])
        error_weight_array = np.array(error_weights)
        # Backpropogate error for hidden neuron
        derivative = 1.0 - y**2
        weighted_error = np.dot(error_weight_array, output_layer_error)
        hidden_layer_error[i] = weighted_error * derivative

def adjust_weights(x):
    global output_layer_w
    global hidden_layer_w
    for i, error in enumerate(hidden_layer_error):
        hidden_layer_w[i] -= (x * LEARNING_RATE * error)
        hidden_output_array = np.concatenate((np.array([1.0]), hidden_layer_y))
        for i, error in enumerate(output_layer_error):
            output_layer_w[i] -= (hidden_output_array * LEARNING_RATE * error)

for i in range(EPOCHS):
    np.random.shuffle(index_list)
    correct_training_results = 0
    for j in index_list:
        x = np.concatenate((np.array([1.0]), x_train[j]))
        forward_pass(x)
        if output_layer_y.argmax() == y_train[j].argmax():
            correct_training_results += 1
        backward_pass(y_train[j])
        adjust_weights(x)
    
    correct_test_results = 0
    for j in range(len(x_test)):
        x = np.concatenate((np.array([1.0]), x_test[j]))
        forward_pass(x)
        if output_layer_y.argmax() == y_test[j].argmax():
            correct_test_results += 1

    show_learning(i, correct_training_results/len(x_train), correct_test_results / len(x_test))