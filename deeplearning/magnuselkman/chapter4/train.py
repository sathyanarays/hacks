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