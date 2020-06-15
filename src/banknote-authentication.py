import numpy as np
import matplotlib.pyplot as plt
np.random.seed(700)

NOTES_FILE = "C:\\Users\\Shan\\Downloads\\data_banknote_authentication.txt"
banknote_data = np.genfromtxt(NOTES_FILE, delimiter=',')
np.random.shuffle(banknote_data)
banknote_data = banknote_data.T

print(f'Min: {banknote_data.min(axis=1)} \n' \
    f'Max: {banknote_data.max(axis=1)}')

train_split = banknote_data[:, :823]
validation_split = banknote_data[:, 823:1097]
test_split = banknote_data[:, 1097:1372]

def print_label_count(dataset_labels):
    class_0_labels = np.count_nonzero(dataset_labels == 0)
    class_1_labels = np.count_nonzero(dataset_labels == 1)
    print(f'Class 0 labels: {class_0_labels}. Class 1 labels: {class_1_labels}. Total labels: {dataset_labels.shape[0]}')    

print_label_count(train_split[-1, :])
print_label_count(validation_split[-1, :])
print_label_count(test_split[-1, :])

def convert_split_to_xy(split):
    Y = split[-1]
    X = np.delete(split, (4), axis=0)
    return (X, Y)

def initialize_coefficients(layers):
    L = len(layers)
    weights = [np.random.randn(layers[l], layers[l-1]) * np.sqrt(2 / layers[l-1]) for l in range(1, L)]
    biases = [np.zeros((layers[l], 1)) for l in range(1, L)]
    return weights, biases

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def average_cross_entropy(Y, A, n):
    cross_entropy = -(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return 1/n * np.sum(cross_entropy)

def feed_forward(w, X, b, activation_fn):
    Z = np.dot(w, X) + b
    A = activation_fn(Z)
    return (A, Z)

def forward_propagate(W, X, b):
    A1, Z1 = feed_forward(W[0], X, b[0], relu)
    A2, Z2 = feed_forward(W[1], A1, b[1], sigmoid)
    return (A1, A2, Z1, Z2)

def backpropagate_2nd_layer(A1, A2, Y, n):
    dL_dZ2 = A2 - Y
    dL_dW2 = 1/n * np.dot(dL_dZ2, A1.T)
    dL_dB2 = 1/n * dL_dZ2.sum(axis=1, keepdims=True)
    return (dL_dZ2, dL_dW2, dL_dB2)

def backpropagate_1st_layer(W, X, dL_dZ2, Z1, n):
    dL_dZ1 = np.dot(dL_dZ2.T, W[1]) * relu_derivative(Z1).T
    dL_dW1 = 1/n * np.dot(dL_dZ1.T, X.T)
    dL_dB1 = 1/n * dL_dZ1.sum(axis=0, keepdims=True).T
    return (dL_dW1, dL_dB1)

def update_weights(alpha, gradients, W, b):
    dL_dW1, dL_dW2, dL_dB1, dL_dB2 = gradients
    W[0] -= alpha * dL_dW1
    W[1] -= alpha * dL_dW2
    b[0] -= alpha * dL_dB1
    b[1] -= alpha * dL_dB2

def binarize(x):
    return (x > 0.5).astype(np.float_)

def calculate_accuracy(A, Y):
    A = binarize(A)
    return np.equal(A, Y).sum() / Y.shape[0] * 100

def evaluate_performance(A, Y, dataset):
    accuracy = calculate_accuracy(A, Y)
    A = binarize(A)
    result = np.equal(A, Y)
    print(f'{dataset} accuracy: {accuracy}. {dataset} misclassifications: {A[np.where(result == False)]}')

train_X, train_Y = convert_split_to_xy(train_split)
validation_X, validation_Y = convert_split_to_xy(validation_split)
test_X, test_Y = convert_split_to_xy(test_split)

alpha = 0.0075
epochs = 2500
n = len(train_Y)
weights, biases = initialize_coefficients([4, 10, 1])
training_loss = []
validation_loss = []
test_loss = []
training_accuracy = []
validation_accuracy = []
test_accuracy = []

for i in range(epochs + 1):
    train_A1, train_A2, train_Z1, _ = forward_propagate(weights, train_X, biases)
    _, validation_A2, _, _ = forward_propagate(weights, validation_X, biases)
    _, test_A2, _, _ = forward_propagate(weights, test_X, biases)

    if i % 100 == 0:
        training_loss.append(average_cross_entropy(train_Y, train_A2, n))
        validation_loss.append(average_cross_entropy(validation_Y, validation_A2, len(validation_Y)))
        test_loss.append(average_cross_entropy(test_Y, test_A2, len(test_Y)))
        training_accuracy.append(calculate_accuracy(train_A2, train_Y))
        validation_accuracy.append(calculate_accuracy(validation_A2, validation_Y))
        test_accuracy.append(calculate_accuracy(test_A2, test_Y))
    
    dL_dZ2, dL_dW2, dL_dB2 = backpropagate_2nd_layer(train_A1, train_A2, train_Y, n)
    dL_dW1, dL_dB1 = backpropagate_1st_layer(weights, train_X, dL_dZ2, train_Z1, n)

    gradients = (dL_dW1, dL_dW2, dL_dB1, dL_dB2)
    update_weights(alpha, gradients, weights, biases)

    if i == epochs:
        evaluate_performance(train_A2, train_Y, 'Training')
        evaluate_performance(validation_A2, validation_Y, 'Validation')
        evaluate_performance(test_A2, test_Y, 'Test')

plt.figure()
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.plot(test_loss, label='Test Loss')
plt.legend()

plt.figure()
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.plot(test_accuracy, label='Test Accuracy')
plt.legend()
plt.show()

