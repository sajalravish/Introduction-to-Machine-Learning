import numpy as np
import matplotlib.pyplot as plt
from scipy import io as io

np.random.seed(189)

###########################################
# Question 3.2: Batch Gradient Descent Code
from math import e

# Logistic function
def sigmoid(z):
    return 1 / (1 + pow(e, -z))

# Cost function with L2 regularization
def compute_cost(X, y, w, lambda_):
    epsilon = 1e-5
    h = sigmoid(X.dot(w))
    cost = (-np.dot(y, np.log(h + epsilon)) - np.dot((1 - y), np.log(1 - h + epsilon))) + lambda_ * np.sum(pow(w[1:], 2))
    return cost

def batch_gradient_descent(X, y, w, learning_rate, lambda_, iterations):
    cost_history = np.zeros(iterations)

    for iteration in range(iterations):
        h = sigmoid(X.dot(w))
        w -= learning_rate * (np.dot(X.T, h - y) + (lambda_ * w))
        cost_history[iteration] = compute_cost(X, y, w, lambda_)

    return w, cost_history


# Load data
load_data = io.loadmat("data.mat")
print(load_data.keys())
features = load_data["X"]
labels = load_data["y"]
X_test = load_data['X_test']

# Normalize data
mean = np.mean(features, axis=0)
std = np.std(features, axis=0)
features = (features - mean) / std
X_test = (X_test - mean) / std

# Add fictitious dimension
features = np.hstack((np.ones((features.shape[0], 1)), features))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Shuffle data
n = len(labels)
shuffle_indices = np.random.permutation(n)
features_shuffled = features[shuffle_indices]
labels_shuffled = labels[shuffle_indices]

# Split data into training and validation sets
split_index = 5000
training_features, validation_features = features_shuffled[:split_index], features_shuffled[split_index:]
training_labels, validation_labels = labels_shuffled[:split_index].flatten(), labels_shuffled[split_index:].flatten()

# Initialize parameters
w = np.zeros(features.shape[1])
learning_rate_bgd = 0.0001
lambda_bgd = 0.1
iterations_bgd = 5000

w_bgd, cost_history_bgd = batch_gradient_descent(training_features, training_labels, w, learning_rate_bgd, lambda_bgd, iterations_bgd)

# Plot cost versus iterations
plt.plot(range(iterations_bgd), cost_history_bgd)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Iterations for Batch Gradient Descent')
plt.show()


################################################
# Question 3.4: Stochastic Gradient Descent Code

def stochastic_gradient_descent(X, y, w, learning_rate, lambda_, iterations):
    cost_history = np.zeros(iterations)
    n = len(y)
    o = 0.0001

    for iteration in range(iterations):
        if iteration == 0:
            learning_rate = o
        elif learning_rate == -1:
            learning_rate = o / iteration

        h = sigmoid(X.dot(w))
        cost_history[iteration] = compute_cost(X, y, w, lambda_)

        rand_index = np.random.randint(0, n) 
        x_i = X[rand_index]
        y_i = y[rand_index]
        h_i = h[rand_index]
        gradient = n * (h_i - y_i) * x_i.T + (lambda_ * w)
        w -= learning_rate * gradient

        if learning_rate == o:
            learning_rate = -1
        elif learning_rate == o / iteration:
            learning_rate = -1

    return w, cost_history

# Specify parameters
w = np.zeros(features.shape[1])
learning_rate_sgd = 0.000001
lambda_sgd = 0.1
iterations_sgd = 5000

# Run stochastic gradient descent
w_sgd, cost_history_sgd = stochastic_gradient_descent(training_features, training_labels, w, learning_rate_sgd, lambda_sgd, iterations_sgd)

# Plot cost versus iterations
plt.plot(range(iterations_sgd), cost_history_sgd)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Iterations for Stochastic Gradient Descent with Constant Learning Rate')
plt.legend()
plt.show()


################################################
# Question 3.5: Stochastic Gradient Descent Code

# Specify parameters
w = np.zeros(features.shape[1])
learning_rate_sgd2 = -1
lambda_sgd2 = 0.1
iterations_sgd2 = 5000

# Run stochastic gradient descent
w_sgd2, cost_history_sgd2 = stochastic_gradient_descent(training_features, training_labels, w, learning_rate_sgd2, lambda_sgd2, iterations_sgd2)

# Plot cost versus iterations
plt.plot(range(iterations_sgd2), cost_history_sgd2)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Iterations for Stochastic Gradient Descent with Shrinking Learning Rate')
plt.legend()
plt.show()

# Compare this plot to the plot from question 3.4
plt.plot(range(iterations_sgd), cost_history_sgd, label = "constant learning rate", color = "red")
plt.plot(range(iterations_sgd2), cost_history_sgd2, label = "shrinking learning rate", color = "blue")
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Iterations for Stochastic Gradient Descent')
plt.legend()
plt.show()


#################################
# Question 3.6: Kaggle Submisison
import pandas as pd

def predict(X, w):
    # Calculate probabilities
    probabilities = sigmoid(X.dot(w))

    # Convert probabilities to binary predictions
    predictions = (probabilities >= 0.5).astype(int)
    return predictions

# Code snippet to help you save your results into a kaggle-accepted csv
def results_to_csv(y_test, file_name):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv(file_name, index_label='Id')

predictions = predict(X_test, w_bgd)
results_to_csv(predictions, "Wine-submission.csv")
print('Done! :)')
