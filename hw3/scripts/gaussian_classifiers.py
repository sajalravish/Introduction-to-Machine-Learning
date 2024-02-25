###############################################
# Question 8.1: Fitting a Gaussian distribution
import numpy as np
np.random.seed(42)

# data_name can be "mnist" or "spam"
def load_training_data(data_name):
    data = np.load(f"../data/{data_name}-data-hw3.npz")
    print("Loaded %s data!" % data_name)
    # fields = "test_data", "training_data", "training_labels"
    return data["training_data"], data["training_labels"]

# Function that normalizes images
def normalize_data(data):
    return np.array([(image / np.linalg.norm(image)).flatten() for image in data])

# Function that fits Gaussian distributions to each digit class
def fit_gaussian(data, labels, classes):
    fitted = {}
    for cls in classes:
        cls_data = data[(labels == cls).flatten()]
        mean = np.mean(cls_data, axis=0)
        covariance = np.cov(cls_data, rowvar=False)
        # If our covariance matrix is singular:
        if np.linalg.det(covariance) == 0:
            covariance += 1e-4 * np.eye(len(covariance))
        prior = len(cls_data) / len(data)
        fitted[cls] = (mean, covariance, prior)
    return fitted

mnist_data, mnist_labels = load_training_data("mnist")
mnist_classes = np.unique(mnist_labels)
mnist_data_normalized = normalize_data(mnist_data)
mnist_fitted = fit_gaussian(mnist_data_normalized, mnist_labels, mnist_classes)


#################################################
# Question 8.2: Visualizing the covariance matrix
import matplotlib.pyplot as plt

class_0_mean, class_0_covariance, class_0_prior = mnist_fitted[0]
plt.figure(figsize=(10, 10))
plt.imshow(class_0_covariance, cmap='Purples')
plt.title('Covariance Matrix Visualization for Digit Class')
plt.colorbar()
plt.show()


######################################################
# Question 8.3, part (a): Linear discriminant analysis
from scipy.stats import multivariate_normal

def partition(data, labels, split_ratio):
    # The split_ratio is what percent/number of training points we want to set aside for the validation set

    # reshape and stack data and labels
    labels = labels.reshape((-1, 1))
    if data.ndim > 2:
        data = data.reshape((data.shape[0], -1))
    stacked_data = np.hstack((data, labels))

    # determine split_size
    if split_ratio <= 1:
        split_size = int(len(stacked_data) * split_ratio)
    else:
        split_size = split_ratio

    # randomly shuffle the data
    np.random.shuffle(stacked_data)

    # partition data
    training_data = stacked_data[split_size:]
    validation_data = stacked_data[:split_size]

    # separate features and labels for training and validation sets
    training_features = training_data[:, :-1]
    training_labels = training_data[:, -1]
    
    validation_features = validation_data[:, :-1]
    validation_labels = validation_data[:, -1]

    return training_features, training_labels, validation_features, validation_labels

# Compute means and pooled covariance for LDA
def fit_lda(data, labels, classes):
    means = {}
    pooled_covariance = 0
    n = len(data)
    for cls in classes:
        cls_data = data[(labels == cls).flatten()]
        mean = np.mean(cls_data, axis=0)
        means[cls] = (mean)
        pooled_covariance += np.sum(np.linalg.norm(cls_data - mean, axis=1)**2)
    pooled_covariance /= n
    return means, pooled_covariance

def predict_lda(data, means, pooled_covariance):
    # Calculate the log likelihood of each class and return the class with the maximum likelihood
    log_likelihoods = []
    for cls in means:
        log_likelihood = multivariate_normal.logpdf(data, mean=means[cls], cov=pooled_covariance, allow_singular=True)
        log_likelihoods.append(log_likelihood)
    
    log_likelihoods = np.array(log_likelihoods)
    predictions = np.argmax(log_likelihoods, axis=0)
    return predictions

mnist_training_features, mnist_training_labels, mnist_validation_features, mnist_validation_labels = partition(mnist_data, mnist_labels, 10000)
training_sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
error_rates = []

for size in training_sizes:
    # Sample the training data
    mnist_training_features_samples = normalize_data(mnist_training_features)[:size]
    mnist_training_labels_samples = mnist_training_labels[:size]
    mnist_validation_features_normalized = normalize_data(mnist_validation_features)
    means, pooled_covariance = fit_lda(mnist_training_features_samples, mnist_training_labels_samples, mnist_classes)
    
    predictions = predict_lda(mnist_validation_features_normalized, means, pooled_covariance)

    # Classify and compute error rate
    correct_predictions = np.sum(predictions == mnist_validation_labels)
    error_rate = 1.0 - (correct_predictions / len(mnist_validation_labels))
    error_rates.append(error_rate)

# Plot the Error Rates
plt.plot(training_sizes, error_rates, marker='o')
plt.xlabel('Training Set Size')
plt.ylabel('Error Rate')
plt.title('LDA Classification Error Rate vs. Training Set Size')
plt.show()

#########################################################
# Question 8.3, part (b): Quadratic discriminant analysis

# Reset error_rates list before computing error rates for QDA
error_rates = []
error_rates_per_digit = {digit: [] for digit in mnist_classes}

def predict_qda(data, fitted):
    # Calculate the log likelihood of each class and return the class with the maximum likelihood
    log_likelihoods = []
    for i in range(len(fitted)):
        mean_val, cov_matrix, prior = fitted[i]
        log_likelihood = multivariate_normal.logpdf(data, mean=mean_val, cov=cov_matrix, allow_singular=True) + np.log(prior)
        log_likelihoods.append(log_likelihood)
    
    log_likelihoods = np.array(log_likelihoods)
    predictions = np.argmax(log_likelihoods, axis=0)
    return predictions

for size in training_sizes:
    # Sample the training data
    mnist_training_features_samples = normalize_data(mnist_training_features)[:size]
    mnist_training_labels_samples = mnist_training_labels[:size]
    mnist_validation_features_normalized = normalize_data(mnist_validation_features)
    
    # Compute means and covariances
    mnist_fitted = fit_gaussian(mnist_training_features_samples, mnist_training_labels_samples, mnist_classes)
    predictions = predict_qda(mnist_validation_features_normalized, mnist_fitted)

    # Classify and compute error rate
    correct_predictions = np.sum(predictions == mnist_validation_labels)
    error_rate = 1.0 - (correct_predictions / len(mnist_validation_labels))
    error_rates.append(error_rate)

    # Update error rates for each digit for LDA
    for digit in mnist_classes:
        digit_indices = np.where(mnist_validation_labels == digit)[0]
        digit_error = 1.0 - (np.sum(predictions[digit_indices] == digit) / len(digit_indices))
        error_rates_per_digit[digit].append(digit_error)

# Plot the Error Rates
plt.plot(training_sizes, error_rates, marker='o')
plt.xlabel('Training Set Size')
plt.ylabel('Error Rate')
plt.title('QDA Classification Error Rate vs. Training Set Size')
plt.show()

###################################################################################################
# Question 8.3, part (c): Plot validation error versus the number of training points for each digit

# Plot
plt.figure(figsize=(10, 6))
for digit, rates in error_rates_per_digit.items():
    plt.plot(training_sizes, rates, label=f'Digit {digit}')
plt.xlabel('Training Set Size')
plt.ylabel('Validation Error Rate')
plt.title('Validation Error Rate vs. Training Set Size for Each Digit')
plt.legend()
plt.grid(True)
plt.show()

#######################################
# Question 8.4: MNIST Kaggle submission

import pandas as pd

# A code snippet to help you save your results into a kaggle accepted csv
# Usage: results_to_csv(clf.predict(X_test))
def results_to_csv(y_test, file_name):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv(file_name, index_label='Id')

# save predictions to csv file
def load_test_data(data_name):
    data = np.load(f"../data/{data_name}-data-hw3.npz")
    print("Loaded %s data!" % data_name)
    # fields = "test_data", "training_data", "training_labels"
    return data["test_data"]

# Sample the training data
mnist_training_features_samples = normalize_data(mnist_training_features)[:15000]
mnist_training_labels_samples = mnist_training_labels[:15000]
    
mnist_test_data = load_test_data("mnist")
mnist_test_data_normalized = normalize_data(mnist_test_data)
# Compute means and covariances
mnist_fitted = fit_gaussian(mnist_training_features_samples, mnist_training_labels_samples, mnist_classes)

# Classify
mnist_test_predictions = predict_qda(mnist_test_data_normalized, mnist_fitted)

results_to_csv(mnist_test_predictions, "MNIST-submission.csv")
print("MNIST data saved to submission.csv")

######################################
# Question 8.5: Spam Kaggle submission

# Sample the training data
spam_data, spam_labels = load_training_data("spam")
spam_classes = np.unique(spam_labels)
spam_fitted = fit_gaussian(spam_data, spam_labels, spam_classes)

# Classify 
spam_test_data = load_test_data("spam")
spam_test_predictions = predict_qda(spam_test_data, spam_fitted)

results_to_csv(spam_test_predictions, "spam-submission.csv")
print("spam data saved to submission.csv")