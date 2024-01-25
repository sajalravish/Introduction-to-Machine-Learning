###################################################################
# QUESTION 3: data partitioning and evaluation metrics
import numpy as np

# data_name can be "mnist", "spam", or "toy"
def load_data(data_name):
    data = np.load(f"../data/{data_name}-data.npz")
    print("Loaded %s data!" % data_name)
    return data["training_data"], data["training_labels"]


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
    training_label = training_data[:, -1]
    
    validation_features = validation_data[:, :-1]
    validation_label = validation_data[:, -1]

    return training_features, training_label, validation_features, validation_label


# Load MNIST dataset
mnist_data, mnist_labels = load_data("mnist")

# Partitioning MNIST dataset (set aside )
mnist_training_features, mnist_training_label, mnist_validation_features, mnist_validation_label = partition(mnist_data, mnist_labels, 10000)

# Load spam dataset
spam_data, spam_labels = load_data("spam")

# Partitioning spam dataset
spam_training_features, spam_training_label, spam_validation_features, spam_validation_label = partition(spam_data, spam_labels, 0.2)

print(f"\nMNIST Training Features: {mnist_training_features.shape}")
print(f"MNIST Training Labels: {mnist_training_label.shape}")
print(f"MNIST Validation Features: {mnist_validation_features.shape}")
print(f"MNIST Validation Labels: {mnist_validation_label.shape}")
print(f"\nSpam Training Features: {spam_training_features.shape}")
print(f"Spam Training Labels: {spam_training_label.shape}")
print(f"Spam Validation Features: {spam_validation_features.shape}")
print(f"Spam Validation Labels: {spam_validation_label.shape}")


def accuracy_eval(true_labels, predicted_labels):
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Inputs must have the same length!")
    total_labels = len(true_labels)

    # Calculate accuracy score
    correct_predictions = 0
    for i in range(total_labels):
        if true_labels[i] == predicted_labels[i]:
            correct_predictions += 1
    score = correct_predictions / total_labels

    return score

# Testing accuracy_eval:
true_labels = [1, 0, 2, 3, 3]
predicted_labels = [1, 0, 1, 0, 1]
score = accuracy_eval(true_labels, predicted_labels)
print(f"Accuracy: {score}")


###################################################################
# QUESTION 4: support vector machines
from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt

# training linear SVM on MNIST dataset
mnist_linear_svc = SVC(kernel='linear')
mnist_training_examples = [100, 200, 500, 1000, 2000, 5000, 10000]

# collect data on the accuracy of our model for the plot
mnist_training_accuracy = []
mnist_validation_accuracy = []

for limit in mnist_training_examples:    
    # The below print statement is for debugging purposes:
    # print(f"Number of unique classes: {np.unique(example_training_label)}")

    # fit the SVM model to the given training data
    mnist_linear_svc.fit(mnist_training_features[:limit], mnist_training_label.flatten()[:limit])
    
    training_score = accuracy_eval(mnist_training_label.flatten()[:limit], mnist_linear_svc.predict(mnist_training_features[:limit]))
    validation_score = accuracy_eval(mnist_validation_label.flatten()[:limit], mnist_linear_svc.predict(mnist_validation_features[:limit]))
    
    mnist_training_accuracy.append(training_score)
    mnist_validation_accuracy.append(validation_score)
    
    print(f"\nTraining with {limit} examples")
    print(f"Training accuracy: {training_score}") 
    print(f"Validation Accuracy: {validation_score}")

# plotting accuracy
plt.plot(mnist_training_examples, mnist_training_accuracy, label="Training Accuracy", marker='o')
plt.plot(mnist_training_examples, mnist_validation_accuracy, label="Validation Accuracy", marker='o')
plt.xlabel("Number of Training Examples")
plt.ylabel("Accuracy")
plt.title("SVM Accuracy for MNIST dataset")
plt.legend()
plt.show()


# training linear SVM on Spam dataset
spam_linear_svc = SVC(kernel='linear')

ALL = 4171
spam_training_examples = [100, 200, 500, 1000, 2000, ALL]

spam_training_accuracy = []
spam_validation_accuracy = []

for limit in spam_training_examples:    
    spam_linear_svc.fit(spam_training_features[:limit], spam_training_label.flatten()[:limit])
    
    training_score = accuracy_eval(spam_training_label.flatten()[:limit], spam_linear_svc.predict(spam_training_features[:limit]))
    validation_score = accuracy_eval(spam_validation_label.flatten()[:limit], spam_linear_svc.predict(spam_validation_features[:limit]))
    
    spam_training_accuracy.append(training_score)
    spam_validation_accuracy.append(validation_score)
    
    print(f"\nTraining with {limit} examples")
    print(f"Training accuracy: {training_score}") 
    print(f"Validation Accuracy: {validation_score}")

# plotting accuracy
plt.plot(spam_training_examples, spam_training_accuracy, label="Training Accuracy", marker='o')
plt.plot(spam_training_examples, spam_validation_accuracy, label="Validation Accuracy", marker='o')
plt.xlabel("Number of Training Examples")
plt.ylabel("Accuracy")
plt.title("SVM Accuracy for Spam Dataset")
plt.legend()
plt.show()


###################################################################
# QUESTION 5: hyperparameter tuning

# Geometric sequence of C-values to try:
# The values 15 I'm testing are: 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1000, 10000
C_values = [10**i for i in range(-10, 4)]

best_accuracy = 0.0
best_C = None
limit = 10000  # required to train with at least 10,000 training examples

for C_value in C_values:
    mnist_C_svc = SVC(kernel='linear', C=C_value)
    mnist_C_svc.fit(mnist_training_features[:limit], mnist_training_label.flatten()[:limit])
    
    training_score = accuracy_eval(mnist_training_label.flatten()[:limit], mnist_C_svc.predict(mnist_training_features[:limit]))
    validation_score = accuracy_eval(mnist_validation_label.flatten()[:limit], mnist_C_svc.predict(mnist_validation_features[:limit]))
    
    print(f"\nTraining with {C_value} C-value")
    print(f"Training accuracy: {training_score}") 
    print(f"Validation Accuracy: {validation_score}")

    # calculate the average accuracy
    accuracy = (training_score + validation_score) / 2

    # determine which C_value results in a model with the highest average accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_C = C_value

print(f"\nBest C value: {best_C}")
print(f"Best Validation Accuracy: {best_accuracy}")

# After running the above hyperparameter tuning algorithm, I found that:
best_C = 0.0001


###################################################################
# QUESTION 6: k-fold cross-validation
# from sklearn.model_selection import KFold

# Geometric sequence of C-values to try
# The values 14 I'm testing are: 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 500
Ck_values = [10**i for i in range(-10, 2)]
Ck_values.append(500)

best_accuracy = 0.0
best_Ck = None

# perform 5-fold cross-validation
n_splits = 5
kf_indices = np.array_split(np.arange(len(spam_training_features)), n_splits)

for Ck_value in Ck_values:
    avg_accuracy = 0.0

    for index in range(n_splits):
        validation_set_index = kf_indices[index]
        training_set_index = np.concatenate([kf_indices[j] for j in range(n_splits) if j != index]) # everything that's not in the validation set

        # train our model on k-1 sets
        spam_C_svc = SVC(kernel='linear', C=Ck_value)
        spam_C_svc.fit(spam_training_features[training_set_index], spam_training_label[training_set_index].flatten())

        # validate our model on the kth set
        avg_accuracy += accuracy_eval(spam_training_label[validation_set_index].flatten(), spam_C_svc.predict(spam_training_features[validation_set_index]))

    # the cross-validation accuracy we report is the accuracy averaged over the k iterations
    avg_accuracy /= 5

    print(f"\nC = {Ck_value}")
    print(f"Cross-Validation Accuracy = {avg_accuracy}")
    
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_Ck = Ck_value

print(f"\nBest C value: {best_Ck}")
print(f"Best Cross-Validation Accuracy: {best_accuracy}")

# After running the above cross-validation algorithm, I found that:
best_Ck = 500


###################################################################
# QUESTION 7: predictions for Kaggle
import pandas as pd

# A code snippet to help you save your results into a kaggle accepted csv
# Usage: results_to_csv(clf.predict(X_test))
def results_to_csv(y_test, file_name):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv(file_name, index_label='Id')

# for MNIST dataset 
# retraining our SVM model
mnist_linear_svc = SVC(kernel='linear', C=best_C)
limit = 30000

mnist_linear_svc.fit(mnist_training_features[:limit], mnist_training_label.flatten()[:limit])

training_score = accuracy_eval(mnist_training_label.flatten(), mnist_linear_svc.predict(mnist_training_features))
validation_score = accuracy_eval(mnist_validation_label.flatten(), mnist_linear_svc.predict(mnist_validation_features))
print(f"Final training accuracy: {training_score}")
print(f"Final validation Accuracy: {validation_score}")

# save predictions to csv file
mnist_data = np.load(f"../data/mnist-data.npz")
mnist_test_data = mnist_data["test_data"]
mnist_test_predictions = mnist_linear_svc.predict(mnist_test_data.reshape((mnist_test_data.shape[0], -1)))

results_to_csv(mnist_test_predictions, "MNIST-submission.csv")
print("MNIST data saved to submission.csv")


# for Spam dataset
# retraining our SVM model
spam_linear_svc = LinearSVC(dual="auto", C=best_Ck)
limit = 2000

spam_linear_svc.fit(spam_training_features[:limit], spam_training_label[:limit])

training_score = accuracy_eval(spam_training_label.flatten(), spam_linear_svc.predict(spam_training_features))
validation_score = accuracy_eval(spam_validation_label.flatten(), spam_linear_svc.predict(spam_validation_features))
print(f"Final training accuracy: {training_score}")
print(f"Final validation Accuracy: {validation_score}")

# saving predictions to .csv file
spam_data = np.load(f"../data/spam-data.npz")
spam_test_data = spam_data["test_data"]
spam_test_predictions = spam_linear_svc.predict(spam_test_data)

results_to_csv(spam_test_predictions, "Spam-submission.csv")
print("Spam data saved to submission.csv")