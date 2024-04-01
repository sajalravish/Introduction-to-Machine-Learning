"""
Have Fun!
- 189 Course Staff
"""
from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from pydot import graph_from_dot_data
import io
import pandas as pd

import random
random.seed(246810)
np.random.seed(246810)

eps = 1e-5  #small number

class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def entropy(y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -1 * np.sum(probabilities * np.log2(probabilities + eps))
        return entropy

    @staticmethod
    def information_gain(X, y, thresh):
        parent_entropy = DecisionTree.entropy(y)

        left_child = X < thresh
        left_child_indices = np.where(left_child)[0]
        left_child_entropy = DecisionTree.entropy(y[left_child_indices])
        left_length = np.sum(left_child)

        right_child = ~left_child
        right_child_indices = np.where(right_child)[0]
        right_child_entropy = DecisionTree.entropy(y[right_child_indices])
        right_length = np.sum(right_child)

        children_entropy = (left_child_entropy * left_length + 
                            right_child_entropy * right_length) / (left_length + right_length)
        
        information_gain = parent_entropy - children_entropy
        return information_gain
        
    @staticmethod
    def gini_impurity(y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        impurity = 1 - np.sum(pow(probabilities, 2))
        return impurity

    @staticmethod
    def gini_purification(X, y, thresh):
        left_child = X < thresh
        left_child_indices = np.where(left_child)[0]
        left_impurity = DecisionTree.gini_impurity(y[left_child_indices])
        left_length = np.sum(left_child)

        right_child = X >= thresh
        right_child_indices = np.where(right_child)[0]
        right_impurity = DecisionTree.gini_impurity(y[right_child_indices])
        right_length = np.sum(right_child)

        gini_purification = (left_impurity * left_length +
                             right_impurity * right_length) / (left_length + right_length)
        return gini_purification

    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        # base condition for creating a leaf node
        if self.max_depth == 0 or len(np.unique(y)) == 1 or len(y) < 10:
            self.data = X
            self.pred = Counter(y).most_common(1)[0][0]
            return
        
        gains = []
        for idx in range(X.shape[1]):                   # loop over each feature in the dataset
            thresh_values = np.unique(X[:, idx])        # get the unique values of the feature
            for thresh in thresh_values:                # loop over the unique feature values
                gain = self.information_gain(X[:, idx], y, thresh)
                gains.append((idx, thresh, gain))

        self.split_idx, self.thresh, _ = max(gains, key=lambda x: x[2]) # choose the split that gives max info gain
        X0, y0, X1, y1 = self.split(X, y, self.split_idx, self.thresh)

        self.left = DecisionTree(max_depth=self.max_depth - 1, feature_labels=self.features)
        self.left.fit(X0, y0)
        self.right = DecisionTree(max_depth=self.max_depth - 1, feature_labels=self.features)
        self.right.fit(X1, y1)

    def predict(self, X):
        if self.max_depth == 0 or self.thresh is None:     # if self is a leaf node
            return np.array([self.pred] * len(X))
        else:
            predictions = np.zeros(len(X))
            left_child = X[:, self.split_idx] < self.thresh
            right_child = ~left_child
            predictions[left_child] = self.left.predict(X[left_child])
            predictions[right_child] = self.right.predict(X[right_child])
            return predictions

    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())


class BaggedTrees(BaseEstimator, ClassifierMixin):

    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # fit n decision trees to a random subsample of the data
        for tree in self.decision_trees:
            bagging_indices = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=True)
            tree.fit(X[bagging_indices], y[bagging_indices])

    def predict(self, X):
        predictions = []
        for tree in self.decision_trees:
            predictions.append(tree.predict(X))
        return np.mean(predictions, axis=0)


class RandomForest(BaggedTrees):

    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        params['max_features'] = m
        self.m = m
        super().__init__(params=params, n=n)


class BoostedRandomForest(RandomForest):

    def fit(self, X, y):
        # TODO
        pass
    
    def predict(self, X):
        # TODO
        pass


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    fill_mode = True

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack(
        [np.array(data, dtype=float),
         np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[1]):    # iterate through each feature
            if np.all(data[:, i] == -1):          # are any feature values missing?
                mode_value = Counter(data[:, i]).most_common(1)[0][0]
                data[:, i][data[:, i] == -1] = mode_value

    return data, onehot_features


def evaluate(clf):
    print("Cross validation", cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)


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

# A code snippet to help you save your results into a kaggle accepted csv
# Usage: results_to_csv(clf.predict(X_test))
# filename [STRING] = 'filename.csv'
def results_to_csv(y_test, filename):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1 # Ensures that the index starts at 1
    df.to_csv(filename, index_label='Id')


if __name__ == "__main__":
    #dataset = "titanic"
    dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=float).astype(int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)
    
    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # sklearn decision tree
    print("\n\nsklearn's decision tree")
    clf = DecisionTreeClassifier(random_state=0, **params)
    clf.fit(X, y)
    evaluate(clf)
    out = io.StringIO()
    export_graphviz(
        clf, out_file=out, feature_names=features, class_names=class_names)
    # For OSX, may need the following for dot: brew install gprof2dot
    graph = graph_from_dot_data(out.getvalue())
    graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)
    

    # debug data preprocessing step
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    print(X[0])

    # Run decision tree
    X_train, y_train, X_val, y_val = partition(X, y, 0.2)
    decisiontree_clf = DecisionTree(max_depth=params["max_depth"])
    decisiontree_clf.fit(X_train, y_train)

    y_pred_train = decisiontree_clf.predict(X_train)
    y_pred_val = decisiontree_clf.predict(X_val)
    training_accuracy_decisiontree = accuracy_eval(y_train, y_pred_train)
    validation_accuracy_decisiontree = accuracy_eval(y_val, y_pred_val)
    print(dataset, " decision tree training accuracy:", training_accuracy_decisiontree)
    print(dataset, " decision tree validation accuracy:", validation_accuracy_decisiontree)

    # Run random forest
    randomforect_clf = RandomForest()
    randomforect_clf.fit(X_train, y_train)

    y_pred_train = randomforect_clf.predict(X_train)
    y_pred_val = randomforect_clf.predict(X_val)
    training_accuracy_randomforest = accuracy_eval(y_train, y_pred_train)
    validation_accuracy_randomforest = accuracy_eval(y_val, y_pred_val)
    print(dataset, " random forest training accuracy:", training_accuracy_randomforest)
    print(dataset, " random forest validation accuracy:", validation_accuracy_randomforest)

    results_to_csv(decisiontree_clf.predict(Z), dataset + '_decisiontree_clf.csv')
    results_to_csv(randomforect_clf.predict(Z), dataset + '_randomforect_clf.csv')
