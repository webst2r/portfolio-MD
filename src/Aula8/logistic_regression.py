import numpy as np
import sys
sys.path.append('..')
from typing import Tuple
from Aula1.Dataset import Dataset


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(y_true == y_pred) / len(y_true)

def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets.
    """
    # set random state
    np.random.seed(random_state)

    # get dataset size
    n_samples = dataset.shape()[0]

    # get number of samples in the test set
    n_test = int(n_samples * test_size)

    # get the dataset permutations
    permutations = np.random.permutation(n_samples)

    # get samples in the test set
    test_idxs = permutations[:n_test]

    # get samples in the training set
    train_idxs = permutations[n_test:]

    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

def sigmoid_function(X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-X))

class LogisticRegression:
    """
    The LogisticRegression is a logistic model using the L2 regularization.
    This model solves the logistic regression problem using an adapted Gradient Descent technique

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    learning_rate: float
        The learning rate
    max_iter: int
        The maximum number of iterations

    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the logistic model.
        For example, sigmoid(x0 * theta[0] + x1 * theta[1] + ...)
    theta_zero: float
        The intercept of the logistic model
    """
    def __init__(self, l2_penalty: float = 1, learning_rate: float = 0.001, max_iter: int = 1000):
        # parameters
        self.l2_penalty = l2_penalty
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        # attributes
        self.theta = None
        self.theta_zero = None

    def fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Fit the model to the dataset.
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        for i in range(self.max_iter):
            # predicted y
            y_pred = np.dot(dataset.X, self.theta) + self.theta_zero

            # apply sigmoid function
            y_pred = sigmoid_function(y_pred)

            # compute the gradient using the learning rate
            gradient = (self.learning_rate * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

            # compute the penalty
            penalization_term = self.learning_rate * (self.l2_penalty / m) * self.theta

            # update the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.learning_rate * (1 / m)) * np.sum(y_pred - dataset.y)

        return self

    def predict(self, dataset: Dataset) -> np.array:
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

        # convert the predictions to 0 or 1 (binarization)
        mask = predictions >= 0.5
        predictions[mask] = 1
        predictions[~mask] = 0
        return predictions

    def score(self, dataset: Dataset) -> float:
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization
        """
        predictions = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        cost = (-dataset.y * np.log(predictions)) - ((1 - dataset.y) * np.log(1 - predictions))
        cost = np.sum(cost) / dataset.shape()[0]
        cost = cost + (self.l2_penalty * np.sum(self.theta ** 2) / (2 * dataset.shape()[0]))
        return cost


if __name__ == '__main__':
    # load and split the dataset
    dataset_ = Dataset()
    dataset_.X = np.random.rand(600, 100)
    dataset_.y = np.random.randint(0, 2, 600)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # fit the model
    model = LogisticRegression(l2_penalty=1, learning_rate=0.001, max_iter=1000)
    model.fit(dataset_train)

    # compute the score
    score = model.score(dataset_test)
    print(f"Score: {score}")