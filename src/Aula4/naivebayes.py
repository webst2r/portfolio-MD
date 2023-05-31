from collections import Counter, defaultdict
from math import log
import random
from abc import ABC, abstractmethod


class DiscreteFeatureVectors:
    """Collection of discrete feature vectors."""

    def __init__(self, use_smoothing):
        """Construct a container for discrete feature vectors.

        :param use_smoothing: A boolean to indicate whether to use smoothing.
        """
        self.use_laplace_smoothing = use_smoothing
        self.possible_categories = defaultdict(set)
        self.frequencies = defaultdict(lambda: defaultdict(lambda: Counter()))

    def add(self, label, index, feature):
        self.frequencies[label][index][feature.value] += 1
        self.possible_categories[index].add(feature.value)

    def probability(self, label, index, feature, num_label_instances):
        """Calculate probability with Laplace smoothing optionally."""
        frequency = self.frequencies[label][index][feature.value]
        frequency += 1 if self.use_laplace_smoothing else 0

        num_classes = len(self.possible_categories[index])
        num_label_instances += num_classes if self.use_laplace_smoothing else 0
        return frequency / num_label_instances


class DiscreteFeature:
    def __init__(self, value):
        self.value = value


class NaiveBayes:
    def __init__(self, extract_features, use_smoothing=True):
        """Create a naive bayes classifier.

        :param extract_features: Callback to map a feature vector to discrete and continuous features.
        :param use_smoothing: Whether to use smoothing when calculating probability.
        """
        self.priors = defaultdict(dict)

        self.label_counts = Counter()
        self.discrete_feature_vectors = DiscreteFeatureVectors(use_smoothing)
        self._extract_features = extract_features
        self._is_fitted = False

    def fit(self, design_matrix, target_values):
        """Fit model according to design matrix and target values."""
        for i, training_example in enumerate(design_matrix):
            label = target_values[i]
            self.label_counts[label] += 1
            features = self._extract_features(training_example)
            for j, feature in enumerate(features):
                self.discrete_feature_vectors.add(label, j, feature)

        total_num_records = len(target_values)
        for label in set(target_values):
            self.priors[label] = self.label_counts[label] / total_num_records

        self._is_fitted = True
        return self

    def predict(self, test_set):
        """Predict target values for test set."""
        self._check_is_fitted()

        predictions = []
        for i in range(len(test_set)):
            result = self.predict_record(test_set[i])
            predictions.append(result)
        return predictions

    def predict_record(self, test_record):
        """Predict the label for the test record."""
        self._check_is_fitted()

        log_likelihood = {k: log(v) for k, v in self.priors.items()}
        for label in self.label_counts:
            features = self._extract_features(test_record)
            for i, feature in enumerate(features):
                probability = self._get_probability(i, feature, label)
                try:
                    log_likelihood[label] += log(probability)
                except ValueError as e:
                    raise ValueError("Error occurred while computing log likelihood")

        return max(log_likelihood, key=log_likelihood.get)

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise ValueError("This instance has not been fitted yet.")

    def _get_probability(self, feature_index, feature, label):
        probability = self.discrete_feature_vectors.probability(label, feature_index, feature, self.label_counts[label])
        return probability


class TestNaiveBayesWithBinaryDataset:
    """Test the Naive Bayes classifier with a toy binary dataset."""

    def setUpClass(cls):
        dataset = cls.get_toy_binary_dataset()
        cls.design_matrix = [row[:-1] for row in dataset]
        cls.target_values = [row[-1] for row in dataset]

    def test_predict_record_with_binary_dataset(self):
        expected_prediction = 1

        test_record = [1, 1, 0]
        clf = NaiveBayes(self.extract_features)
        clf.fit(self.design_matrix, self.target_values)
        prediction = clf.predict_record(test_record)

        if expected_prediction == prediction:
            print("Test passed: Predicted label matches the expected label.")
        else:
            print("Test failed: Predicted label does not match the expected label.")

    def test_zero_observations_error(self):
        clf = NaiveBayes(self.extract_features, use_smoothing=False)
        clf.fit(self.design_matrix, self.target_values)

        test_record = [0, 1, 0]
        try:
            clf.predict_record(test_record)
        except:
            print("Test passed: Zero observations error was handled.")
        else:
            print("Test failed: Zero observations error was not raised.")

    def test_accuracy(self):
        clf = NaiveBayes(self.extract_features)
        clf.fit(self.design_matrix, self.target_values)

        test_records = [
            [1, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ]
        expected_labels = [1, 0, 1, 0]

        predictions = clf.predict(test_records)

        correct = sum(1 for pred, expected in zip(predictions, expected_labels) if pred == expected)
        accuracy = correct / len(expected_labels) * 100

        print(f"Accuracy: {accuracy:.2f}%")
        print("Predictions:", predictions)

    @staticmethod
    def extract_features(feature_vector):
        return [
            DiscreteFeature(feature_vector[0]),
            DiscreteFeature(feature_vector[1]),
            DiscreteFeature(feature_vector[2]),
        ]

    @staticmethod
    def get_toy_binary_dataset():
        """The third with value 0 is never observed with class 0.

        This is called the zero observations problem,
        and is dealt with by additive smoothing.
        """
        return [
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 1],
        ]


class TestNaiveBayesWithLargeDataset:
    """Test the Naive Bayes classifier with a large discrete values dataset."""

    def setUpClass(cls):
        dataset = cls.get_large_discrete_dataset()
        cls.design_matrix = [row[:-1] for row in dataset]
        cls.target_values = [row[-1] for row in dataset]

    def test_predict_record_with_large_dataset(self):
        test_dataset = self.get_large_discrete_dataset()
        test_records = [row[:-1] for row in test_dataset]
        expected_labels = [row[-1] for row in test_dataset]

        clf = NaiveBayes(self.extract_features)
        clf.fit(self.design_matrix, self.target_values)
        predictions = clf.predict(test_records)

        correct_predictions = sum(1 for pred, exp in zip(predictions, expected_labels) if pred == exp)
        accuracy = correct_predictions / len(predictions) * 100

        print(f"Accuracy: {accuracy:.2f}%")

        print("Predicted\tExpected")
        for pred, exp in zip(predictions, expected_labels):
            print(f"{pred}\t\t{exp}")

        if correct_predictions == len(predictions):
            print("Test passed: All predictions match the expected labels.")
        else:
            print("Some predictions do not match the expected labels.")


    @staticmethod
    def extract_features(feature_vector):
        return [
            DiscreteFeature(feature_vector[0]),
            DiscreteFeature(feature_vector[1]),
            DiscreteFeature(feature_vector[2]),
            DiscreteFeature(feature_vector[3]),
            DiscreteFeature(feature_vector[4]),
            DiscreteFeature(feature_vector[5]),
        ]

    @staticmethod
    def get_large_discrete_dataset():
        """Generate a large discrete values dataset."""
        dataset = []
        num_instances = 1000  # Number of instances in the dataset

        for _ in range(num_instances):
            row = []
            for _ in range(6):  # Number of features in each instance
                # Generate random discrete values between 0 and 1
                value = random.randint(0, 1)
                row.append(value)
            # Add a random label (0 or 1) at the end
            label = random.randint(0, 1)
            row.append(label)
            dataset.append(row)

        return dataset


def main():
    print("Test NaiveBayes with Binary Dataset")
    test_case = TestNaiveBayesWithBinaryDataset()
    test_case.setUpClass()
    test_case.test_predict_record_with_binary_dataset()
    test_case.test_zero_observations_error()
    test_case.test_accuracy()

    print("\n\nTest NaiveBayes with Large Dataset")
    test_case = TestNaiveBayesWithLargeDataset()
    test_case.setUpClass()
    test_case.test_predict_record_with_large_dataset()

if __name__ == "__main__":
    main()
