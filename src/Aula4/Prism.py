class PRISM:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        node = self.tree_
        while node['type'] != 'leaf':
            if inputs[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['class']

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(set(y))

        # Check termination conditions
        if n_labels == 1:
            return {'type': 'leaf', 'class': y[0]}
        if n_samples < 2 or depth == self.max_depth:
            return {'type': 'leaf', 'class': np.bincount(y).argmax()}

        # Find the best split
        best_feature, best_threshold, best_score = self._find_split(X, y)

        # Check termination condition
        if best_score == 0:
            return {'type': 'leaf', 'class': np.bincount(y).argmax()}

        # Recur on the sub-trees
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth+1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth+1)

        # Return the current node
        return {'type': 'split', 'feature': best_feature, 'threshold': best_threshold,
                'left': left_subtree, 'right': right_subtree}

    def _find_split(self, X, y):
        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        best_score = -1

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                left_labels = y[left_indices]
                right_labels = y[right_indices]
                score = self._gini_impurity(left_labels, right_labels)

                if score > best_score:
                    best_feature = feature
                    best_threshold = threshold
                    best_score = score

        return best_feature, best_threshold, best_score

    def _gini_impurity(self, left_labels, right_labels):
        n_left, n_right = len(left_labels), len(right_labels)
        n_total = n_left + n_right
        gini_left = 1.0 - sum([(np.sum(left_labels == c) / n_left) ** 2 for c in range(self.n_classes_)])
        gini_right = 1.0 - sum([(np.sum(right_labels == c) / n_right) ** 2 for c in range(self.n_classes_)])
        return (n_left / n_total) * gini_left + (n_right / n_total) * gini_right


    def _leaf_node(self, y):
        n_samples = len(y)
        counts = np.bincount(y, minlength=self.n_classes_)
        probabilities = counts / n_samples
        return {'leaf': True, 'probabilities': probabilities}

def main():
    # Generate sample data
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # Fit the PRISM model
    prism = PRISM(max_depth=2)
    prism.fit(X, y)

    # Make predictions
    y_pred = prism.predict(X)

    # Print results
    print("Sample Data:")
    print("X =", X[:10])
    print("y =", y[:10])
    print("Predicted y =", y_pred[:10])
    print("Accuracy =", np.mean(y == y_pred))

main()
