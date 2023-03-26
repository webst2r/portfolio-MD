import numpy as np


class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.variance = None
        self.prior = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        num_features = X.shape[1]
        self.mean = np.zeros((num_classes, num_features))
        self.variance = np.zeros((num_classes, num_features))
        self.prior = np.zeros(num_classes)

        for i, c in enumerate(self.classes):
            X_c = X[c == y]
            self.mean[i, :] = X_c.mean(axis=0)
            self.variance[i, :] = X_c.var(axis=0)
            self.prior[i] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            posterior = np.zeros(len(self.classes))

            for j, c in enumerate(self.classes):
                prior = np.log(self.prior[j])
                likelihood = np.sum(np.log(self.gauss_function(x, self.mean[j, :], self.variance[j, :])))
                posterior[j] = prior + likelihood

            y_pred[i] = self.classes[np.argmax(posterior)]

        return y_pred

    def gauss_function(self, x, mean, variance):
        exponent = np.exp(-(x - mean) ** 2 / (2 * variance))
        return exponent / (np.sqrt(2 * np.pi * variance))


def main():


    # Crie um conjunto de dados artificial com 2 classes e 4 recursos
    X_train = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7],
                        [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 10], [8, 9, 10, 11]])
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    X_test = np.array([[1, 3, 5, 7], [2, 4, 6, 8], [3, 5, 7, 9], [4, 6, 8, 10]])
    y_test = np.array([0, 0, 1, 1])

    # Crie uma instância do classificador Naive Bayes
    naive = NaiveBayes()

    # Treine o modelo usando o conjunto de dados de treinamento
    naive.fit(X_train, y_train)

    # Faça previsões no conjunto de dados de teste
    y_pred = naive.predict(X_test)

    print(y_pred)

    # Avalie o desempenho do modelo usando a precisão
    accuracy = np.mean(y_pred == y_test)

    print("A precisão do modelo é:", accuracy)



if __name__ == "__main__":
        main()