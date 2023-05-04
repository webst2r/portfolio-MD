from typing import Callable
import numpy as np
import sys
sys.path.append('..')
from Aula1.Dataset import Dataset
from Aula8.layer import Layer, SigmoidActivation

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
        Retorna a 'accuracy' do modelo para um determinado dataset.
    """
    return np.sum(y_true == y_pred) / len(y_true)

def mse(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """
        Calcula e retorna o Erro Quadrado Médio do modelo num dataset.
    """
    return np.sum((true_labels - predicted_labels) ** 2) / (len(true_labels) * 2)


def mse_deriv(true_labels: np.ndarray, predicted_labels: np.ndarray) -> np.ndarray:
    """
        Retorna a derivada do Erro Quadrado Médio para a variável predicted_labels.
    """
    return -2 * (true_labels - predicted_labels) / (len(true_labels) * 2)



class MLP:
    """
    Modelo de Rede Neuronal com várias camadas.
    A lista de camadas passada como parâmetro para a inicialização da classe inclui a camada de entrada, a camada intermediária e a camada de saída.
    O método de ajuste ('fit') implementa a propagação para frente e para trás nas camadas, o que é uma parte essencial do algoritmo de retropropagação.
    """
    def __init__(self,
                 layers: list,
                 epochs: int = 1000,
                 learning_rate: float = 0.01,
                 loss: Callable = mse,
                 loss_derivative: Callable = mse_deriv,
                 verbose: bool = False):
        
        # parametros
        self.layers = layers
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss = loss                        # funcao de loss
        self.loss_derivative = loss_derivative  # derivada da funcao de loss
        self.verbose = verbose                  # se é para imprimir a loss em cada epoch

    def fit(self, dataset: Dataset) -> 'MLP':
        """
            Ajustar o modelo para o dataset dado.
        """
        X = dataset.X
        y = dataset.y
        
        for epoch in range(1, self.epochs + 1):

            # forward propagation
            for layer in self.layers:
                X = layer.forward(X)

            # backward propagation
            #error = self.loss_derivative(y, X)
            #for layer in self.layers[::-1]:
            #    error = layer.backward(error, self.learning_rate)
        
            # calculate cost
            cost = self.loss(y, X)

            # print loss
            if self.verbose:
                print(f'Epoch {epoch}/{self.epochs} - cost: {cost}')

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
            Prever o output do dataset.
            Retorna um array (prediction)
        """
        X = dataset.X

        # forward propagation
        for layer in self.layers:
            X = layer.forward(X)

        return X
    

    def cost(self, dataset: Dataset) -> float:
        """
            Computar o custo do modelo no dataset dado.
        """
        y_pred = self.predict(dataset)
        return self.loss(dataset.y, y_pred)

    def score(self, dataset: Dataset, scoring_func: Callable = accuracy) -> float:
        """
            Computar o score do modelo no dataset dado.
        """
        y_pred = self.predict(dataset)
        return scoring_func(dataset.y, y_pred)




def main():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([[1],
                  [0],
                  [0],
                  [1]])

    # XNOR problem is a binary classification problem
    dataset = Dataset(X, y, features=['x1', 'x2'], label='X1 XNOR X2')
    dataset.to_dataframe()

   # weights for Dense Layer 1
    w1 = np.array([[20, -20],
                [20, -20]])
    b1 = np.array([[-30, 10]])

    l1 = Layer(input_size=2, output_size=2)
    l1.weights = w1
    l1.bias = b1


    # weights for Dense Layer 2

    w2 = np.array([[20],
                [20]])
    b2 = np.array([[-10]])

    l2 = Layer(input_size=2, output_size=1)
    l2.weights = w2
    l2.bias = b2

    l1_sg = SigmoidActivation()
    l2_sg = SigmoidActivation()

    # MLP
    model = MLP(layers=[l1, l1_sg, l2, l2_sg])

    #print("Fit: ", model.fit(dataset))
    print("Predictions: ", model.predict(dataset))


if __name__ == '__main__':
    main()
