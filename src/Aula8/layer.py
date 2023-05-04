import numpy as np

class Layer:
    """
    A dense layer.
    """
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Executa a propagação para frente da camada, retornando o output resultante.
        """
    
        self.inputs = inputs

        print("self.inputs: ", self.inputs.shape)
        print("self.weights: ", self.weights.shape)

        self.output = np.dot(inputs, self.weights) + self.bias

        return self.output

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Executa a propagação para trás da camada, atualizando os pesos e bias da camada e retornando o erro propagado.
        """
        """
        # derivada da loss em relação ao output
        d_output = error

        # derivada da loss em relação aos pesos
        d_weights = np.dot(self.inputs.T, d_output)

        # derivada da loss em relação ao bias
        d_bias = np.sum(d_output, axis=0, keepdims=True)
        
        # atualiza pesos e bias
        print('self.weights shape:', self.weights.shape)
        print('d_weights shape:', d_weights.shape)

        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias

        print('2Weights shape:', self.weights.shape)

        # calcula e retorna erro propagado
        return np.dot(d_output, self.weights)
        """
        return error
    
class SigmoidActivation:
    """
    A sigmoid activation layer.
    """

    def __init__(self):
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the layer using the given input.
        Returns a 2d numpy array with shape (1, output_size).
        """
        return 1 / (1 + np.exp(-X))

    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        """
        return error