from scipy.stats import linregress
import numpy as np
import sys
sys.path.append('..')
from Aula1.Dataset import Dataset

class FRegression:
    """
    A classe FRegression recebe um parâmetro threshold, que é o valor limite para o p-value.
    Se o p-value de um atributo for menor que threshold, então o atributo é selecionado.
    """
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.selected_features_ = []
        
    def fit(self, dataset):
        """
        Recebe um objeto Dataset, extrai as features e a label do dataset, e para cada feature calcula o valor de p-value a partir da regressão linear.
        As features com p-value menor que threshold são selecionadas e armazenadas em self.selected_features_.
        """
        self.features = dataset.features
        self.label = dataset.label
        
        p_values = []
        for feature in enumerate(dataset.X.T):
            _, _ , _ , p_value, _ = linregress(feature, dataset.y)
            p_values.append(p_value)
            
        p_values = np.array(p_values)
        self.selected_features_ = np.where(p_values < self.threshold)[0]
        
        return self
    
    def transform(self, dataset):
        """
        Recebe um objeto Dataset e seleciona apenas as features armazenadas em self.selected_features_,
        retornando um novo objeto Dataset com as features selecionadas.
        """
        selected_X = dataset.X.reshape(-1, 1)[:, self.selected_features_]

        # selected_X = dataset.X[:, self.selected_features_]
        selected_features = [self.features[i] for i in self.selected_features_]
        return Dataset(X=selected_X, y=dataset.y, features=selected_features, label=self.label)
    
    def fit_transform(self, dataset):
        self.fit(dataset) 
        return self.transform(dataset)


if __name__ == "__main__":
    dataset = Dataset()
    # preenche o dataset com os dados

    f_regression = FRegression(threshold=0.05)
    selected_dataset = f_regression.fit_transform(dataset)