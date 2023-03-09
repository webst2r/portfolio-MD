import numpy as np
import sys
sys.path.append('..')
from Aula1.Dataset import Dataset


class VarianceThreshold:
    """
    Feature Selection -> VarianceThreshold:
    Features com variância (array-like) no dataset de treino mais baixa do que o threshold (float) devem ser removidas do dataset.
    """

    def __init__(self, threshold: float = 0.0):
        
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")

        # parameters
        self.threshold = threshold

        # attributes
        self.variance = None
    
    
    def fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """ 
        Fazer fit do modelo de acordo com os dados de treino.
        """
        self.variance = np.var(dataset.X, axis=0)
        return self
    

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Remover features com variância menor que o threshold.
        """
        X = dataset.X

        features_mask = self.variance > self.threshold
        X = X[:, features_mask]
        features = np.array(dataset.features)[features_mask]
        return Dataset(X=X, y=dataset.y, features=list(features), label=dataset.label)


    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fazer fit e de seguida transformar.
        """
        self.fit(dataset)
        return self.transform(dataset)