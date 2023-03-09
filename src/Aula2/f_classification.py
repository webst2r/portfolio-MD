import numpy as np
from scipy import stats
import sys
sys.path.append('..')
from Aula1.Dataset import Dataset


class FClassification:
    def __init__(self):
        self.f_values = None
        self.p_values = None
    
    def fit(self, dataset: Dataset):
        classes = np.unique(dataset.y)
        n_classes = len(classes)
        n_features = dataset.X.shape[1]
        self.f_values = np.zeros(n_features)
        self.p_values = np.zeros(n_features)

        for i in range(n_features):
            feature = dataset.X[:, i]
            f_values = np.zeros(n_classes - 1)
            for j in range(n_classes):
                mask = (dataset.y == classes[j])
                feature_class = feature[mask]
                f_values[j-1], _ = stats.f_oneway(feature_class)
            self.f_values[i] = np.mean(f_values)
            self.p_values[i] = stats.f.sf(self.f_values[i], n_classes-1, dataset.X.shape[0]-n_classes)

    def transform(self, dataset: Dataset, threshold: float = 0.05) -> Dataset:
        selected_features = dataset.X[:, self.p_values < threshold]
        selected_dataset = Dataset()
        selected_dataset.X = selected_features
        selected_dataset.y = dataset.y
        selected_dataset.features = [dataset.features[i] for i in range(len(dataset.features)) if self.p_values[i] < threshold]
        selected_dataset.label = dataset.label
        return selected_dataset


    def fit_transform(self, dataset: Dataset, threshold: float = 0.05) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset, threshold)