import numpy as np
import pandas as pd
from typing import Sequence

class Dataset:
    
    def __init__(self, X=None, y=None, features=None, label=''):
        self.X = X if X is not None else np.array([])
        self.y = y if y is not None else np.array([])
        self.features = features if features is not None else []
        self.label = label
    
    def __len__(self):
        if self.X is not None:
            return len(self.X)
        else:
            return 0
    
    def shape(self):
        """
        Returns the shape of the dataset as a tuple of the form (n_samples, n_features)
        """
        return self.X.shape


    def describe(self):
        if self.X is not None:
            df = pd.DataFrame(data=self.X, columns=self.features)
            return df.describe()
        else:
            return None
    
    def from_random(self, n_samples: int, n_features: int, n_classes: int = 2, features: Sequence[str] = None, label: str = None):
        """
        Creates a Dataset object from random data
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return Dataset(X, y, features=features, label=label)

    def read_csv(self, filename, label_col=-1):
        """
        This implementation uses the dtype=object argument when creating the NumPy array from the Pandas DataFrame.
        This ensures that the array can contain elements of different types, including strings.
        """
        df = pd.read_csv(filename)
        self.features = list(df.columns.values[:-1])
        self.label = df.columns.values[-1]
        self.y = df.iloc[:, label_col].values
        df = df.drop(df.columns[label_col], axis=1)
        self.X = np.array(df, dtype=object)
        return self
      
    def read_tsv(self, filename, label=None):
        self.read_csv(filename,label,'\t')
 
    def write_csv(self,filename):
        df = pd.DataFrame(data=np.column_stack((self.X, self.y)), columns=self.features+[self.label])
        df.to_csv(filename, index=False)

    def write_tsv(self,filename):
        df = pd.DataFrame(data=np.column_stack((self.X, self.y)), columns=self.features+[self.label])
        df.to_tsv(filename, index=False)

    def from_dataframe(self, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame
        """
        self.features = df.columns.tolist()
        self.label = label
        
        if label:
            self.X = df.drop(label, axis=1).values
            self.y = df[label].values
        else:
            self.X = df.values
            self.y = None

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        data = np.column_stack((self.X, self.y)) if self.y is not None else self.X
        columns = self.features + [self.label] if self.label else self.features
        return pd.DataFrame(data=data, columns=columns)

    # Getters & Setters
    def get_X(self):
        return self.X

    def set_X(self,new):
        self.X = new

    def get_y(self):
        return self.y
    
    def get_features(self):
        return self.features

    def get_label(self):
        return self.label
    
    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset
        Returns
        -------
        numpy.ndarray (n_classes)
        """
        if self.y is None:
            raise ValueError("Dataset does not have a label")
        return np.unique(self.y)
    
    def get_mode(self) -> np.ndarray:
        """
        Returns the mode of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), 0, self.X)
    
    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        mode = self.get_mode()
        mean = np.mean(self.X, axis=0)
        return np.mean(np.abs(self.X - mode) ** 2, axis=0) - np.abs(mean - mode) ** 2
    
    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.median(self.X, axis=0)
    
    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmax(self.X, axis=0)

    # Methods
    def count_nulls(self):
        null_counts = {}
        for i, feature in enumerate(self.features):
            null_counts[feature] = np.sum(pd.isnull(self.X[:, i]))
            
        null_counts[self.label] = np.sum(pd.isnull(self.y))
        return null_counts
    
    def replace_nulls(self, strategy='most_frequent'):
        """
        Determines if a feature is numerical or string.
        If the feature is numerical, it uses the mean value to replace nulls.
        If the feature is string, it uses the mode to replace nulls.
        If a feature has mixed data types, the method will raise an error.
        """
        null_counts = self.count_nulls()
        for i, feature in enumerate(self.features):
            if null_counts[feature] > 0:
                if strategy == 'most_frequent':
                    non_null_values = self.X[:, i][~pd.isnull(self.X[:, i])]
                    if len(non_null_values) > 0:
                        most_frequent_value = pd.Series(non_null_values).mode()[0]
                    else:
                        most_frequent_value = None
                    self.X[:, i][pd.isnull(self.X[:, i])] = most_frequent_value
                elif strategy == 'mean':
                    non_null_values = [x for x in self.X[:, i] if pd.notna(x)]
                    if len(non_null_values) > 0:
                        mean_value = np.mean(non_null_values)
                    else:
                        mean_value = None
                    self.X[:, i][pd.isnull(self.X[:, i])] = mean_value


def main():
    # Create a new dataset instance and read in a CSV file
    dataset = Dataset()
    dataset.read_csv('../../datasets/sample.csv')

    # Print some basic statistics about the dataset
    print("Number of instances:", len(dataset))
    print("Number of features:", len(dataset.features))
    print("Shape: ", dataset.shape())
    print("Feature names:", dataset.features)
    print("Label name:", dataset.label)
    print("X: ", dataset.X, "\n")
    print("y: ", dataset.y, "\n")
    
    # Tratamento de valores nulos
    print("Null value counts before:", dataset.count_nulls(), "\n")
    dataset.replace_nulls()
    print("Null value counts after:", dataset.count_nulls(), "\n")

    print("Description of features:")
    print(dataset.describe())

    # Create a new dataset from a pandas DataFrame
    df = pd.read_csv('../../datasets/sample.csv')
    new_dataset = Dataset()
    new_dataset.from_dataframe(df)

    # Convert the dataset back to a pandas DataFrame
    df_new = new_dataset.to_dataframe()
    print("Converted DataFrame:")
    print(df_new)


if __name__ == "__main__":
    main()