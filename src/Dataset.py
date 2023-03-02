import pandas as pd
import numpy as np

class Dataset:
    def __init__(self):
        self.X = np.array([])
        self.y = np.array([])
        self.features = []
        self.label = ''
    
    def __len__(self):
        if self.X is not None:
            return len(self.X)
        else:
            return 0
         

    def describe(self):
        if self.X is not None:
            df = pd.DataFrame(data=self.X, columns=self.features)
            return df.describe()
        else:
            return None

    # Read
    """
    This implementation uses the dtype=object argument when creating the NumPy array from the Pandas DataFrame.
    This ensures that the array can contain elements of different types, including strings.
    """
    def read_csv(self, filename, label_col=-1):
        df = pd.read_csv(filename)
        self.features = list(df.columns.values[:-1])
        self.label = df.columns.values[-1]
        self.y = df.iloc[:, label_col].values
        df = df.drop(df.columns[label_col], axis=1)
        self.X = np.array(df, dtype=object)
        return self
      
    def read_tsv(self, filename, label=None):
        self.read_csv(filename,label,'\t')
    
    # Write
    def write_csv(self,filename):
        df = pd.DataFrame(data=np.column_stack((self.X, self.y)), columns=self.features+[self.label])
        df.to_csv(filename, index=False)

    def write_tsv(self,filename):
        df = pd.DataFrame(data=np.column_stack((self.X, self.y)), columns=self.features+[self.label])
        df.to_tsv(filename, index=False)


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
    


    
    """
    The method checks the type of each feature and determines whether it is numerical or string. If the feature is numerical, it uses the mean value to replace nulls.
    If the feature is string, it uses the mode to replace nulls. If a feature has mixed data types, the method will raise an error.
    """
    def replace_nulls(self, strategy='most_frequent'):
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
    dataset.read_csv('../data/sample.csv')

    # Replace null values with the most frequent value
    # dataset.replace_nulls()

    # Print some basic statistics about the dataset
    print("Number of instances:", len(dataset))
    print("Number of features:", len(dataset.features))
    print("Feature names:", dataset.features)
    print("Label name:", dataset.label)
    print("X: ", dataset.X, "\n")
    print("y: ", dataset.y, "\n")
    
    # tratamento de nulos
    print("Null value counts before:", dataset.count_nulls(), "\n")
    dataset.replace_nulls()
    print("Null value counts after:", dataset.count_nulls(), "\n")

    print("Description of features:")
    print(dataset.describe())

main()