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
