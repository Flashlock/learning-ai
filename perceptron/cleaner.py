import pandas as pd
import numpy as np
"""
Takes data from a dataset, and cleans it up for the perceptron to use in 2d
"""

def get_data(path: str):
    df = pd.read_csv(path, nrows=100)
    # create the key
    y = df.iloc[:, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # create the x values
    x = df.iloc[: ,[0, 2]].values
    
    return (x, y)
