import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
