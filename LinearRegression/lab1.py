import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

file_path = r'C:\Users\tstambaugh\source\repos\MachineLearning\FirstLab\FuelConsumptionCo2.csv'

df = pd.read_csv(file_path)

# take a look at the dataset
df.head()

# summarize the data
df.describe()
