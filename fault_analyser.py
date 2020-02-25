import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import statistics as ss
from scipy.stats import norm

# Importing dataset
data = pd.read_csv('comb_data.csv')

# Global Variables
df = pd.DataFrame(data)
num_col=len(df.axes[1])
num_row=len(df.axes[0])
