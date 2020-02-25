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

# Section 1: Generating heat maps of  the correlation between each factore
def generating_heatmap():

    # Using Pearson to find pairwise relation across all the columns
    pearsoncorr = data.corr(method='pearson')

    # Magnifying image 
    plt.subplots(figsize=(40,35))

    # Generating heatmap using seaborn
    svm = sb.heatmap(pearsoncorr, 
                xticklabels=pearsoncorr.columns,
                yticklabels=pearsoncorr.columns,
                cmap='RdBu_r',
                annot=True,
                linewidth=0.8)

    # Saving image
    figure = svm.get_figure()    
    return figure.savefig('heatmap_hydrolic_press.png', dpi=400)
