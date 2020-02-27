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

# Section 2: Generate graph of each feature 
def  generate_graph():
    
    plot = True
    while plot:
        quest = int(input('Type the column number you want to plot or type 999 for plotting every graph or type 1234 to quit: '))
        if quest == 999:
            graph = 1
            for _ in range(0, num_col-2):
                data.plot.scatter(x='1',y=data.columns[graph],c='Result',cmap='coolwarm')
                plt.title('Scatter Plot: ' + data.columns[graph])
                plt.show()
                graph+=1

        elif quest < num_col:
            data.plot.scatter(x='1', y=data.columns[quest], c='Result', cmap='coolwarm')
            plt.title('Scatter Plot: ' + data.columns[quest])
            plt.show()
            
        elif quest == 1234:
            plot = False

        else:
            print('Incorrect Column number... try again')

     
# Section 3: Finding out top correlated features
def print_top_10_feat():
    # Dropping features that are correlated to itself
    def get_redundant_pairs(df):
        '''Get diagonal and lower triangular pairs of correlation matrix'''
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

    # Finding the top 10 features
    def get_top_abs_correlations(df):
        au_corr = df.corr().abs().unstack()
        labels_to_drop = get_redundant_pairs(df)
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        return au_corr[0:10]

    print('------  TOP 10 Linearly Correlated features -------\n')
    print(get_top_abs_correlations(df))

