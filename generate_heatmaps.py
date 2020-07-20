# Importing the libraries
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

# Importing dataset
data = pd.read_csv('comb_data.csv')

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
figure.savefig('heatmap_hydrolic_press.png', dpi=400)
