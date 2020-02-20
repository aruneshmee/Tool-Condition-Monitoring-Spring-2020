import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

data = pd.read_csv('comb_data.csv')

pearsoncorr = data.corr(method='pearson')

plt.subplots(figsize=(40,35))

svm = sb.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.8)

figure = svm.get_figure()    
figure.savefig('heatmap_hydrolic_press.png', dpi=400)
