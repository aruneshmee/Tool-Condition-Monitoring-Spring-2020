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

# Section 4: Plotting PDF for each feature
def plotting_pdf(num_col):
    # Grpah 1
    col = 1
    for _ in range(0, num_col-2):
        x = data.iloc[:,col].values
        M = ss.mean(x) #Mean of the column
        STD = np.std(x) #Standard Deviation of the colum
        y = norm.pdf(x, M, STD)
        plt.plot(x, y)
        plt.title('PDF curve for ' + data.columns[col])
        plt.xlabel('Distribution')
        plt.show()
        col+=1
  
# Section 5: Applying Machine Learning alg to the data
def apply_ML(num_col):
    # Dividing the data into X and Y arrays
    num_col -= 2
    X = data.iloc[:, 0: num_col].values
    y = data.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Applying PCA where n = 3
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 3)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance = pca.explained_variance_ratio_

    # Fitting Decision Tree DT Classification to the Training set
    # Logistic regression can also be applied instead DT
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    # Top left and bottom right give correctly predicted values
    # Top Right and bottom left give values that were predicted wrong
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    correct = (cm[0][0] + cm[1][1])
    incorrect = (cm[0][1] + cm [1][0])
    total = correct + incorrect
    print('--- Machine Learning results ---\n')
    print('Total Observations: ', total)
    print('Correct predictions: ', correct)
    print('Incorrect Predictions: ', incorrect)
    print('Accuracy achieved: {0:.2f}'.format(correct/total*100))
    
