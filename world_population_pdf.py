# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import statistics as ss
from scipy.stats import norm

# Importing the dataset
dff = pd.read_csv('world.csv')

# Number of years
df = pd.DataFrame(databyc)
total_c=len(df.axes[0])

c = 1
for _ in range(0,total_c):

    x = dff.iloc[:,c].values
    m = ss.mean(x)
    st = np.std(x)
    print('STD of World Population in',(1959+c),' : ',st)
    print('Mean of World Population in',(1959+c),' : ',m)
    y = norm.pdf(x, m, st)
    plt.plot(x,y)
    plt.xlabel('Population in 100 Million')
    plt.ylabel('Normal Distribution')
    c+=1
plt.show()
