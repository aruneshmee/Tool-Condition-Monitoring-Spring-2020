# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import statistics as ss
from scipy.stats import norm
import numpy as np
import scipy.stats

# Importing the file
file_name = input('Please enter the name of the file: ')
databyc = pd.read_csv(file_name)

# Finding the numbers of dimensions/features in the dataset
df = pd.DataFrame(databyc)
total_rows=len(df.axes[0])
total_cols=len(df.axes[1])
ml = []
stl = []

# Function to find the roots for finding the area under the curve
def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

ask = True

while ask:
    quest = input('Do want to see area between the curve y/n: ')
    if quest == 'y':
        g_1 = int(input('Enter first graph: '))
        g_2 = int(input('Enter second graph: '))
        print('-----------')
        print(' Graph: ')
        print('-----------')
        ml = []
        stl = []
        c = 0
        for _ in range(0,total_cols-1):
            x = databyc.iloc[:,c].values
            m = ss.mean(x)
            ml.append(m)
            st = np.std(x)
            stl.append(st)
            y = norm.pdf(x, m, st)
            #name = 'Country',(c+1)
            #print('STD ',(c+1),' : ',st)
            plt.plot(x,y,label = m)
            plt.xlabel('x')
            plt.ylabel('Normal Distribution')
            #plt.show()
            c+=1
        #plt.show()
        
        #Get point of intersect
        m1 = ml[g_1-1]
        m2 = ml[g_2-1]
        std1 = stl[g_1-1]
        std2 = stl[g_2-1]
        result = solve(m1,m2,std1,std2)
        #plot3=plt.plot(result,norm.pdf(result,m1,std1),'y')

        #Plots integrated area
        r = result[0]
        olap = plt.fill_between(x[x>r], 0, norm.pdf(x[x>r],m1,std1),alpha=0.4)
        olap = plt.fill_between(x[x<r], 0, norm.pdf(x[x<r],m2,std2),alpha=0.4)  

        area = norm.cdf(r,m2,std2) + (1.-norm.cdf(r,m1,std1))

        print('-'*38)
        print("Area under curves ", area)
        print('-'*38)

        plt.legend(loc="upper right")
        plt.grid()
        plt.show()
    elif quest == 'n':
        ask = False
    else:
        print(' ')
        print('Error...try again!\n')
