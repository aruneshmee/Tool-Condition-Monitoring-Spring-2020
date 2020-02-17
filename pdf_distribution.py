# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import statistics as ss
from scipy.stats import norm
import numpy as np
import scipy.stats

x_min = 0.0
x_max = 16.0

mean = 8.0 
std = 2.0

x = np.linspace(x_min, x_max, 100)

# Loop to graph
change = 2
for _  in range(0,7):
    print('-----------')
    print(' Graph: ', change-1)
    print('-----------')
    if change%2==0:
        mean+=1
        y = scipy.stats.norm.pdf(x,mean,std)
        plt.plot(x,y, color='coral',label = mean)
        mean-=1
        y = scipy.stats.norm.pdf(x,mean,std)
        plt.plot(x,y, color='blue', label = mean)
        mean+=1
        plt.title('Different MEAN, same STD',fontsize=10)
        print('STD for Both Curves: ', std)
    else:
        std+=0.5
        y = scipy.stats.norm.pdf(x,mean,std)
        plt.plot(x,y, color='coral',label = mean)
        print('STD for Orange Curve: ', std)
        std-=0.5
        y = scipy.stats.norm.pdf(x,mean,std)
        plt.plot(x,y, color='blue', label = mean)
        print('STD for Blue Curve: ', std)
        std+=0.5
        plt.title('Different STD, same MEAN',fontsize=10)
    
    plt.legend(loc="upper left")
    plt.grid()

    plt.xlim(x_min,x_max)
    plt.ylim(0,0.25)

    plt.xlabel('x')
    plt.ylabel('Normal Distribution')

    plt.show()
    change+=1
