ml = []
stl = []

# Function to find the roots for finding the area under the curve
def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

# Function to check the fault
def check_fault(ml, stl, limit):
    g=len(ml)
    for _ in range(0,len(ml)-1):
        m1 = ml[0]
        m2 = ml[g-1]
        std1 = stl[0]
        std2 = stl[g-1]
        g-=1
        area = norm.cdf(r,m2,std2) + (1.-norm.cdf(r,m1,std1))
        print('Area under Graph 1 and Graph ',g+1,' is {0:.2f}'.format(area))
        if area > limit:
            print('-'*45)
            print(f'Graph {g+1} crossed the threshold limit of {limit}')
            print('-'*45)
            break
        else:
            continue

ask = True
m = 8.0 
st = 2.0

x_min = 0.0
x_max = 16.0
x = np.linspace(x_min, x_max, 100)


while ask:
    quest = input('Do want to see area between the curve y/n: ')
    if quest == 'y':
        g_1 = int(input('Enter first graph: '))
        g_2 = int(input('Enter second graph: '))
        limit = float(input('Enter the threshold value that you want to check: '))
        print('----------------')
        print(' Fault Analysis ')
        print('----------------')
        ml = []
        stl = []
        c = 0
        for _ in range(0,5):
            #x = databyc.iloc[:,c].values
            #m = ss.mean(x)
            ml.append(m)
            #st = np.std(x)
            stl.append(st)
            y = norm.pdf(x, m, st)

            plt.plot(x, y, label=(f'Graph {c+1}: {m}'))
            plt.xlabel('x')
            plt.ylabel('Normal Distribution')
            #plt.show()
            c+=1
            m+=2
            st+=0.5
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

        check_fault(ml, stl, limit)
        #area = norm.cdf(r,m2,std2) + (1.-norm.cdf(r,m1,std1))

        plt.legend(loc="upper left")
        plt.grid()
        plt.show()
        ask = False
    else:
        print(' ')
        print('Error...try again!\n')
