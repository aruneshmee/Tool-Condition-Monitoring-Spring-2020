#  Tool Condition Monitoring (TCM)

Develop an intelligent real-time hard labor safety monitoring system, incorporating Artificial Intelligence analytics for early fault detection, diagnosis and prognosis. In simpler terms, acquire daignosos data from a machine for eg car engine and use it to analyse current condition of the machine and later followed by applying machine learning to predict the future failure of the machine. 

- Step 1: Acquire the data
- Step 2: Cleaning/Preprocessing of the dataset
- Step 3: Feature Scaling the dataset and creating dummmy variables if necessary
- Step 4: Apply ML algorithm to predict the results 
- Step 5: Visualise the results

Purpose: To save downtime required to repair or even replace the machine due to an unexpected failure or technical fault. Knowing the health of the machine before hand may help in periodic mantainance and help to save time and money. 

## Details:
- Description: The data is acquired from a hydrolic pressured injection. It data contains 9803 rows and 59 columns. Each columns representing one of the features of the hydrolic machine.
- Preprocessing: The dataset was first cleaned by converting the Pressure column into 0 and 1 (Good and Bad). All the values above 500 were coverted into 0 and all the below it as 1. Moreover, 6 columns had the same value in all its rows so the the columns were removed as they didn't contributed anything to the problem
- Splitting the dataset: The dataset was divided into training set and test set with a ratio of 3 is to 1. (75% and 25%)
- Feature Scaling: All the values in the dataset was scaled down to the same level.
- PCA was applied with n=2 to the training dataset and later followed by logistic regression
- An accuracy of 87.6% was achieved on the Test set
