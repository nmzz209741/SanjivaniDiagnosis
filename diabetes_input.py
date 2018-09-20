import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
#from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
#from sklearn.model_selection import train_test_split # to split the data into two parts

# read the values
_preg = input('Number of times pregnant: ')
_glucose = input('Enter your glucose level: ')
_pressure = input('Diastolic Blood Pressure: ')
_insulin = input('Serum Insulin: ')
_bmi = input('Body Mass Index: ')
_age = input('Age(years): ')

# load the CSV file as a numpy matrix
dataset = np.loadtxt("pima-indians-diabetes.data", delimiter=",")
# separate the data from the target attributes
X = dataset[:,0:8]
y = dataset[:,8]

print ("Inputs: ")
print (_glucose,_pressure,_insulin,_bmi,_age,_preg)

X_test=[[_preg,_glucose,_pressure,0,_insulin,_bmi,0.134,_age]]

diab_dt = DecisionTreeClassifier().fit(X, y)
y_pred = diab_dt.predict(X_test)
output= int(y_pred[0])
if output == 0:
    print("Congratulations!! You are safe from diabetes.Keep visiting nearby hospitals for regular checkups.")
else:
    print("There are high chances of having diabetes. Please visit nearby hospital soon.")
