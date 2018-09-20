import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split # to split the data into two parts
def heartprediction(_age,_sex,_cpt,_bp,_cholesterol,_sugar,_restEcg,_maxHeartRate):
# read the posted values from the UI
#     _age = input('Enter Your Age: ')
#     _sex = input('Enter your sex: ')
#     _cpt = input('Chest_pain_types: ')
#     _bp = input('Resting_BP : ')
#     _cholesterol = input('serum_cholesterol: ')
#     _sugar = input('bloodSugar: ')
#     _restEcg = input('restEcg: ')
#     _maxHeartRate = input('maxHeartRate: ')
    print ("heelolololo")

    data = pd.read_csv("G:\Desktop\Minor Project\heart_disease.csv",header=0)


    features=list(data.columns[0:13])
    train, test = train_test_split(data, test_size = 0.1)
    X_train = train[features]
    y_train = train.outcome
    X_test=[[_age,_sex,_cpt,_bp,_cholesterol,_sugar,_restEcg,_maxHeartRate,0,2.5,3,0,6]]

    print ("For input: ")
    print (X_test)
    heart_dt = SVC(kernel="linear", C=1.0).fit(X_train, y_train)
    y_pred = heart_dt.predict(X_test)
    output= int(y_pred[0])

    if output == 0:
        print("Congratulations!! You are safe from heart disease.Keep visiting nearby hospitals for regular checkups.")
    else:
        print("There are high chances of having heart disease. Please visit nearby hospital soon.")
