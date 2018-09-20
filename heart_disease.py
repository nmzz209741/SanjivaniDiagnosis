import pandas
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


location = "heart_disease.csv"
names = ['age','sex','chest_pain_types','resting_BP','serum_cholesterol','fast_blood_sugar','rest_ecg,max_heart_rate','exercise_induced_angina','old_peak','slope','ca','thal','outcome']
dataframe = pandas.read_csv(location, names=names)
array = dataframe.values
#takes all the column from 0-8
X = array[:,0:14]
#takes only 8th column
Y = array[:,14]
seed = 13

#Naive Bayes
kfold = model_selection.KFold(n_splits=10, random_state=None)
model = GaussianNB()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print (" Mean cross-validation score of Naive Bayes Classification")
print(results.mean())


# SVM Classification
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = SVC()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print (" Mean cross-validation score of SVM Classification")
print(results.mean())


# Logistic Regression Classification
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LogisticRegression()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print (" Mean cross-validation score of Logical Regression Classfication")
print(results.mean())

# LDA Classification
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LinearDiscriminantAnalysis()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print (" Mean cross-validation score of LDA Clasification")
print(results.mean())

# KNN Classification
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = KNeighborsClassifier()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print (" Mean cross-validation score of KNN Clasification")
print(results.mean())


# CART Classification
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = DecisionTreeClassifier()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print (" Mean cross-validation score of CART Clasification")
print(results.mean())
print(results)
