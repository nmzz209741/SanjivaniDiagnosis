import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
from sklearn.model_selection import train_test_split
import seaborn as sns # used for plot interactive graph. I like it most for plot
from sklearn.ensemble import RandomForestClassifier # for random forest classifier

# Any results we write to the current directory are saved as output.
data = pd.read_csv("b_cancer.csv",header=0)# here header 0 means the 0 th row is our coloumn 
# have a look at the data
print(data.head(2))
data.drop("Unnamed: 32",axis=1,inplace=True)
#print (data.columns)
# like this we also don't want the Id column for our analysis
data.drop("id",axis=1,inplace=True)

features_mean= list(data.columns[1:11])
features_se= list(data.columns[11:20])
features_worst=list(data.columns[21:31])

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
data.describe()
#print data[0: 10]
#sns.countplot(data['diagnosis'],label="Count")
corr = data[features_mean].corr() # .corr is used for find corelation


			   
#prediction_var = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']
#prediction_var = features_mean
#low acuracy|||above both have low accuracy

prediction_var = features_worst
# the accuracy for RandomForest invcrease it means the value are more catogrical in Worst part
#lets get the important features
model=RandomForestClassifier(n_estimators=100,criterion="entropy")



#featimp = pd.Series(model.feature_importances_, index=prediction_var).sort_values(ascending=False)
#print(featimp)
# this is the property of Random Forest classifier that it provide us the importance 
# of the features used

#prediction_var = ['concave points_worst','radius_worst','area_worst','perimeter_worst','concavity_worst'] 
prediction_var = ['concave points_worst','radius_worst','area_worst','perimeter_worst','concavity_worst'] 

train, test = train_test_split(data, test_size = 0.3)

#RANDOMFORREST
train_X = train[prediction_var]# taking the training data input 
train_Y = train.diagnosis# This is output of our training data
# same we have to do for test
test_X= test[prediction_var] # taking test data inputs
test_Y =test.diagnosis   #output value of test dat
#model=RandomForestClassifier(n_estimators=100, criterion="entropy")
model.fit(train_X,train_Y)
print("------------------------------------")
# read the values
_concave_points_worst = input('concave points_worst: ')
_radius_worst = input('radius_worst: ')
_area_worst = input('area_worst: ')
_perimeter_worst = input('perimeter_worst: ')
_concavity_worst = input('concavity_worst: ')


print ("inputs")
print (_concave_points_worst, _radius_worst,_area_worst, _perimeter_worst, _concavity_worst)
X_test=[[_concave_points_worst, _radius_worst,_area_worst, _perimeter_worst, _concavity_worst]]

prediction=model.predict(X_test)
print (prediction)
#print metrics.accuracy_score(prediction,test_Y)
output= int(prediction[0])
if output == 0:
    print("Congratulations!! You are safe from breast cancer.Keep visiting nearby hospitals for regular checkups.  ")
else:
    print("There are high chances of having breast cancer. Please visit nearby hospital soon.")