# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data. 
2.Print the placement data and salary data. 
3.Find the null and duplicate values. 
4.Using logistic regression find the predicted values of accuracy , confusion matrices

## Program:
```python3
#Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
#Developed by: Sanjay Ragavendar M K 
#RegisterNumber:  212222100045
```
```py
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
```
```py
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()
```
```py
data1.isnull().sum()
```
```py
data1.duplicated().sum()
```
```py
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )
data1["status"]=le.fit_transform(data1["status"])
data1
```
```py
x=data1.iloc[:,:-1]
x
```
```py
y=data1["status"]
y
```
```py
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```py
from sklearn.linear_model  import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
```py
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
```py
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```
```py
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
```
```py
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
##  Output:
### Opening File:
![image](https://github.com/SanjayRagavendar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/91368803/5d0b2dd9-511d-45a6-86c3-c2029ebbbc64)
### Droping File
![image](https://github.com/SanjayRagavendar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/91368803/85a84ba7-965d-448b-bccd-ec7fbc7a25e5)
### Isnull check
![image](https://github.com/SanjayRagavendar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/91368803/c0d507d9-7b5a-4c5a-ba3b-264d9ebc57b4)
### Duplicated()
![image](https://github.com/SanjayRagavendar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/91368803/e2e1cfb6-00fe-450f-801b-7033664c2b89)
### Label Encoding
![image](https://github.com/SanjayRagavendar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/91368803/5aa62739-e62e-4063-9e94-d9261b1679ca)
### Spliting x,y
![image](https://github.com/SanjayRagavendar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/91368803/0e2d5ae8-5748-4d43-8c42-7234356dd55f)
![image](https://github.com/SanjayRagavendar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/91368803/2f15a690-062f-4758-be6b-0b65f175fa55)

### Prediction Score
![image](https://github.com/SanjayRagavendar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/91368803/5a81c6af-9bc1-45bf-a5cc-2357bee24743)

### Testing accuracy
![image](https://github.com/SanjayRagavendar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/91368803/39cb845f-9457-4909-9c4b-b34332dae6eb)

### Confusion Matrix
![image](https://github.com/SanjayRagavendar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/91368803/c1f8ffdb-711f-46c3-b8ec-a5112219d3d3)

### Classification Report
![image](https://github.com/SanjayRagavendar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/91368803/747a4648-b6ba-4874-9916-478d75b82878)

### Testing Model
![image](https://github.com/SanjayRagavendar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/91368803/c6f99824-421a-4f71-8c24-dce9582f3859)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
