# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Import necessary libraries**: Use `pandas` for efficient data manipulation and `numpy` for numerical operations.  
2. **Load the dataset**: Read the CSV file (`Placement_Data.csv`) into a pandas DataFrame and preview the first five records using `df.head()` for initial inspection.  
3. **Preserve the original data**: Create a copy of the original DataFrame to ensure the raw data remains intact for future reference or rollback.  
4. **Drop unnecessary columns**: Remove the `sl_no` (serial number) and `salary` columns from the dataset, as they are not required for the predictive modeling process.  
5. **Check for missing values**: Use `df.isnull().sum()` to identify any missing values in each column.  
6. **Detect duplicates**: Use `df.duplicated().sum()` to find and count duplicate rows that may affect model performance.  
7. **Encode categorical variables**: Apply `LabelEncoder` from `sklearn.preprocessing` to convert categorical string columns—such as `gender`, `ssc_b`, `hsc_b`, `hsc_s`, `degree_t`, `workex`, `specialisation`, and `status`—into numeric labels, making them suitable for machine learning models.  
8. **Define features and target**: Assign all columns except the last one (`status`) to the feature variable `x`, and set the target variable `y` as the `status` column, indicating whether a student was placed (binary classification).  
9. **Split the dataset**: Use `train_test_split` to divide the dataset into training and testing sets, with 80% for training and 20% for testing. Set `random_state=0` to ensure reproducibility of results.  
10. **Initialize the model**: Instantiate a logistic regression model using `LogisticRegression(solver="liblinear")`, which is suitable for small datasets.  
11. **Train the model**: Fit the logistic regression model using the training data (`x_train`, `y_train`) with `lr.fit()`.  
12. **Make predictions**: Predict the target variable for the test set (`x_test`) using the trained model.  
13. **Evaluate the model**:  
    - Use `accuracy_score(y_test, y_pred)` to calculate the model’s accuracy.  
    - Generate a **confusion matrix** using `confusion_matrix(y_test, y_pred)` to assess true vs. predicted classifications.  
    - Obtain a detailed performance summary using `classification_report(y_test, y_pred)`, which includes precision, recall, F1-score, and support for each class.  
14. **Prepare for user input**: Ensure any future input to the model matches the format and encoding of the features used during training (i.e., numeric representations of categorical attributes).


## Program and Output:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: PREETHI D 
RegisterNumber:  212224040250
*/
```
```
import pandas as pd
import numpy as np
df = pd.read_csv('Placement_Data.csv')
df.head()
```
![image](https://github.com/user-attachments/assets/127d325f-47d7-427e-a0a4-64544914a6bd)

```
df1 = df.copy()
df1 = df1.drop(["sl_no","salary"],axis = 1)
df1.head()
```
![image](https://github.com/user-attachments/assets/3215ba84-eaa0-43ee-942b-7d40d0d4ca39)

```
df1.isnull().sum()
```
![image](https://github.com/user-attachments/assets/a8030e62-5ef1-4588-aa32-5f18b1e97358)

![image](https://github.com/user-attachments/assets/bf82c31c-8184-4c22-8cbb-825daac806fe)

```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1['gender'] = le.fit_transform(df1['gender'])
df1['ssc_b'] = le.fit_transform(df1['ssc_b'])
df1['hsc_b'] = le.fit_transform(df1['hsc_b'])
df1['hsc_s'] = le.fit_transform(df1['hsc_s'])
df1['degree_t'] = le.fit_transform(df1['degree_t'])
df1['workex'] = le.fit_transform(df1['workex'])
df1['specialisation'] = le.fit_transform(df1['specialisation'])
df1['status'] = le.fit_transform(df1['status'])
df1
```
![image](https://github.com/user-attachments/assets/967125fe-9c8f-4448-ba7e-2bf8cb428bca)
```
x=df1.iloc[:,:-1]
x
```
![image](https://github.com/user-attachments/assets/74ec65c9-86eb-4238-bbed-c2b9d2b29d3b)
```
y=df1['status']
y
```
![image](https://github.com/user-attachments/assets/20df691f-8fa2-48e8-a1f0-a966c242f397)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
```
```
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/0f32ed8f-bd39-447c-993b-2d36d36a316b)

```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/f1ac11f4-3579-46ab-87ed-5a1838ce7a67)

```
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```
![image](https://github.com/user-attachments/assets/6cece211-e93b-4055-80d5-2ec4152dd0f5)

```
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
```
![image](https://github.com/user-attachments/assets/888142e1-b76a-4ca3-86c2-31025e8d46a8)

```
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
![image](https://github.com/user-attachments/assets/d887b409-b5a3-4611-8a78-22c90f1c9ec5)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
