# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries.
2.Load the dataset using pd.read_csv().
3.Display data types, basic statistics, and class distributions.
4.Visualize class distributions with a bar plot.
5.Scale feature columns using MinMaxScaler.
6.Encode target labels with LabelEncoder.
7.Split data into training and testing sets with train_test_split().
8.Train LogisticRegression with specified hyperparameters and evaluate the model using metrics and a confusion matrix plot.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("food_items (1).csv")

# Inspect the dataset
print("Name:Jijo.H.Jebas ")
print("Reg. No:212225040156 ")
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())

X_raw = df.iloc[:, :-1]
y_raw = df.iloc[:, -1:]

scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw.values.ravel())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123
)

# Model parameters
penalty = 'l2'
multi_class = 'multinomial'
solver = 'lbfgs'
max_iter = 1000

# Define logistic regression model
l2_model = LogisticRegression(
    random_state=123,
    penalty=penalty,
    multi_class=multi_class,
    solver=solver,
    max_iter=max_iter
)

l2_model.fit(X_train, y_train)

y_pred = l2_model.predict(X_test)

# Evaluate the model
print('Name: Jijo.H.Jebas')
print('Reg. No: 212225040156')
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print('Name: Jijo.H.Jebas')
print('Reg. No: 212225040156')
```

## Output:
<img width="960" height="465" alt="image" src="https://github.com/user-attachments/assets/23f21f6a-19a4-48a4-84bb-0e10bbab0bbb" />
<img width="950" height="453" alt="image" src="https://github.com/user-attachments/assets/4dc0ca42-c94f-49b4-8a46-ccf6a061f8f4" />
<img width="961" height="303" alt="image" src="https://github.com/user-attachments/assets/32110e71-e31e-4eb2-80a9-709c0f54666b" />





## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
