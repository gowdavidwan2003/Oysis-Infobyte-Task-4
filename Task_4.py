#importing library
import numpy as np
import tensorflow as tf
import pandas as pd 
import matplotlib.pyplot as plt

#importing data
sms = pd.read_csv("spam.csv", encoding = "ISO-8859-1", usecols=[0,1])
sms.head(5)

#renaming columns for better view
sms.columns = ['label', 'message']
sms.head(5)

#to check for null value
sms.isnull().sum()

from sklearn.preprocessing import OrdinalEncoder
categorical_columns = ['label'] 
encoder = OrdinalEncoder()
sms[categorical_columns] = encoder.fit_transform(sms[categorical_columns])
sms.head(5)

sms.duplicated().sum()

#delete duplicated values 
sms=sms.drop_duplicates()
sms.shape

#0 is for ham and 1 is for spam
sms['label'].value_counts()

sms['num_char']=sms['message'].apply(len)
sms.head(5)

print("for ham message")
print(sms[sms['label']==0][['num_char']].describe())


print("\nfor spam message")
print(sms[sms['label']==1][['num_char']].describe())

import seaborn as sns 
sns.histplot(sms[sms['label'] == 0]['num_char'], color='blue')
sns.histplot(sms[sms['label'] == 1]['num_char'], color='red')

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(sms['message']).toarray()
y=sms['label'].values

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(C=20.0,max_iter=1000)
lg.fit(X_train,y_train)
lg_pred = lg.predict(X_test)
lg_accuracy = accuracy_score(y_test, lg_pred)
lg_report = classification_report(y_test, lg_pred)
lg_accuracy*100

print("\nLogistic Regression Classifier:")
print(f"Accuracy: {lg_accuracy:.2f}")
print("Classification Report:\n", lg_report)

# Your text data
text_data = ['I wanna be rich','Hello i am Vidwan']


x_pred = cv.transform(text_data)

# Fit and transform the text data
y_pred= lg.predict(x_pred)
y_pred
