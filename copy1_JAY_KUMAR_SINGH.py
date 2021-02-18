#!/usr/bin/env python
# coding: utf-8

# In[1]:



#importing all relevant packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from imblearn.over_sampling import SMOTE  
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


# In[2]:



# Performing basic statistics by importing the dataset 
eirdataset = pd.read_csv("EireJet.csv")
pd.set_option('display.max_columns', None) 
print(eirdataset.head())
print(eirdataset.shape)
print(eirdataset.info())
print(eirdataset.describe())


# In[3]:





eirdataset = eirdataset.dropna(how='any',axis=0) 
print(eirdataset.info())


# In[4]:





#  Changing categorical features to numeric features

eirdataset['Gender'] = eirdataset['Gender'].map({'Female':1, 'Male':0})
eirdataset['Frequent Flyer'] = eirdataset['Frequent Flyer'].map({'Yes':1, 'No':0})
eirdataset['Type of Travel'] = eirdataset['Type of Travel'].map({'Personal Travel':1, 'Business travel':0})
eirdataset['Class'] = eirdataset['Class'].map({'Eco':0,'Eco Plus':1 ,'Business':2 })
eirdataset['satisfaction'] = eirdataset['satisfaction'].map({'neutral or dissatisfied':0, 'satisfied':1})
print(eirdataset.info())


# In[5]:





# differntiating data into label set and feature sets
X = eirdataset.drop('satisfaction', axis = 1) # Feature 
Y = eirdataset['satisfaction'] # Label
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)


# In[6]:





# Normalizing numerical features so it has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)


# In[7]:




# Differntiating the processed data into training and test set 
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

print(X_train.shape)
print(X_test.shape)


# In[8]:





# Differntiating dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

print(X_train.shape)
print(X_test.shape)

# Now,  implementing Oversampling to balance the data; Synthetic Minority Oversampling Technique(SMOTE)
print("Number of observations in each class before oversampling (training data): \n", pd.Series(Y_train).value_counts())

smote = SMOTE(random_state = 101)
X_train,Y_train = smote.fit_sample(X_train,Y_train)

print("Number of observations in each class after oversampling (training data): \n", pd.Series(Y_train).value_counts())


# In[9]:



#tuning random forest parameter 'n_estimators' , using cross-validation with Grid Search


ranfo = RandomForestClassifier(criterion='entropy', max_features='auto', random_state=1)
gridpar = {'n_estimators': [50, 100, 150, 200, 250, 300]}

gds = GridSearchCV(estimator=ranfo, param_grid=gridpar, scoring='precision', cv=5)

"""
In the above GridSearchCV(), scoring parameter should be set as follows:
scoring = 'accuracy' when you want to maximize prediction accuracy
scoring = 'recall' when you want to minimize false negatives
scoring = 'precision' when you want to minimize false positives
scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
"""
gds.fit(X_train, Y_train)

bestp = gds.best_params_
print(bestp)
 
bestr = gds.best_score_ # Mean cross-validated score of the best_estimator
print(bestr)


# In[10]:





# Building random forest using the tuned parameter
ranfo = RandomForestClassifier(n_estimators=150, criterion='entropy', max_features='auto', random_state=1)
ranfo.fit(X_train,Y_train)
featimp = pd.Series(ranfo.feature_importances_, index=list(X)).sort_values(ascending=False)
print(featimp)

Y_pred = ranfo.predict(X_test)
print('Classification report: \n', metrics.classification_report(Y_test, Y_pred))

conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])


# In[12]:






#selecting features and defining the feature set again

X1 = eirdataset[['Inflight wifi service','Online boarding','Type of Travel', 'Class', 'Inflight entertainment','Ease of Online booking']]

feature_scaler = StandardScaler()
X1_scaled = feature_scaler.fit_transform(X1)


# In[13]:




# differntiating data into training and testing set and applying smote again

X1_train, X1_test, Y1_train, Y1_test = train_test_split( X1_scaled, Y, test_size = 0.3, random_state = 100)

smote = SMOTE(random_state = 101)
X1_train,Y1_train = smote.fit_sample(X1_train,Y1_train)


# In[14]:



ranfo = RandomForestClassifier(n_estimators=150, criterion='entropy', max_features='auto', random_state=1)
ranfo.fit(X1_train,Y1_train)

Y_pred = ranfo.predict(X1_test)
print('Classification report: \n', metrics.classification_report(Y1_test, Y_pred))

conf_mat = metrics.confusion_matrix(Y1_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])


# In[15]:



# Tuning AdaBoost parameter 'n_estimators' and using cross-validation using Grid Search method


abt = AdaBoostClassifier(random_state=1)
gridpar = {'n_estimators': [30,35,40,45,50,55,60]}

gds = GridSearchCV(estimator=abt, param_grid=gridpar, scoring='precision', cv=5)

"""
In the above GridSearchCV(), scoring parameter should be set as follows:
scoring = 'accuracy' when you want to maximize prediction accuracy
scoring = 'recall' when you want to minimize false negatives
scoring = 'precision' when you want to minimize false positives
scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
"""

gds.fit(X_train, Y_train)

bestp = gds.best_params_
print(bestp)

bestr = gds.best_score_ # Mean cross-validated score of the best_estimator
print(bestr)


# In[17]:



# Creating AdaBoost using tuned parameter

Abt = AdaBoostClassifier(n_estimators=50, random_state=1)
Abt.fit(X_train,Y_train)
featimp = pd.Series(Abt.feature_importances_, index=list(X)).sort_values(ascending=False)
print(featimp)

Y_pred = Abt.predict(X_test)
print('Classification report: \n', metrics.classification_report(Y_test, Y_pred))

conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])


# In[ ]:




# tuning the gradient boost parameter 'n_estimators' and using cross-validation using grid search method

gnib = GradientBoostingClassifier(random_state=1)
gridpar = {'n_estimators': [100,150,200], 'max_depth' : [9,10,11,12], 'max_leaf_nodes': [8,12,16,20,24,28,32]}

grd = GridSearchCV(estimator=gnib, param_grid=gridpar, scoring='precision', cv=5)

"""
In the above GridSearchCV(), scoring parameter should be set as follows:
scoring = 'accuracy' when you want to maximize prediction accuracy
scoring = 'recall' when you want to minimize false negatives
scoring = 'precision' when you want to minimize false positives
scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
"""

grd.fit(X_train, Y_train)

ppm = grd.best_params_
print(ppm)

result = grd.best_score_  #best score of the estimator , mean cross validates score
print(result)


# In[ ]:



# Forming Gradient Boost using the tuned parameter
Gnib = GradientBoostingClassifier(n_estimators=50, max_depth=11, max_leaf_nodes=32, random_state=1)
Gnib.fit(X_train,Y_train)
featimportance = pd.Series(Gradientboost.feature_importances_, index=list(X)).sort_values(ascending=False)
print(featimportance)

Y_pred = Gnib.predict(X_test)
print('Classification report: \n', metrics.classification_report(Y_test, Y_pred))

conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])


# In[ ]:




