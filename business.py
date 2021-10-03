'''

Author: Altair
'''

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix 
import mlflow

def forest_model(X_train,Y_train, N_ESTIMATORS):

    forest = RandomForestClassifier(n_estimators = N_ESTIMATORS, criterion = 'entropy', random_state = 0)
    forest.fit(X_train, Y_train)

    return forest

def create_and_save_model(N_ESTIMATORS):

    titanic = sns.load_dataset('titanic')

    titanic = titanic.drop(['deck', 'embark_town', 'alive', 'class', 'alone', 'adult_male', 'who'], axis=1)

    titanic = titanic.dropna(subset =['embarked', 'age'])

    labelencoder = LabelEncoder()

    titanic.iloc[:,2]= labelencoder.fit_transform(titanic.iloc[:,2].values)
    titanic.iloc[:,7]= labelencoder.fit_transform(titanic.iloc[:,7].values)

    X = titanic.iloc[:, 1:8].values 
    Y = titanic.iloc[:, 0].values 

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = forest_model(X_train,Y_train,N_ESTIMATORS)

    cm = confusion_matrix(Y_test, model.predict(X_test)) 

    TN, FP, FN, TP = cm.ravel()

    accuracy = (TP + TN) / (TP + TN + FN + FP)

    return accuracy

def load_model():

    model = mlflow.sklearn.load_model("model")

    return model

def predict(data):

    model = load_model()

    # data = [[3,1,21,0, 0, 0, 1]]

    pred = model.predict(data)

    return pred

def startpy():

    predict()

if __name__ == '__main__':

    startpy()