import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import numpy as np


def keyToID(selected, X_test):
    dataset = pd.read_csv('dataset.txt', sep=";")
    selectedKeyArray = []
    for i, row in dataset.iterrows():
        for j in range(len(selected)):
            if dataset.loc[i, 'Id'] == selected[j]:
                selectedKeyArray.append(dataset.loc[i, 'Key'])
    df = pd.DataFrame(selectedKeyArray)
    df.to_excel("selected.xlsx")

    testSetIDs = []
    for i, row in dataset.iterrows():
        for j in range(len(X_test)):
            if dataset.loc[i, 'Id'] == X_test[j][0]:
                testSetIDs.append(dataset.loc[i, 'Key'])
    df = pd.DataFrame(testSetIDs)
    df.to_excel("test_ids.xlsx")

def logisticRegression(X_train, X_test, y_train, y_test):
    classifier_lr = LogisticRegression(solver='liblinear')
    classifier_lr.fit(X_train, y_train)

    # Test set sonuçlarının tahmin edilmesi
    y_pred_lr = classifier_lr.predict(X_test)

    selected_ids = []
    for i in range(len(X_test)):
        if y_pred_lr[i] == 1:
            selected_ids.append(X_test[i][0])

    # print(selected_ids)
    keyToID(selected_ids, X_test)

    # Confusion Matrix'in oluşturulması
    cm_lr = confusion_matrix(y_test, y_pred_lr)

    # Modelin tutarlılığının hesaplanması
    print(f'**Regression Classifier Accuracy Score is {classifier_lr.score(X_train, y_train)} for Train Data Set**')
    print(f'**Regression Classifier Accuracy Score is {classifier_lr.score(X_test, y_test)} for Test Data Set**')
    print(f'**Regression Classifier F1 Score is {f1_score(y_test, y_pred_lr)}**')
    print(f'**Confusion Matrix for Regression Classifer {cm_lr}**')


def gaussianNB(X_train, X_test, y_train, y_test):
    # Training setiminin Gaussian modeline uyarlanması
    classifier_gnb = GaussianNB()
    classifier_gnb.fit(X_train, y_train)

    # Test set sonuçlarının tahmin edilmesi
    y_pred_gnb = classifier_gnb.predict(X_test)

    selected_ids = []
    for i in range(len(X_test)):
        if y_pred_gnb[i] == 1:
            selected_ids.append(X_test[i][0])

    # print(selected_ids)
    keyToID(selected_ids, X_test)

    # Confusion Matrix'in oluşturulması
    cm_gnb = confusion_matrix(y_test, y_pred_gnb)
    print(f'**--------------------------------------------------------------------------------**')
    print(f'**GaussianNB Classifier Accuracy Score is {classifier_gnb.score(X_train, y_train)} for Train Data Set**')
    print(f'**GaussianNB Classifier Accuracy Score is {classifier_gnb.score(X_test, y_test)} for Test Data Set**')
    print(f'**GaussianNB Classifier F1 Score is {f1_score(y_test, y_pred_gnb)}**')
    print(f'**Confusion Matrix for GaussianNB Classifer {cm_gnb}**')







def multinomialNB(X_train, X_test, y_train, y_test):
    # Training setiminin Naive Bayes modeline uyarlanması
    classifier_nb = MultinomialNB(alpha=0.1)
    classifier_nb.fit(X_train, y_train)

    # Test set sonuçlarının tahmin edilmesi
    y_pred_nb = classifier_nb.predict(X_test)

    #print(y_pred_nb)
    print(X_test)

    selected_ids = []
    for i in range(len(X_test)):
        if y_pred_nb[i] == 1:
            selected_ids.append(X_test[i][0])

    #print(selected_ids)
    keyToID(selected_ids, X_test)

    # Confusion Matrix'in oluşturulması
    cm_nb = confusion_matrix(y_test, y_pred_nb)

    print(f'**--------------------------------------------------------------------------------**')
    print(f'**MultinomialNB Classifier Accuracy Score is {classifier_nb.score(X_train, y_train)} for Train Data Set**')
    print(f'**MultinomialNB Classifier Accuracy Score is {classifier_nb.score(X_test, y_test)} for Test Data Set**')
    print(f'**MultinomialNB Classifier F1 Score is {f1_score(y_test, y_pred_nb)}**')
    print(f'**Confusion Matrix for MultinomialNB Classifer {cm_nb}**')


