import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import cohen_kappa_score, make_scorer, accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import chi2, mutual_info_classif
from copy import deepcopy
import matplotlib.pyplot as plt

"""
This script uses a 70/30 train/test split to compare the performance of different models.
It also prints test accuracy, balanced accuracy, classification report, and confusion matrix.
"""

def notrans(X):
    return X

# ignore warnings
simplefilter(action='ignore', category=ConvergenceWarning)
simplefilter(action='ignore', category=FutureWarning)

# load data
X_all = pd.read_csv("feature_table.csv", header=None).to_numpy()
Y = pd.read_csv("Y.csv", header=None).to_numpy().reshape((-1,))

# 70/30 train/test split
X_train_all, X_test_all, Y_train, Y_test = train_test_split(
    X_all,
    Y,
    test_size=0.30,
    stratify=Y,
    random_state=42
)

# get sort indices of mutual information scores using training data only
disMask = [False, False] + ([True] * (X_train_all.shape[1] - 2))
mi_scores = mutual_info_classif(X_train_all, Y_train, discrete_features=disMask)
mi_inds = np.argsort(mi_scores)

# get sort indices of chi squared scores using training data only
chi_scores = chi2(X_train_all, Y_train)[0]
chi_inds = np.argsort(chi_scores)

# classifiers used and their names
classifiers = [LogisticRegression(penalty="l2", max_iter=10000), RandomForestClassifier(), LinearSVC()]
strClf = ["Logistic Regression", "Random Forest", "Linear SVM"]

# transformations used and their names
scaler1 = MinMaxScaler()
scaler2 = StandardScaler()
trans = [notrans, scaler1.fit_transform, np.log1p]
strTrans = ["None", "MinMax", "log(x+1)"]

# parameter set for grid search for each classifier
parameterSets = [
    {'C': [0.1, 1, 10], 'solver': ["lbfgs"]},
    {'criterion': ['gini', 'entropy'], 'n_estimators': [100, 200], 'bootstrap': [True]},
    [
        {'penalty': ['l2'], 'C': [0.1, 1], 'max_iter': [10000], 'loss': ['hinge', 'squared_hinge']},
        {'penalty': ['l1'], 'C': [0.1, 1], 'max_iter': [10000], 'loss': ['squared_hinge'], 'dual': [False]}
    ]
]

# File scores are outputted to
fp = open("ModelComparison_latest.txt", "w")
fp.write("Train/Test Split: 70/30, stratified, random_state=42\n\n")

max_score = 0.0
best_balanced_score = 0.0
winnerString = ""
best_model = None
best_X_test = None

for n_clf, clf in enumerate(classifiers):

    fp.write(str(strClf[n_clf]) + ":\n")

    for n_trans in range(0, 3):

        fp.write("\t" + str(strTrans[n_trans]) + ":\n")

        # copy features before transformation
        X_train = deepcopy(X_train_all)
        X_test = deepcopy(X_test_all)

        # transform features
        X_train[1:3] = trans[n_trans](X_train[1:3] + np.finfo(float).eps)
        X_test[1:3] = trans[n_trans](X_test[1:3] + np.finfo(float).eps)

        for n_feat in [3500]:

            fp.write("\t\tChi_feat: " + str(n_feat) + "\n")

            # remove unimportant features using chi squared
            X_selected = X_train[:, chi_inds[n_feat:]]
            X_test_selected = X_test[:, chi_inds[n_feat:]]

            parameters = parameterSets[n_clf]

            # grid search only on training data
            GS = GridSearchCV(clf, parameters, n_jobs=-1)
            GS.fit(X_selected, Y_train)

            # test on unseen 30 percent data
            y_pred = GS.best_estimator_.predict(X_test_selected)

            test_acc = accuracy_score(Y_test, y_pred)
            test_bal_acc = balanced_accuracy_score(Y_test, y_pred)

            fp.write(
                "\t\t\tCV Acc:" + "{:0.16f}".format(GS.best_score_) +
                "\tTest Acc:" + "{:0.16f}".format(test_acc) +
                "\tBalanced Acc:" + "{:0.16f}".format(test_bal_acc) +
                "\t"
            )

            for (x, y) in GS.best_params_.items():
                fp.write(str(x) + ":" + str(y) + " ")
            fp.write("\n")
            fp.flush()

            if test_acc > max_score:
                winnerString = strClf[n_clf] + " is the winner with " + str(n_feat) + " features removed using Chi_squared, " + strTrans[n_trans] + " feature transformation, and the following params:"
                winnerString += str(GS.best_params_)
                max_score = test_acc
                best_balanced_score = test_bal_acc
                best_model = GS.best_estimator_
                best_X_test = X_test_selected

            # repeat above for MI
            fp.write("\t\tMI_feat: " + str(n_feat) + "\n")

            # remove unimportant features using MI
            X_selected = X_train[:, mi_inds[n_feat:]]
            X_test_selected = X_test[:, mi_inds[n_feat:]]

            parameters = parameterSets[n_clf]

            # grid search only on training data
            GS = GridSearchCV(clf, parameters, n_jobs=-1)
            GS.fit(X_selected, Y_train)

            # test on unseen 30 percent data
            y_pred = GS.best_estimator_.predict(X_test_selected)

            test_acc = accuracy_score(Y_test, y_pred)
            test_bal_acc = balanced_accuracy_score(Y_test, y_pred)

            fp.write(
                "\t\t\tCV Acc:" + "{:0.16f}".format(GS.best_score_) +
                "\tTest Acc:" + "{:0.16f}".format(test_acc) +
                "\tBalanced Acc:" + "{:0.16f}".format(test_bal_acc) +
                "\t"
            )

            for (x, y) in GS.best_params_.items():
                fp.write(str(x) + ":" + str(y) + " ")
            fp.write("\n")
            fp.flush()

            if test_acc > max_score:
                winnerString = strClf[n_clf] + " is the winner with " + str(n_feat) + " features removed using Mutual Information, " + strTrans[n_trans] + " feature transformation, and the following params:"
                winnerString += str(GS.best_params_)
                max_score = test_acc
                best_balanced_score = test_bal_acc
                best_model = GS.best_estimator_
                best_X_test = X_test_selected

    fp.write("\n\n")


# repeat above using all features
fp.write("All features\n")

for n_clf, clf in enumerate(classifiers):

    fp.write("\t" + str(strClf[n_clf]) + ":\n")

    for n_trans in range(0, 3):

        fp.write("\t\t" + str(strTrans[n_trans]) + ":\n")

        # copy features before transformation
        X_train = deepcopy(X_train_all)
        X_test = deepcopy(X_test_all)

        # transform features
        X_train[1:3] = trans[n_trans](X_train[1:3] + np.finfo(float).eps)
        X_test[1:3] = trans[n_trans](X_test[1:3] + np.finfo(float).eps)

        parameters = parameterSets[n_clf]

        # grid search only on training data
        GS = GridSearchCV(clf, parameters, n_jobs=-1)
        GS.fit(X_train, Y_train)

        # test on unseen 30 percent data
        y_pred = GS.best_estimator_.predict(X_test)

        test_acc = accuracy_score(Y_test, y_pred)
        test_bal_acc = balanced_accuracy_score(Y_test, y_pred)

        fp.write(
            "\t\t\tCV Acc:" + "{:0.16f}".format(GS.best_score_) +
            "\tTest Acc:" + "{:0.16f}".format(test_acc) +
            "\tBalanced Acc:" + "{:0.16f}".format(test_bal_acc) +
            "\t"
        )

        for (x, y) in GS.best_params_.items():
            fp.write(str(x) + ":" + str(y) + " ")
        fp.write("\n")
        fp.flush()

        if test_acc > max_score:
            winnerString = strClf[n_clf] + " is the winner using all features, " + strTrans[n_trans] + " feature transformation, and the following params:"
            winnerString += str(GS.best_params_)
            max_score = test_acc
            best_balanced_score = test_bal_acc
            best_model = GS.best_estimator_
            best_X_test = X_test


fp.write("\n\nBest Performing Model:\n")
fp.write("Test Accuracy: " + str(max_score) + "\n")
fp.write("Balanced Accuracy: " + str(best_balanced_score) + "\n")
fp.write(str(winnerString) + "\n")

# final report for best model
best_pred = best_model.predict(best_X_test)

fp.write("\n\nClassification Report for Best Model:\n")
fp.write(classification_report(Y_test, best_pred))

fp.write("\nConfusion Matrix for Best Model:\n")
fp.write(str(confusion_matrix(Y_test, best_pred)))

fp.close()

print("Done. Results saved to ModelComparison.txt")
print("Best Test Accuracy:", max_score)
print("Best Balanced Accuracy:", best_balanced_score)
print("\nClassification Report:")
print(classification_report(Y_test, best_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, best_pred))