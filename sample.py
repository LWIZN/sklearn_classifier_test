import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


CLF_NAME = ["XGBClassifier", "SVC", "LogisticRegression", "KNeighborsClassifier", "DecisionTreeClassifier",
            "RandomForestClassifier", "GradientBoostingCLassifier", "GaussianNB", "BernoulliNB",
            "MLPClassifier", "AdaBoostClassifier", "QuadraticDiscriminantAnalysis", "GaussianProcessClassifier"]

kernel = 1.0 * RBF(1.0)
CLF = [XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic'),
       SVC(probability=True),
       LogisticRegression(random_state=0),
       KNeighborsClassifier(n_neighbors=3),
       DecisionTreeClassifier(random_state=0),
       RandomForestClassifier(max_depth=2, random_state=0),
       GradientBoostingClassifier(
           n_estimators=100, learning_rate=1, max_depth=1, random_state=0),
       GaussianNB(),
       BernoulliNB(force_alpha=True),
       MLPClassifier(max_iter=300, random_state=1),
       AdaBoostClassifier(n_estimators=100, random_state=0),
       QuadraticDiscriminantAnalysis(),
       GaussianProcessClassifier(kernel=kernel, random_state=0)]


def select_function(num, X_train, Y_train, X_test, Y_test):
    clf = CLF[num]

    clf.fit(X_train, Y_train)

    train_predict = clf.predict(X_train)
    print(f'\ntrain_target:\n{list(Y_train)}')
    print(f'train_predict:\n{list(train_predict)}')
    print(
        f'\nacc over train set: {metrics.accuracy_score(Y_train, train_predict)}')
    print(
        f'precision over train set: {metrics.precision_score(Y_train, train_predict)}')
    print(
        f'recall over train set: {metrics.recall_score(Y_train, train_predict)}')
    print(
        f'F1 over train set: {metrics.f1_score(Y_train, train_predict)}')

    test_predict = clf.predict(X_test)
    print(f'\ntest_target:\n{list(Y_test)}')
    print(f'test_predict:\n{list(test_predict)}')
    print(
        f'\nacc over test set: {metrics.accuracy_score(Y_test, test_predict)}')
    print(
        f'precision over test set: {metrics.precision_score(Y_test, test_predict)}')
    print(
        f'recall over test set: {metrics.recall_score(Y_test, test_predict)}')
    print(
        f'F1 over test set: {metrics.f1_score(Y_test, test_predict)}')

    # Y_score = clf.decision_function(X_test)
    # display = metrics.PrecisionRecallDisplay.from_predictions(
    #     Y_test, Y_score, name=CLF_NAME[num])
    # _ = display.ax_.set_title("2-class Precision-Recall curve")
    # plt.show()


# Load dataset
data = pd.read_excel(r'data\data.xlsx')
print(f'data:\n{data}')

data_x = pd.read_excel(r'data\data.xlsx', usecols=["x"])
data_y = pd.read_excel(r'data\data.xlsx', usecols=["y"])
data_train = data.drop("target", axis=1)

# Show dataset
plt.plot(data_x, data_y, "o")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    data_train, data['target'], test_size=.2)

for i, j in enumerate(CLF_NAME):
    print(f'\nClassifier: {j}')
    select_function(i, X_train, Y_train, X_test, Y_test)
