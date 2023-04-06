import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


CLF_NAME = ["XGBClassifier", "SVC", "LogisticRegression", "KNeighborsClassifier", "DecisionTreeClassifier",
            "RandomForestClassifier", "GradientBoostingCLassifier", "GaussianNB", "BernoulliNB", "MultinomialNB",
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
       MultinomialNB(force_alpha=True),
       MLPClassifier(max_iter=300, random_state=1),
       AdaBoostClassifier(n_estimators=100, random_state=0),
       QuadraticDiscriminantAnalysis(),
       GaussianProcessClassifier(kernel=kernel, random_state=0)]


def select_function(num, X_train, Y_train, X_test, Y_test, df_X_train):
    clf = CLF[num - 1]

    clf.fit(X_train, Y_train)
    # Make predictions
    train_predict = clf.predict(X_train)
    df_X_train['train_target'] = train_predict
    df_X_train['true_target'] = Y_train

    # print(f'X_train: \n{df_X_train}\n')
    print(f'Train predict:\t{Y_train}\n\n\t\t{train_predict}\n')
    print(f'Accuracy over train set: {clf.score(X_train, Y_train)}\n')

    test_predict = clf.predict(X_test)
    # test_preds_proba = clf.predict_proba(X_test)

    print(f'Test predict:\t{Y_test}\n\n\t\t{test_predict}\n')
    # print(f'proba: {test_preds_proba}\n')
    print(f'Accuracy over test set: {clf.score(X_test, Y_test)}\n')


# Load dataset and generate dataframe
data = load_iris()
df_data = pd.DataFrame(data.data, columns=data.feature_names)
df_data['target'] = data.target  # add dataset's target to "target" tag

# print(f'data: \n{df_data}\n')

# Split data for the train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    data['data'], data['target'], test_size=.2)
df_X_train = pd.DataFrame(X_train, columns=data.feature_names)

# Select and use classifier
for i, j in enumerate(CLF_NAME):
    select_num = i + 1
    print(f'Classifier: {j}\n')
    select_function(select_num, X_train, Y_train, X_test, Y_test, df_X_train)
