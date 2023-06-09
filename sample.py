import pandas as pd
import wiz.classifier as cla
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics


def select_function(num, X_train, Y_train, X_test, Y_test):
    clf = cla.CLF[num]

    clf.fit(X_train, Y_train)

    # train_predict = clf.predict(X_train)

    # tn, fp, fn, tp = metrics.confusion_matrix(Y_train, train_predict).ravel()

    # print(f'\ntrain_target:\n{list(Y_train)}')
    # print(f'train_predict:\n{list(train_predict)}')
    # print(
    #     f'\nacc over train set: {metrics.accuracy_score(Y_train, train_predict)}')
    # print(
    #     f'precision over train set: {metrics.precision_score(Y_train, train_predict)}')
    # print(
    #     f'recall over train set: {metrics.recall_score(Y_train, train_predict)}')
    # print(
    #     f'F1 over train set: {metrics.f1_score(Y_train, train_predict)}')
    # print(
    #     f'specificity over train set: {metrics.f1_score(Y_train, train_predict)}')
    # print(
    #     f'specificity over train set: {tn / (tn+fp)}')

    test_predict = clf.predict(X_test)

    tn, fp, fn, tp = metrics.confusion_matrix(Y_test, test_predict).ravel()

    print(f"\ntest_target:\n{list(Y_test)}")
    print(f"test_predict:\n{list(test_predict)}")
    print(f"\nacc over test set: {metrics.accuracy_score(Y_test, test_predict)}")
    print(f"precision over test set: {metrics.precision_score(Y_test, test_predict)}")
    print(f"recall over test set: {metrics.recall_score(Y_test, test_predict)}")
    print(f"F1 over test set: {metrics.f1_score(Y_test, test_predict)}")
    print(f"specificity over test set: {tn / (tn+fp)}")

    # Y_score = clf.decision_function(X_test)
    # display = metrics.PrecisionRecallDisplay.from_predictions(
    #     Y_test, Y_score, name=CLF_NAME[num])
    # _ = display.ax_.set_title("2-class Precision-Recall curve")
    # plt.show()


# Load dataset
data = pd.read_excel(r"data\data.xlsx")
print(f"data:\n{data}")

data_train = data.drop("target", axis=1)

# Visualization dataset
# data_x = pd.read_excel(r"data\data.xlsx", usecols=["x"])
# data_y = pd.read_excel(r"data\data.xlsx", usecols=["y"])
# plt.plot(data_x, data_y, "o")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    data_train, data["target"], test_size=0.2
)

for i, j in enumerate(cla.CLF_NAME):
    print(f"\nClassifier: {j}")
    select_function(i, X_train, Y_train, X_test, Y_test)
