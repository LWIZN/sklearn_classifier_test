import pandas as pd
import wiz.classifier as cla
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def select_function(num, X_train, Y_train, X_test, Y_test, df_X_train):
    clf = cla.CLF[num]

    clf.fit(X_train, Y_train)
    # Make predictions
    train_predict = clf.predict(X_train)
    df_X_train["train_target"] = train_predict
    df_X_train["true_target"] = Y_train

    # print(f'X_train: \n{df_X_train}\n')
    print(f"Train predict:\t{Y_train}\n\n\t\t{train_predict}\n")
    print(f"Accuracy over train set: {clf.score(X_train, Y_train)}\n")

    test_predict = clf.predict(X_test)
    # test_preds_proba = clf.predict_proba(X_test)

    print(f"Test predict:\t{Y_test}\n\n\t\t{test_predict}\n")
    # print(f'proba: {test_preds_proba}\n')
    print(f"Accuracy over test set: {clf.score(X_test, Y_test)}\n")


# Load dataset and generate dataframe
data = load_iris()
df_data = pd.DataFrame(data.data, columns=data.feature_names)
df_data["target"] = data.target  # add dataset's target to "target" tag

print(f"data: \n{df_data}\n")

# Split data for the train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    data["data"], data["target"], test_size=0.2
)
df_X_train = pd.DataFrame(X_train, columns=data.feature_names)

# Select and use classifier
for i, j in enumerate(cla.CLF_NAME):
    print(f"Classifier: {j}\n")
    select_function(i, X_train, Y_train, X_test, Y_test, df_X_train)
