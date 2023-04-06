from sklearn import svm


def svm_(X_train, Y_train, X_test, Y_test):
    # Create model instance and fit model
    clf = svm.SVC(probability=True)
    clf.fit(X_train, Y_train)

    # Make predictions
    train_predict = clf.predict(X_train)
    train_proba = clf.predict_proba(X_train)
    print(
        f'Y_train: {Y_train}\ntrain_predict: {train_predict}\ntrain_proba: \n{train_proba}\naccuracy: {clf.score(X_train, Y_train)}\n')

    test_predict = clf.predict(X_test)
    test_proba = clf.predict_proba(X_test)
    print(
        f'Y_test: {Y_test}\ntest_predict: {test_predict}\ntest_proba: \n{test_proba}\naccuracy: {clf.score(X_test, Y_test)}\n')


if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split

    # Create a dataset and give any value
    X, Y = np.arange(10).reshape((5, 2)), range(5)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)
    print(f"X:\n{X}\nY:{Y}")

    svm_(X_train, Y_train, X_test, Y_test)
