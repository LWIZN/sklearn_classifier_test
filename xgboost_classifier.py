import xgboost as xgb


def xgboost_classifier(X_train, Y_train, X_test, Y_test, df_X_train):
    # Create model instance and fit model
    clf = xgb.XGBClassifier(n_estimators=2, max_depth=2,
                            learning_rate=1, objective='binary:logistic')
    clf.fit(X_train, Y_train)

    # Make predictions
    train_predict = clf.predict(X_train)
    df_X_train['train_target'] = train_predict
    df_X_train['true_target'] = Y_train
    train_proba = clf.predict_proba(X_train)

    print(f'X_train: \n{df_X_train}\n')
    print(f'Train predict:\t{Y_train}\n\n\t\t{train_predict}\n')
    print(f'Accuracy over train set: {clf.score(X_train, Y_train)}\n')

    test_predict = clf.predict(X_test)
    test_proba = clf.predict_proba(X_test)

    print(f'Test predict:\t{Y_test}\n\n\t\t{test_predict}\n')
    # print(f'proba: {test_preds_proba}\n')
    print(f'Accuracy over test set: {clf.score(X_test, Y_test)}\n')


if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split

    # Create a dataset and give any value
    X, Y = np.arange(10).reshape((5, 2)), range(5)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)
    print(f"X: {X}, Y: {Y}")
    xgboost_classifier(X_train, Y_train, X_test, Y_test)
