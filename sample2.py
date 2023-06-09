import pandas as pd
import seaborn as sns
import wiz.classifier as cla
from wiz.metrics import metrics_info_print
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def train(num, X_train, Y_train, X_test, Y_test, cmap):
    clf = cla.CLF[num]
    clf.fit(X_train, Y_train)

    # pred_n_visual(clf, X_train, Y_train, "train", num, cmap)
    pred_n_visual(clf, X_test, Y_test, "test", num, cmap)


def pred_n_visual(clf, X_target, Y_target, head, num, cmap):
    predict = clf.predict(X_target)
    tn, fp, fn, tp = confusion_matrix(Y_target, predict).ravel()
    metrics_info_print(head, Y_target, predict, tn, fp)

    score = []
    probability = clf.predict_proba(X_target)
    for i in probability:
        score.append(i[1])

    fpr, tpr, thresholds = roc_curve(Y_target, score)
    roc_auc_FULL = auc(fpr, tpr)
    plt.plot(
        fpr,
        tpr,
        color=cmap(num),
        label=f"{cla.CLF_NAME[num]} AUC: {roc_auc_FULL:.2f})",
    )


def main():
    data = pd.read_excel(r"data\Sarcopenia_20230606.xlsx")
    data_c = data.columns
    data_train = data.drop(
        [
            data_c[0],
            data_c[3],
            data_c[10],
        ],
        axis=1,
    )
    print(f"data:\n{data_train}")

    # correlation matrix
    df_corr = data_train.corr(method="pearson")
    sns.heatmap(df_corr, annot=True)
    plt.show()

    X_train, X_test, Y_train, Y_test = train_test_split(
        data_train, data["Sarcopenia"], test_size=0.2
    )

    cmap = plt.cm.get_cmap("hsv", len(cla.CLF_NAME))

    for i, j in enumerate(cla.CLF_NAME):
        print(f"\nClassifier: {j}")
        train(i, X_train, Y_train, X_test, Y_test, cmap)

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    # data = pd.read_excel(r"data\Sarcopenia_20230606.xlsx")
    # print(data.columns[0])
