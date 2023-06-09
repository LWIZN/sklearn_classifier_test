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


kernel = 1.0 * RBF(1.0)

CLF_DICT = {
    "XGBClassifier": XGBClassifier(
        n_estimators=2, max_depth=2, learning_rate=1, objective="binary:logistic"
    ),
    "SVC": SVC(probability=True),
    "LogisticRegression": LogisticRegression(random_state=0),
    "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=3),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
    "RandomForestClassifier": RandomForestClassifier(max_depth=2, random_state=0),
    "GradientBoostingClassifier": GradientBoostingClassifier(
        n_estimators=100, learning_rate=1, max_depth=1, random_state=0
    ),
    "GaussianNB": GaussianNB(),
    "BernoulliNB": BernoulliNB(force_alpha=True),
    "MLPClassifier": MLPClassifier(max_iter=300, random_state=1),
    "AdaBoostClassifier": AdaBoostClassifier(n_estimators=100, random_state=0),
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
    "GaussianProcessClassifier": GaussianProcessClassifier(
        kernel=kernel, random_state=0
    ),
}

CLF = list(CLF_DICT.values())
CLF_NAME = list(CLF_DICT.keys())
