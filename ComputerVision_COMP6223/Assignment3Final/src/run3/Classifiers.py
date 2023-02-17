import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier

import joblib
from xgboost import XGBClassifier

names = np.array(["Nearest Neighbors",
                  "Linear SVM",
                  "RBF SVM",
                  "Gaussian Process",
                  "Decision Tree",
                  "Random Forest",
                  "Neural Net",
                  "AdaBoost",
                  "XGBClassifier",
                  "GradientBoostingClassifier",
                  "BaggingClassifier"])

classifiers = np.array([
    KNeighborsClassifier(5),
    SVC(kernel="linear", C=0.025),
    SVC(gamma='scale', decision_function_shape='ovr'),
    GaussianProcessClassifier(kernel=1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10, n_estimators=10, max_features=10),
    MLPClassifier(hidden_layer_sizes=(100, 50, 50)),
    AdaBoostClassifier(),
    XGBClassifier(),
    GradientBoostingClassifier(),
    BaggingClassifier()
])


vgg_train = joblib.load("vgg19_/vgg19_train_feats.joblib")
vgg_validation = joblib.load("vgg19_/vgg19_val_feats.joblib")

vgg_x_train = np.array(vgg_train["feats"])
vgg_y_train = np.asarray(vgg_train["labels"])

vgg_x_val = np.array(vgg_validation["feats"])
vgg_y_val = np.asarray(vgg_validation["labels"])


res_train = joblib.load("resnet50_/resnet50_train_feats.joblib")
res_validation = joblib.load("resnet50_/resnet50_val_feats.joblib")

res_x_train = np.array(res_train["feats"])
res_y_train = np.asarray(res_train["labels"])


res_x_val = np.array(res_validation["feats"])
res_y_val = np.asarray(res_validation["labels"])

scores_vgg = np.zeros(classifiers.shape[0])
scores_res = np.zeros(classifiers.shape[0])

i = 0
for name, clf in zip(names, classifiers):
    print(name)

    clf.fit(vgg_x_train, vgg_y_train)
    score_vgg = clf.score(vgg_x_val, vgg_y_val)
    scores_vgg[i] = score_vgg
    print(score_vgg)

    clf.fit(res_x_train, res_y_train)
    score_res = clf.score(res_x_val, res_y_val)
    scores_res[i] = score_res
    print(score_res)

    i += 1
