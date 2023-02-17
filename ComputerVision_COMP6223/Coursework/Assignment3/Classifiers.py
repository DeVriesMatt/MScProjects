import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

vgg_train = joblib.load("vgg19_/vgg19_train_feats.joblib")
vgg_validation = joblib.load("vgg19_/vgg19_val_feats.joblib")

vgg_x_train = np.array(vgg_train["feats"])
vgg_y_train = np.asarray(vgg_train["labels"])


vgg_x_val = np.array(vgg_validation["feats"])
vgg_y_val = np.asarray(vgg_validation["labels"])

# fit model no training data
model = XGBClassifier()
model.fit(vgg_x_train, vgg_y_train)

y_pred = model.predict(vgg_x_train)
accuracy = accuracy_score(vgg_y_val, y_pred)
print("Accuracy: {0}".format(accuracy * 100.0))

# names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
#          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#          "Naive Bayes", "QDA"]
#
# classifiers = [
#     KNeighborsClassifier(5),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma='scale', decision_function_shape='ovo'),
#     GaussianProcessClassifier(kernel=1.0 * RBF(1.0), multi_class='one_vs_all'),
#     DecisionTreeClassifier(max_depth=10),
#     RandomForestClassifier(max_depth=10, n_estimators=10, max_features=10),
#     MLPClassifier(alpha=1, max_iter=1000),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]
#
# vgg_train = joblib.load("vgg19_/vgg19_train_feats.joblib")
# vgg_validation = joblib.load("vgg19_/vgg19_val_feats.joblib")
#
# vgg_x_train = np.array(vgg_train["feats"])
# vgg_y_train = np.asarray(vgg_train["labels"])
#
#
# vgg_x_val = np.array(vgg_validation["feats"])
# vgg_y_val = np.asarray(vgg_validation["labels"])

# scores = []
# # i = 0
# # for name, clf in zip(names, classifiers):
# #     clf.fit(vgg_x_train, vgg_y_train)
# #     score = clf.score(vgg_x_val, vgg_y_val)
# #     scores[i] = score
# #     i += 1



