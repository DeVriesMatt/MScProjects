import joblib
import numpy as np

vgg_train = joblib.load("vgg19_/vgg19_train_feats.joblib")

vgg_x_train = np.array(vgg_train["feats"])
vgg_y_train = np.asarray(vgg_train["labels"])
print(vgg_x_train.shape)
print(vgg_y_train.shape)


vgg_validation = joblib.load("vgg19_/vgg19_val_feats.joblib")
vgg_x_val = np.array(vgg_validation["feats"])
vgg_y_val = np.asarray(vgg_validation["labels"])
print(vgg_x_val.shape)
print(vgg_y_val.shape)
