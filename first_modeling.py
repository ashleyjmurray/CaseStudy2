import pandas as pd
import numpy as np
np.random.seed(440)
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_predict

final = pd.read_csv("final.csv") # Most up to date final.csv you have on your machine

# The way I'm currently doing models
na_cols = []
for col in final.columns:
    numnas = sum(final[col].isna())
    print(col, numnas)
    if numnas > 0:
        na_cols.append(col)
na_cols.extend(["Unnamed: 0"])
final = final.drop(columns = na_cols)

# Get X and Y from loocv_data formatted for use in sklearn models
X = final.drop(columns = na_cols)
X["subject"] = X["subject"].astype("category")
X = pd.get_dummies(X, drop_first = True)
Y = np.array(X["label"])
X.drop(columns = ["label"], inplace = True)
X = inf_to_mean(X.to_numpy())

# Fit lr
lr = LogisticRegression(max_iter = 10000, penalty = "l1", solver = "liblinear")
cv = GridSearchCV(lr, param_grid = {
    "C":[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
},
                 scoring = "accuracy",
                 cv = 15
                 )

cv.fit(X, Y)
lr_cv_output = pd.DataFrame(cv.cv_results_).sort_values("rank_test_score")
best_lr = cv.best_estimator_

# Determine which coefficients remain in best logistic chosen
X_colnames = pd.get_dummies(loocv_data.drop(columns = na_cols).drop(columns = ["label"]), drop_first = True).columns
print("Coefficients in best Lasso Logistic:")
lr_coef_best_idx = np.flip(np.argsort(np.abs(best_lr.coef_).reshape(-1)))
lr_coef_best = best_lr.coef_.reshape(-1)[lr_coef_best_idx]
coef_touse = []
coef_toexclude = []
for idx, coef in zip(lr_coef_best_idx, lr_coef_best):
    if coef != 0:
        print(f"{X_colnames[idx]}: {coef}")
        coef_touse.append(X_colnames[idx])
    else:
        coef_toexclude.append(X_colnames[idx])

# Fitting an XGB
xgb = XGBClassifier()
xgb_params = {
    "eta":[0.01,0.1,0.2],
    #"min_child_weight":[1, 5, 10],
    "max_depth":list(np.arange(3,11, 2)),
    "gamma" : [0, 0.1, 0.5],
    "subsample":[0.5,1],
    "colsample_bytree":[0.5,1],
    "alpha":[0,1,10,100]
}
cv = GridSearchCV(xgb, param_grid = xgb_params,
                 scoring = "accuracy",
                 cv = 15
                 )
cv.fit(X,Y)
xgb_cv_output = pd.DataFrame(cv.cv_results_).sort_values("rank_test_score")
xgb_best = cv.best_estimator_

# Create ensemble

# Set up models to be used during ensembling using params found from CV before
ens_lr = LogisticRegression()
ens_lr.set_params(**best_lr.get_params())
ens_xgb = XGBClassifier()
ens_xgb.set_params(**xgb_best.get_params())

final_estimator = LogisticRegression(max_iter = 10000, penalty = "l1", solver = "liblinear")

stacking_estimator = StackingClassifier(estimators = [("logistic",ens_lr), ("xgboost",ens_xgb)], final_estimator = final_estimator, cv = 15)

stacking_estimator.fit(X,Y)
y_pred_ensemble = cross_val_predict(stacking_estimator, X, Y, cv = 15)
ensemble_accuracy = accuracy_score(Y.reshape(-1), y_pred_ensemble)

stacking_estimator_svm = StackingClassifier(estimators = [("logistic",ens_lr), ("xgboost",ens_xgb)], final_estimator = SVC(kernel = "rbf"), cv = 15)
stacking_estimator_svm.fit(X,Y)
y_pred_ensemble_svm = cross_val_predict(stacking_estimator_svm, X, Y, cv = 15)
svm_ensemble_accuracy = accuracy_score(Y.reshape(-1), y_pred_ensemble_svm)