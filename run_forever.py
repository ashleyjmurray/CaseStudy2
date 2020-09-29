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

final = pd.read_csv("final.csv")

# Removing columns with NA values
na_cols = []
for col in final.columns:
    numnas = sum(final[col].isna())
    print(col, numnas)
    if numnas > 5:
        na_cols.append(col)
na_cols.extend(["Unnamed: 0"])
final = final.drop(columns = na_cols)
print("Dropped", len(na_cols), "columns")

# Create dummies for subject
final["subject"] = final["subject"].astype("category")
final = pd.get_dummies(final, drop_first = True)

# Convert label to 1 for stress, 0 for amusement
final["label"] = (final["label"] == 2.0).astype(int)
final["label"].value_counts()

def inf_to_mean(X):
    """
    Takes numpy array X and returns a version replacing inf and na values with their column means
    """
    X = np.nan_to_num(X, nan = np.nan, posinf = np.nan)
    col_mean = np.nanmean(X, axis = 0)
    inds = np.where(np.isnan(X)) 
    X[inds] = np.take(col_mean, inds[1]) 
    return X

# Format data into numpy arrays to be used in sklearn models
Y = np.array(final["label"])
X = final.drop(columns = ["label"])
X = inf_to_mean(X.to_numpy())

X_colnames = final.drop(columns = ["label"]).columns

best_models = {}

# Initialize penalized logistic regression
lr = LogisticRegression(max_iter = 10000, penalty = "l1", solver = "liblinear")

# Initialize Cross Validation over L1 penalty coefficient
cv = GridSearchCV(lr, param_grid = {
    "C":[0.01, 0.1, 1, 100, 1000]},
                 scoring = "accuracy",
                 cv = 15
                 )

# Perform CV
cv.fit(X,Y) # This is the speed bottleneck

# Show CV Results
lr_cv_results = pd.DataFrame(cv.cv_results_).sort_values("rank_test_score")
lr_cv_results

best_lr = cv.best_estimator_
best_models["logistic"] = best_lr

n_chosen_coefs = len([x for x in best_lr.coef_.reshape(-1) if x != 0])

# Variables chosen by the Lasso model and their coefficients
print("Coefficients:")
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

        

