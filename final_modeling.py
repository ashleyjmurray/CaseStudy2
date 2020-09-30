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
from sklearn.pipeline import Pipeline

final = pd.read_csv("final_w_bvp.csv")

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

# Determine all subsets of the data on which we want to model
var_subsets = [
	#("all",[x for x in final.columns if x != "subject" and x != "label"]),
	# ("ecg_only",[x for x in final.columns if x.startswith("HRV") or x.startswith("ECG")]),
	# ("eda_chest", [x for x in final.columns if x.startswith("eda") and x.endswith("chest")]),
	# ("eda_wrist", [x for x in final.columns if x.startswith("eda") and x.endswith("wr")]),
	# ("resp_only", ['resp_rate', 'mean_inhale_duration', 'std_inhale_duration', 'mean_exhale_duration', 'std_exhale_duration', 'ie_ratio', 'resp_stretch']),
	# ("temp_only", [x for x in final.columns if x.startswith("temp")]),
	("acc_only", [x for x in final.columns if x.startswith("acc")]),
	("bvp_only", [x for x in final.columns if x.startswith("bvp")]),
	("wrist_only", [x for x in final.columns if (x.startswith('eda') and x.endswith('wr')) or x.startswith('acc') or x.startswith('temp') or x.startswith('bvp')]),
	("chest_only", [x for x in final.columns if (x.startswith('eda') and x.endswith('chest')) or (x.startswith("HRV") or x.startswith("ECG"))] + ['resp_rate', 'mean_inhale_duration', 'std_inhale_duration', 'mean_exhale_duration', 'std_exhale_duration', 'ie_ratio', 'resp_stretch'])
]

for var_subset_name, var_subset_columns in var_subsets: 
	print(f"Repeating for {var_subset_name}...")
	final_subset = final[var_subset_columns + ["label"]]
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
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
	Y = np.array(final_subset["label"])
	X = final_subset.drop(columns = ["label"])
	X = inf_to_mean(X.to_numpy())
	X = scaler.fit_transform(X)

	X_colnames = final_subset.drop(columns = ["label"]).columns

	best_models = {}
	model_accuracies = {}
	model_f1 = {}

	# Initialize penalized logistic regression
	lr = LogisticRegression(max_iter = 10000, penalty = "l1", solver = "liblinear")

	# Initialize Cross Validation over L1 penalty coefficient
	cv = GridSearchCV(lr, param_grid = {
	    "C":[0.01, 0.1, 1, 100, 1000]},
	                 scoring = "accuracy",
	                 cv = 15,
	                 n_jobs = -1,
	                 verbose = 10
	                 )

	# Perform CV
	cv.fit(X,Y)

	best_lr = cv.best_estimator_
	best_models["logistic"] = best_lr

	# Variables chosen by the Lasso model and their coefficients
	coef_names_ordered = []
	coef_values_ordered = []
	print("Coefficients:")
	lr_coef_best_idx = np.flip(np.argsort(np.abs(best_lr.coef_).reshape(-1)))
	lr_coef_best = best_lr.coef_.reshape(-1)[lr_coef_best_idx]
	coef_touse = []
	coef_toexclude = []
	for idx, coef in zip(lr_coef_best_idx, lr_coef_best):
	    if coef != 0:
	        print(f"{X_colnames[idx]}: {coef}")
	        coef_touse.append(X_colnames[idx])
	        coef_names_ordered.append(X_colnames[idx])
	        coef_values_ordered.append(coef)
	    else:
	        coef_toexclude.append(X_colnames[idx])

	# Export coefficient table
	pd.DataFrame({"Variable Name":coef_names_ordered, "Coefficient":coef_values_ordered}).to_csv(f"lasso_logistic_coefs_{var_subset_name}.csv", index = False)

	lr_cv_preds = cross_val_predict(best_lr, X,Y, cv = 15)
	model_accuracies["logistic"] = accuracy_score(lr_cv_preds, Y.reshape(-1))
	model_f1["logistic"] = f1_score(lr_cv_preds, Y.reshape(-1))

	# Below is code to get the XGB Model from scratch using CV

	# Initialize XGBoost Model
	xgb = XGBClassifier()

	# Set parameter grid for cross-validation
	xgb_params = {
	    "eta":[0.01,0.1,0.2],
	    #"min_child_weight":[1, 5, 10],
	    "max_depth":list(np.arange(3,11, 2)),
	    "gamma" : [0, 0.1, 0.5],
	    "subsample":[0.5,1],
	    "colsample_bytree":[0.5,1],
	    "alpha":[0,1,10,100]
	}

	# Initialize cross-validation over selected parameters
	cv = GridSearchCV(xgb, param_grid = xgb_params,
	                 scoring = "accuracy",
	                 cv = 15, verbose = 10,
	                  n_jobs = -1
	                 )

	cv.fit(X,Y)

	xgb_cv_results = pd.DataFrame(cv.cv_results_).sort_values("rank_test_score")
	xgb_cv_results

	best_xgb = cv.best_estimator_
	best_models["xgb"] = best_xgb

	fi = best_xgb.feature_importances_
	fi_best_idx = np.flip(fi.argsort())
	fi_best = np.flip(np.sort(fi))
	fi_vars_ordered = []
	fi_values_ordered = []
	print("MOST IMPORTANT FEATURES:\n")
	for i in range(len(fi_best_idx)):
	    print("Feature name: {:>12}     Feature importance: {:>12}".format(X_colnames[fi_best_idx[i]], fi_best[i]))
	    fi_vars_ordered.append(X_colnames[fi_best_idx[i]])
	    fi_values_ordered.append(fi_best[i])

	pd.DataFrame({"Variable Name":fi_vars_ordered, "Feature Importance":fi_values_ordered}).to_csv(f"feature_importance_chart_{var_subset_name}.csv", index = False)

	xgb_cv_preds = cross_val_predict(best_xgb, X,Y, cv = 15, n_jobs = -1)
	model_accuracies["xgb"] = accuracy_score(xgb_cv_preds, Y.reshape(-1))
	model_f1["xgb"] = f1_score(xgb_cv_preds, Y.reshape(-1))


	cv_linear_svm = GridSearchCV(SVC(kernel = "linear"), param_grid = {
	    "C":[1, 10, 100]
	}, verbose = 10, n_jobs = -1, cv = 15)
	cv_linear_svm.fit(X,Y)
	best_linear_svm = cv_linear_svm.best_estimator_
	best_models["linear_svm"] = best_linear_svm
	linear_svm_cv_preds = cross_val_predict(best_linear_svm, X,Y, cv = 15)
	model_accuracies["linear_svm"] = accuracy_score(linear_svm_cv_preds, Y.reshape(-1))
	model_f1["linear_svm"] = f1_score(linear_svm_cv_preds, Y.reshape(-1))

	cv_rbf_svm = GridSearchCV(SVC(kernel = "rbf"), param_grid = {
	    "C":[1, 10, 100]
	}, verbose = 10, n_jobs = -1, cv = 15)
	cv_rbf_svm.fit(X,Y)
	best_rbf_svm = cv_rbf_svm.best_estimator_
	best_models["rbf_svm"] = best_rbf_svm
	rbf_svm_cv_preds = cross_val_predict(best_rbf_svm, X,Y, cv = 15)
	model_accuracies["rbf_svm"] = accuracy_score(rbf_svm_cv_preds, Y.reshape(-1))
	model_f1["rbf_svm"] = f1_score(rbf_svm_cv_preds, Y.reshape(-1))

	cv_poly_svm = GridSearchCV(SVC(kernel = "poly"), param_grid = {
	    "C":[1, 10, 100],
	    "degree":[2,3,5]},
	    verbose = 10,
	    n_jobs = -1
	)
	cv_poly_svm.fit(X,Y)
	best_poly_svm = cv_poly_svm.best_estimator_
	best_models["poly_svm"] = best_poly_svm
	poly_svm_cv_preds = cross_val_predict(best_poly_svm, X,Y, cv = 15)
	model_accuracies["poly_svm"] = accuracy_score(poly_svm_cv_preds, Y.reshape(-1))
	model_f1["poly_svm"] = f1_score(poly_svm_cv_preds, Y.reshape(-1))

	class PLSRegressionWrapper(PLSRegression):

	    def transform(self, X):
	        return super().transform(X)

	    def fit_transform(self, X, Y):
	        return self.fit(X,Y).transform(X)

	pls_rbf = Pipeline([("pls", PLSRegressionWrapper()), ("rbf_svm", SVC(kernel = "rbf"))])
	pls_rbf_cv_preds = cross_val_predict(pls_rbf, X, Y, cv = 15, n_jobs = -1)
	model_accuracies["pls_rbf"] = accuracy_score(pls_rbf_cv_preds, Y.reshape(-1))
	model_f1["pls_rbf"] = f1_score(pls_rbf_cv_preds, Y.reshape(-1))

	pls_linear = Pipeline([("pls", PLSRegressionWrapper()), ("linear_svm", SVC(kernel = "linear"))])
	pls_linear_cv_preds = cross_val_predict(pls_linear, X, Y, cv = 15, n_jobs = -1)
	model_accuracies["pls_linear"] = accuracy_score(pls_linear_cv_preds, Y.reshape(-1))
	model_f1["pls_linear"] = f1_score(pls_linear_cv_preds, Y.reshape(-1))

	pls_polynomial = Pipeline([("pls", PLSRegressionWrapper()), ("poly_svm", SVC(kernel = "poly"))])
	pls_polynomial_cv_preds = cross_val_predict(pls_polynomial, X, Y, cv = 15, n_jobs = -1)
	model_accuracies["pls_poly"] = accuracy_score(pls_polynomial_cv_preds, Y.reshape(-1))
	model_f1["pls_poly"] = f1_score(pls_polynomial_cv_preds, Y.reshape(-1))

	pls_lr = Pipeline([("pls", PLSRegressionWrapper()), ("logistic", LogisticRegression())])
	pls_lr_cv_preds = cross_val_predict(pls_lr, X, Y, cv = 15, n_jobs = -1)
	model_accuracies["pls_lr"] = accuracy_score(pls_lr_cv_preds, Y.reshape(-1))
	model_f1["pls_lr"] = f1_score(pls_lr_cv_preds, Y.reshape(-1))

	model_list = [("logistic", best_lr), ("xgboost", best_xgb)]
	final_estimator = LogisticRegression(max_iter = 10000, penalty = "l1", solver = "liblinear")
	stacking_estimator = StackingClassifier(estimators = model_list, final_estimator = final_estimator, cv = 15)

	y_pred_ensemble = cross_val_predict(stacking_estimator, X, Y, cv = 5, n_jobs = -1)
	model_accuracies["logistic_xgb_stack"] = accuracy_score(y_pred_ensemble, Y.reshape(-1))
	model_f1["logistic_xgb_stack"] = f1_score(y_pred_ensemble, Y.reshape(-1))

	#model_list = [("logistic", best_lr), ("xgboost", best_xgb), ("pls_rbf", pls_rbf), ("pls_linear", pls_linear), ("pls_poly", pls_polynomial), ("pls_logistic", pls_lr)]
	model_list = [("logistic", best_lr), ("xgboost", best_xgb), ("linear_svm", best_linear_svm), ("rbf", best_rbf_svm), ("polynomial", best_poly_svm),
	             ("pls_rbf", pls_rbf), ("pls_linear", pls_linear), ("pls_polynomial", pls_polynomial), ("pls_lr", pls_lr)]
	final_estimator = LogisticRegression(max_iter = 10000, penalty = "l1", solver = "liblinear")
	stacking_estimator = StackingClassifier(estimators = model_list, final_estimator = final_estimator, cv = 15)
	y_pred_full_ensemble = cross_val_predict(stacking_estimator, X, Y, cv = 5, n_jobs = -1)
	model_accuracies["full_stack"] = accuracy_score(y_pred_full_ensemble, Y.reshape(-1))
	model_f1["full_stack"] = f1_score(y_pred_full_ensemble, Y.reshape(-1))

	# Accuracy df
	pd.DataFrame({"Accuracy":model_accuracies, "F1 Score":model_f1}).to_csv(f"model_results_{var_subset_name}.csv", index = False)

	model_params = {}
	for name,model in model_list:
	    model_params[name] = [str(model.get_params())]
	pd.DataFrame(model_params).to_csv(f"final_model_parameters_{var_subset_name}.csv", index = False)