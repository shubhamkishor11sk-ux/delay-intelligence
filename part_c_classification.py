import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.feature_selection import SelectKBest,f_classif

from src.shipment_prep import FEATURES, NOISE_COLUMNS

df=pd.read_csv("data/modelling_table.csv")
X=df[FEATURES]
y=df["is_delayed"]

# splitting data
X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=0.5,random_state=42,stratify=y_temp)

print("rows :", len(X_train), len(X_val), len(X_test))
print("delays :", y_train.sum(), y_val.sum(), y_test.sum())

# Blind baseline and three classifiers

def cls_row(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
    }

model_rows=[]

# blind baseline: predicts "not delayed" for every parcel
model_rows.append(cls_row("blind_all_on_time", y_val, np.zeros(len(y_val), dtype=int)))

# making logistic regression inside a pipeline
logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
logreg.fit(X_train, y_train)
model_rows.append(cls_row("logistic_regression", y_val, logreg.predict(X_val)))

# decision tree
best_d, best_f1 = None, -1.0
for d in [2, 3, 4, 5, 6, 8, 10]:
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    t.fit(X_train, y_train)
    f1 = f1_score(y_val, t.predict(X_val))
    if f1 > best_f1:                      
        best_f1, best_d = f1, d
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42)
tree.fit(X_train, y_train)
model_rows.append(cls_row(f"decision_tree_depth_{best_d}", y_val, tree.predict(X_val)))

# random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
model_rows.append(cls_row("random_forest_default", y_val, rf.predict(X_val)))

models_df = pd.DataFrame(model_rows,
                         columns=["model", "TN", "FP", "FN", "TP",
                                  "accuracy", "precision", "recall", "f1"])
models_df.to_csv("outputs/part_c_models.csv", index=False)
print(models_df.to_string(index=False))

# Which column carry signal?

rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
rf_all.fit(X_train, y_train)

# one F-score per column:
selector = SelectKBest(f_classif, k="all").fit(X_train, y_train)

scores_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": selector.scores_,
    "rf_importance": rf_all.feature_importances_,
})
scores_df["f_rank"] = scores_df["f_score"].rank(ascending=False, method="first").astype(int)
scores_df["is_noise"] = scores_df["feature"].isin(NOISE_COLUMNS)
scores_df = scores_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
scores_df[["f_score", "rf_importance"]] = scores_df[["f_score", "rf_importance"]].round(4)
scores_df.to_csv("outputs/part_c_feature_scores.csv", index=False)
print(scores_df.sort_values("f_rank").to_string(index=False))

# validation F1 for different numbers of selected columns
selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([
        ("select", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    selectk_rows.append({"k": k, "val_f1": round(f1_score(y_val, pipe.predict(X_val)), 4)})
selectk_df = pd.DataFrame(selectk_rows)
selectk_df.to_csv("outputs/part_c_selectk.csv", index=False)
print(selectk_df.to_string(index=False))


# tuning on the 7 selected columns

sel7 = SelectKBest(f_classif, k=7).fit(X_train, y_train)
X_train_7 = sel7.transform(X_train)
X_val_7 = sel7.transform(X_val)
print("selected 7:", [f for f, keep in zip(FEATURES, sel7.get_support()) if keep])

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}

# grid search: 27 candidates x 5 folds
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train_7, y_train)

# randomized search: 9 candidates x 5 folds
random = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid,
                          n_iter=9, cv=5, random_state=42)
random.fit(X_train_7, y_train)

# untuned forest on the same 7 columns
default_rf = RandomForestClassifier(n_estimators=100, random_state=42)
default_rf.fit(X_train_7, y_train)

tuning_rows = [
    {
        "search": "grid_search",
        "candidates": len(grid.cv_results_["params"]),
        "cv_folds": 5,
        "total_fits": len(grid.cv_results_["params"]) * 5,
        "best_params": str(grid.best_params_),
        "best_cv_accuracy": round(grid.best_score_, 4),
        "val_f1": round(f1_score(y_val, grid.best_estimator_.predict(X_val_7)), 4),
    },
    {
        "search": "randomized_search",
        "candidates": len(random.cv_results_["params"]),
        "cv_folds": 5,
        "total_fits": len(random.cv_results_["params"]) * 5,
        "best_params": str(random.best_params_),
        "best_cv_accuracy": round(random.best_score_, 4),
        "val_f1": round(f1_score(y_val, random.best_estimator_.predict(X_val_7)), 4),
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": "",
        "val_f1": round(f1_score(y_val, default_rf.predict(X_val_7)), 4),
    },
]
tuning_df = pd.DataFrame(tuning_rows, columns=["search", "candidates", "cv_folds", "total_fits",
                                               "best_params", "best_cv_accuracy", "val_f1"])
tuning_df.to_csv("outputs/part_c_tuning.csv", index=False)
print(tuning_df.to_string(index=False))