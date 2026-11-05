import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

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