import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from src.shipment_prep import FEATURES

df=pd.read_csv("data/modelling_table.csv")
X=df[FEATURES]
y=df["delivery_hours"]

def scores(y_val,y_pred):
    mse= mean_squared_error(y_val,y_pred)
    return{
        "MAE": mean_absolute_error(y_val,y_pred),
        "MSE": mse,
        "R2": r2_score(y_val,y_pred),
        "RMSE": float(np.sqrt(mse))
    }

# Splitting into training and validation data

X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=0.3,random_state=42)
X_test,X_val,y_test,y_val=train_test_split(X_temp,y_temp,test_size=0.5,random_state=42)

# Scaling data

scaler=StandardScaler().fit(X_train)
scaled_X_train=scaler.transform(X_train)
scaled_X_test=scaler.transform(X_test)

checks=pd.DataFrame({
    "metric":["train_rows","val_rows","test_rows",
              "train_scaled_max_abs_mean","test_scaled_max_abs_mean"],
    "value":[len(X_train),len(X_val),len(X_test),
             np.abs(scaled_X_train.mean(axis=0)).max(),
             np.abs(scaled_X_test.mean(axis=0)).max()],
})

checks["value"]=checks["value"].astype(float).round(4)
checks.to_csv("outputs/part_b_checks.csv", index=False)

# Baseline

baseline_pred=np.full(len(y_val),y_train.mean())
rows=[{"model": "mean_baseline", "split":"val", **scores(y_val,baseline_pred)}]

# Tree depth diagnostic

depths=[1,2,3,4,6,8,12,None]
depth_rows=[]
best_depth=None
best_rmse=np.inf

for d in depths:
    tree=DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train,y_train)
    tr=scores(y_train, tree.predict(X_train))["RMSE"]
    va=scores(y_val, tree.predict(X_val))["RMSE"]
    depth_rows.append({"max_depth":"unlimited" if d is None else d,
                       "train_mse": round(tr,4),
                       "val_mse": round(va,4)})

    if va<best_rmse:
        best_rmse, best_depth=va,d

pd.DataFrame(depth_rows).to_csv("outputs/part_b_tree_depth.csv", index=False)
print("best tree depth:", best_depth)

# Four Candidate Models

models = {
    "linear_regression": Pipeline([("scaler", StandardScaler()),
                                   ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                   random_state=42),
}
val_rmse = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    s = scores(y_val, m.predict(X_val))
    val_rmse[name] = s["RMSE"]
    rows.append({"model": name, "split": "val", **s})

# Learning rate experiment

lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    g = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    g.fit(X_train, y_train)
    s = scores(y_val, g.predict(X_val))
    lr_rows.append({"learning_rate": lr, "val_rmse": round(s["RMSE"], 4),
                    "val_r2": round(s["R2"], 4)})
pd.DataFrame(lr_rows).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)

# Cross-validated comparison and final model

ranked = sorted(val_rmse, key=val_rmse.get)      
top_two = ranked[:2]

X_trval = pd.concat([X_train, X_val])            
y_trval = pd.concat([y_train, y_val])
cv = KFold(n_splits=5, shuffle=True, random_state=42)   

cv_rows = []
for name in top_two:
    r2s = cross_val_score(models[name], X_trval, y_trval, cv=cv)
    cv_rows.append({"model": name, "mean_r2": round(r2s.mean(), 4),
                    "std_r2": round(r2s.std(), 4)})
pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv", index=False)

final_name = ranked[0]
final_scores = scores(y_test, models[final_name].predict(X_test))  
rows.append({"model": final_name, "split": "test", **final_scores})

metrics = pd.DataFrame(rows, columns=["model", "split", "MAE", "MSE", "RMSE", "R2"])
metrics[["MAE", "MSE", "RMSE", "R2"]] = metrics[["MAE", "MSE", "RMSE", "R2"]].round(4)
metrics.to_csv("outputs/part_b_metrics.csv", index=False)

print(checks)
print(metrics)
print("final model:", final_name)
