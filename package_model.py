import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from src.shipment_prep import FEATURES, clean_shipments

# reading the final configuration from Part C
cfg = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
max_depth = None if str(cfg["max_depth"]) == "unlimited" else int(cfg["max_depth"])
threshold = float(cfg["threshold"])

# rebuilding Part C's stratified split
df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
)

# fitting the final pipeline on the training rows
model = Pipeline([
    ("selector", SelectKBest(f_classif, k=int(cfg["selector_k"]))),
    ("model", RandomForestClassifier(
        random_state=42,
        n_estimators=int(cfg["n_estimators"]),
        max_depth=max_depth,
        min_samples_leaf=int(cfg["min_samples_leaf"]),
    )),
])
model.fit(X_train, y_train)

# saving the model
joblib.dump(model, "models/delay_model.pkl")
pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

raw = pd.read_csv("data/raw_shipments.csv")
_, medians = clean_shipments(raw)
pd.DataFrame({
    "column": list(medians.keys()),
    "median": [round(v, 4) for v in medians.values()],
}).to_csv("models/impute_values.csv", index=False)

# reload checking
reloaded = joblib.load("models/delay_model.pkl")
p_mem = model.predict_proba(X_test)[:, 1]
p_re = reloaded.predict_proba(X_test)[:, 1]

check = pd.DataFrame({
    "metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
    "value": [len(X_test), float(np.abs(p_mem - p_re).max()), threshold],
})
check["value"] = check["value"].astype(float).round(4)
check.to_csv("outputs/part_d_reload_check.csv", index=False)
print(check.to_string(index=False))