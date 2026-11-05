import joblib
import pandas as pd

from src.shipment_prep import clean_shipments, build_features


new = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
segments = pd.read_csv("outputs/hub_segments.csv")

model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
imp = pd.read_csv("models/impute_values.csv")
medians = {row["column"]: float(row["median"]) for _, row in imp.iterrows()}

# cleaning and building features
clean, _ = clean_shipments(new, medians)
feats = build_features(clean, hubs)

# probabilities
X = feats[feature_order]
feats["delay_probability"] = model.predict_proba(X)[:, 1].round(4)
feats["flagged"] = (feats["delay_probability"] >= threshold).astype(int)

# joining the hub segment (left merge)
out = feats.merge(segments[["hub_id", "segment"]], on="hub_id", how="left")

# action rule
def choose_action(row):
    if row["flagged"] == 1 and row["segment"] == "Escalate":
        return "Escalate to hub manager"
    if row["flagged"] == 1:
        return "Proactive customer call"
    return "No action"

out["action"] = out.apply(choose_action, axis=1)

result = out[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
result.to_csv("outputs/new_shipment_actions.csv", index=False)
print(result.to_string(index=False))
print(result["action"].value_counts())