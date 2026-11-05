import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("data/modelling_table.csv")

# Task 1 --> hub profile
profile = df.groupby("hub_id").agg(
    shipments=("hub_id", "size"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean"),
).reset_index().sort_values("hub_id")

profile[["mean_delivery_hours", "delay_rate",
         "mean_distance_km", "mean_n_stops"]] = profile[
    ["mean_delivery_hours", "delay_rate",
     "mean_distance_km", "mean_n_stops"]].round(4)

profile.to_csv("outputs/hub_profile.csv", index=False)
print(profile.to_string(index=False))


# Task 2 --> K-Means segmentation
CLUSTER_COLS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]

# profile is already sorted by hub_id
profile = profile.sort_values("hub_id").reset_index(drop=True)

scaled = StandardScaler().fit_transform(profile[CLUSTER_COLS])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
profile["cluster"] = kmeans.fit_predict(scaled)

print(profile[["hub_id"] + CLUSTER_COLS + ["cluster"]].to_string(index=False))
print(profile["cluster"].value_counts().sort_index())