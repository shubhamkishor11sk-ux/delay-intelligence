import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations

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

# Task 3 --> separation evidence 
def across_over_within(X, labels):
    across, within = [], []
    for i, j in combinations(range(len(X)), 2):      
        d = np.linalg.norm(X[i] - X[j])              
        (within if labels[i] == labels[j] else across).append(d)
    return np.mean(across) / np.mean(within)

sep_rows = []
for K in [2, 3, 4, 5]:
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)
    sizes = sorted(np.bincount(labels), reverse=True)
    sep_rows.append({
        "K": K,
        "across_over_within": round(across_over_within(scaled, labels), 4),
        "cluster_sizes": "-".join(str(int(s)) for s in sizes),
    })

sep_df = pd.DataFrame(sep_rows, columns=["K", "across_over_within", "cluster_sizes"])
sep_df.to_csv("outputs/part_d_separation.csv", index=False)
print(sep_df.to_string(index=False))