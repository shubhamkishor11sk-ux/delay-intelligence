import pandas as pd

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