import pandas as pd
from src.shipment_prep import clean_shipments,build_features,FEATURES

# reading raw files
raw_file=pd.read_csv("data/raw_shipments.csv")
hubs=pd.read_csv("data/hubs.csv")

# calling functions
clean,medians=clean_shipments(raw_file)
df=build_features(clean,hubs)

# delayed when delivery hours is greater than promised hours
df["is_delayed"]=(df["delivery_hours"]>df["promised_hours"]).astype(int)

# results to csv
df.to_csv("data/modelling_table.csv", index=False)

print((len(df)), df["is_delayed"].sum(), round(df["is_delayed"].mean(), 4))


# counts before any filling
dropped = raw_file.drop_duplicates()

# quartiles of delivery_hours
q1 = df["delivery_hours"].quantile(0.25)
q3 = df["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

checks = {
    "raw_rows": len(raw_file),
    "duplicate_rows_removed": len(raw_file) - len(dropped),
    "rows_after_dedup": len(dropped),
    "missing_distance_km_before": dropped["distance_km"].isna().sum(),
    "missing_parcel_weight_before": dropped["parcel_weight_kg"].isna().sum(),
    "sentinel_weight_rows": (dropped["parcel_weight_kg"] > 100).sum(),
    "missing_declared_value_before": dropped["declared_value"].isna().sum(),
    "missing_payment_mode_before": dropped["payment_mode"].isna().sum(),
    "service_type_spellings_before": dropped["service_type"].nunique(),
    "service_type_spellings_after": df["service_type"].nunique(),
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "rows_after_hub_merge": len(df),
    "nulls_in_features_and_target": df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum(),
    "delayed_count": df["is_delayed"].sum(),
    "delay_rate": df["is_delayed"].mean(),
    "hours_q1": q1,
    "hours_q3": q3,
    "hours_lower_bound": lower,
    "hours_upper_bound": upper,
    "hours_outlier_count": ((df["delivery_hours"] < lower) | (df["delivery_hours"] > upper)).sum(),
    "declared_value_mean_inr": df["declared_value_inr"].mean(),
    "declared_value_median_inr": df["declared_value_inr"].median(),
    "log_declared_value_mean": df["log_declared_value"].mean(),
    "log_declared_value_median": df["log_declared_value"].median(),
}

checks_df = pd.DataFrame({
    "metric": list(checks.keys()),
    "value": [round(float(v), 4) for v in checks.values()],
})
checks_df.to_csv("outputs/part_a_checks.csv", index=False)
print(checks_df)

# # Cutting distance into bins
df["distance_band"] = pd.cut(
    df["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
)

tables = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = df.groupby(dim, observed=True).agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean"),
    ).reset_index()
    g = g.rename(columns={dim: "group"})
    g.insert(0, "dimension", dim)
    g["group"] = g["group"].astype(str)
    tables.append(g)

delay_table = pd.concat(tables, ignore_index=True)
delay_table["delay_rate"] = delay_table["delay_rate"].round(4)
delay_table["mean_hours"] = delay_table["mean_hours"].round(4)
delay_table.to_csv("outputs/part_a_delay_by_group.csv", index=False)
print(delay_table)