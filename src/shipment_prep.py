import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no",
                 "sticker_serial_mod", "shelf_slot_no"]
FILL_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]


def clean_shipments(raw, medians=None):
    # 1. drop exact duplicate rows
    df = raw.drop_duplicates().copy()

    # 2. service_type: strip spaces, title case
    df["service_type"] = df["service_type"].str.strip().str.title()

    # 3. payment_mode: missing -> "Unknown"
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. declared_value text -> number in declared_value_inr, drop text column
    df["declared_value_inr"] = pd.to_numeric(
        df["declared_value"]
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df = df.drop(columns=["declared_value"])

    # 5. dispatch_time -> datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # 6. weight above 100 is a sentinel -> missing
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # 7. fill gaps with medians
    if medians is None:
        medians = {c: float(df[c].median()) for c in FILL_COLUMNS}
    for c in FILL_COLUMNS:
        df[c] = df[c].fillna(medians[c])

    return df, medians

def build_features(clean, hubs):

    hub_cols=["hub_id","hub_city","region","hub_tier","hub_daily_capacity"]

    # Now we perform left merge:
    df=clean.merge(hubs[hub_cols],on="hub_id", how="left")

    # finding promised hours from service type(int)
    df["promised_hours"]=df["service_type"].map(PROMISED_HOURS).astype(int)

    # 1 when payment mode is COD
    df["is_cod"]=(df["payment_mode"] == "COD").astype(int)

    # checking weekends
    df["is_weekend"]= (df["dispatch_time"].dt.dayofweek >=5).astype(int)

    # hour of dispatch should be integer
    df["dispatch_hour"]= df["dispatch_time"].dt.hour.astype(int)

    # natural log of rupee
    df["log_declared_value"]= np.log(df["declared_value_inr"])

    return df
