import pandas as pd
from src.shipment_prep import clean_shipments,build_features

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

print(len(df))
print(df["is_delayed"].sum(), round(df["is_delayed"].mean(), 4))