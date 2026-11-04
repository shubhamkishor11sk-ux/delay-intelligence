# Delay Intelligence

Regional operations analytics for a logistics network, using synthetic shipment data.
Everything here runs offline with pandas, NumPy, scikit-learn and joblib.

## How to run

Run from the repository root, in this order:

1. `pip install -r requirements.txt`
2. `python generate_data.py`
3. `python part_a_clean_eda.py`


## Data findings

All numbers below are copied from `outputs/part_a_delay_by_group.csv`,
`outputs/part_a_correlations.csv` and `outputs/part_a_checks.csv`.

1. **Express parcels are delayed most often.** Express has a delay rate of 0.4897
   (119 of 243 parcels), against 0.2056 for Standard and 0.0857 for Economy.

2. **Delay rate rises sharply with hub tier.** It goes from 0.0400 in Tier 1 to
   0.2150 in Tier 2 and 0.4400 in Tier 3.

3. **Delay rate rises with distance.** It is 0.0219 for 0-200 km, 0.1368 for
   201-400 km, 0.4364 for 401-600 km and 0.8523 for 601+ km.

4. **The three columns most correlated with delivery_hours** (ranked by absolute
   value, excluding delivery_hours itself) are `promised_hours` (0.8372),
   `distance_km` (0.4013) and `n_stops` (0.3247).

5. **Why declared_value_inr was log-transformed.** The filled declared value is
   right-skewed, with a mean of INR 1,533.05 well above its median of INR 1,121.50,
   while after the natural log the mean (7.0280) and median (7.0220) are nearly equal,
   so the transformed feature is roughly symmetric.