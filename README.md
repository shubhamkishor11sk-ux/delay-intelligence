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

## Model results (regression)

All numbers come from the `outputs/part_b_*.csv` files.

- **Baseline.** Predicting the training mean gives R² = -0.0212. This means
  the baseline explains almost none of the variation in delivery hours,
  so it is the floor every model must beat.
- **Best model on validation.** Linear regression model won with a validation RMSE of
  4.8912 hours (R² = 0.9313), so its typical error is about 4.8912 hours.
- **Decision-tree depth.** Depths [1,2] are high-bias: train RMSE are 11.6334 and 9.3687 and
  validation RMSE are 12.1324 and 10.4114 are both high with a small gap. Depths [12,None] are
  high-variance: train RMSE falls to 1.3613 and 0.0000 while validation RMSE is 7.4173 and 7.7890,
  a gaps of 6.056 and 7.7890. Depth [6] had the lowest validation RMSE (6.2796).
- **Learning rate.** 0.10 was best (validation RMSE 4.9165,R² 0.9306). At 0.01 the
  model learns too slowly in 100 trees and underfits (RMSE 10.0018, R² 0.7127). At 1.0 each
  tree overshoots the correction and the fit becomes unstable (RMSE 7.3701, R² 0.8440).
- **Cross-validation.** 5-fold CV gave linear regression mean R² 0.9261 (std 0.0048)
  and gradient boosting mean R² 0.9352 (std 0.0044). This disagrees with the validation ranking, I
  would ship gradient boosting model because CV averages over all 1020 training-plus-validation rows 
  instead of one 180-row split, gradient boosting has higher mean R² and more stable scores across folds and validation RMSE is only about 0.02 hours behind a gap   small enough to overcome easily.
- **Final test score.** Gradient boosting scored RMSE 4.9165 hours on the 180 test 
  rows, scored once.


## Model results (classification)

All numbers come from the `outputs/part_c_*.csv` files.

- **Blind baseline.** Predicting "not delayed" for every parcel gives accuracy
  0.7667 but recall 0 (0 of 42 delayed parcels caught), so it is useless for
  catching delays even though its accuracy looks high.
- **Noise columns.** The five bookkeeping columns hold a combined 0.2245 of the
  forest's importance. Their F-score ranks are [8,12,14,16,17], so all five
  fall outside the top 7.
- **Feature count (k).** Validation F1 for k = 5, 7, 10, 12, 17 is 0.6988,0.8333,
  0.7857.0.8049,0.7848. The value increases till k=7 then decreases,so extra columns
  add noise. The brief fixes k = 7 which is used from here on.
- **Tuning.** Grid search reached validation F1 0.7750, randomized search 0.7848
  and the untuned forest on the same 7 columns 0.8333. Tuning did not beat the
  untuned forest. With only 180 validation rows (42 delayed), differences of
  this size are within noise, so tuning on one small validation set is unreliable.
  The chosen configuration is default_forest_same_7_features.
- **Alert threshold.** The threshold is 0.5, the highest with validation recall
  at least 0.80 (recall 0.8333, precision 0.7955). At 0.6 recall falls to 0.7619 
  while precision rises to 0.8205. Lower thresholds raise more unnecessary calls
  (9 at the chosen one) but miss fewer delayed parcels, which is the cheaper error.
- **Test result.** At that threshold the final model scored precision 0.6889,
  recall 0.7561 and F1 0.7209 on the 180 test rows, catching 31 of 41 delayed
  parcels. Validation recall was 0.8333. Test recall is lower and falls below the
  0.80 target, so the validation recall of 0.8333 was somewhat optimistic: the
  threshold was chosen on validation rows, so its recall there is biased upwards,
  while the test rows were never used for any choice. The test set has
  only 41 delayed parcels, so each one moves recall by about 0.024 and the gap
  (about 0.077, roughly three parcels) is partly sampling noise. I would treat
  the true recall as somewhere around 0.76 to 0.83 rather than claim 0.80 is
  guaranteed.