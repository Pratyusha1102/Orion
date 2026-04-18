"""
╔══════════════════════════════════════════════════════════════════╗
║      MODEL 5: OPD WAIT TIME PREDICTION                          ║
║      Hospital Resource Management System                         ║
║      Production-Level Pipeline — Fully Executable                ║
╚══════════════════════════════════════════════════════════════════╝

Problem:
    Predict OPD waiting time (in minutes) based on hospital
    infrastructure, patient load, doctor availability and location.

Dataset:
    Uses hospital_resource_dataset_10000_updated.csv
    Wait time and patient count are SIMULATED (not in raw data).
"""

# ─────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import pickle
import os

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')
np.random.seed(42)

print("=" * 65)
print("   MODEL 5: OPD WAIT TIME PREDICTION")
print("=" * 65)


# ═══════════════════════════════════════════════════════════════
# STEP 1 — LOAD & RESHAPE RAW DATA
# ═══════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading and reshaping raw hospital dataset...")

RAW_PATH = "/mnt/user-data/uploads/hospital_resource_dataset_10000_updated.csv"

try:
    raw = pd.read_csv(RAW_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset not found at: {RAW_PATH}")

print(f"  Raw shape: {raw.shape}")

# ── The raw dataset is in long format (one row per allocation).
#    We aggregate per hospital to get one row per hospital with
#    infrastructure-level columns matching the problem statement.
# ──────────────────────────────────────────────────────────────

# Pivot mean beds by bed_type per hospital
bed_pivot = raw[raw['bed_type'].notna()].pivot_table(
    index='hospital_id',
    columns='bed_type',
    values='beds_allocated',
    aggfunc='mean'
).reset_index()
bed_pivot.columns = ['hospital_id'] + [f'beds_{c}' for c in bed_pivot.columns[1:]]

# Average doctor count per hospital (staff_allocated averaged across records)
doctor_agg = raw[raw['staff_type'] == 'Doctor'].groupby('hospital_id').agg(
    Total_Doctors=('staff_allocated', 'mean')
).reset_index()

# Average ventilator count per hospital
vent_agg = raw[raw['equipment_type'] == 'Ventilator'].groupby('hospital_id').agg(
    Ventilators=('equipment_allocated', 'mean')
).reset_index()

# Mean beds (use ICU mean as proxy for Total_Beds — scale to realistic hospital size)
# ICU beds typically 10-20% of total; we'll infer Total_Beds from General + ICU mean
total_beds = raw[raw['bed_type'].notna()].groupby('hospital_id').agg(
    Total_Beds=('beds_allocated', 'mean')   # mean per allocation ≈ beds in each category
).reset_index()
# Scale up: multiply by 3 (3 bed types: General, ICU, Emergency) to approximate total
total_beds['Total_Beds'] = (total_beds['Total_Beds'] * 3).round()

# Location info (take first occurrence per hospital)
location = raw[['hospital_id', 'hospital_city', 'hospital_state', 'hospital_country']].drop_duplicates('hospital_id')

# Merge all pieces together
df = (location
      .merge(total_beds,  on='hospital_id', how='left')
      .merge(bed_pivot,   on='hospital_id', how='left')
      .merge(doctor_agg,  on='hospital_id', how='left')
      .merge(vent_agg,    on='hospital_id', how='left'))

# Rename to match problem-statement column names
df.rename(columns={
    'hospital_city':    'City',
    'hospital_state':   'State',
    'hospital_country':'Country',
}, inplace=True)

# Map bed type columns to problem-statement names
# ICU_Beds: from pivot, fallback to 15% of Total_Beds
if 'beds_ICU' in df.columns:
    df['ICU_Beds'] = df['beds_ICU'].fillna(df['Total_Beds'] * 0.15)
else:
    df['ICU_Beds'] = df['Total_Beds'] * 0.15

# Non_ICU_Beds: General ward beds, fallback to 70% of Total_Beds
if 'beds_General' in df.columns:
    df['Non_ICU_Beds'] = df['beds_General'].fillna(df['Total_Beds'] * 0.70)
else:
    df['Non_ICU_Beds'] = df['Total_Beds'] * 0.70

# Clip to realistic hospital sizes: 20 – 500 beds
df['Total_Beds']  = df['Total_Beds'].clip(20, 500)
df['ICU_Beds']    = df['ICU_Beds'].clip(2, 80)
df['Non_ICU_Beds']= df['Non_ICU_Beds'].clip(10, 400)

# Synthetic columns that aren't in the raw data
np.random.seed(42)
df['District']        = df['City']   # city ≈ district in this dataset
df['City_Type']       = np.random.choice(['Urban', 'Semi-Urban', 'Rural'], size=len(df), p=[0.6, 0.3, 0.1])
df['Total_Rooms']     = (df['Total_Beds'] * np.random.uniform(0.6, 0.9, len(df))).astype(int)
df['Emergency_Rooms'] = (df['Total_Rooms'] * np.random.uniform(0.08, 0.18, len(df))).astype(int).clip(lower=1)
df['Total_Doctors']   = df['Total_Doctors'].fillna(df['Total_Beds'] / 20).clip(lower=1)
df['Ventilators']     = df['Ventilators'].fillna(df['ICU_Beds'] * 0.5).clip(lower=0)

# Drop irrelevant columns
df.drop(columns=['hospital_id', 'Country'], errors='ignore', inplace=True)
df.drop(columns=[c for c in df.columns if c.startswith('beds_')], errors='ignore', inplace=True)

# Reset index cleanly
df.reset_index(drop=True, inplace=True)

# ── Expand the dataset to a realistic size via resampling ──────
# 5 hospitals × varied patient loads gives too few rows.
# We replicate with mild jitter so the model sees variety.
N_TARGET = 5000
reps     = int(np.ceil(N_TARGET / len(df)))
df       = pd.concat([df] * reps, ignore_index=True)
df       = df.head(N_TARGET).copy()
# Add tiny absolute jitter (±2%) to numeric cols to break exact duplicates
num_cols = df.select_dtypes(include=np.number).columns.tolist()
for col in num_cols:
    scale = df[col].std() * 0.02 + 0.001   # 2% of std, tiny
    df[col] = df[col] + np.random.normal(0, scale, len(df))
    df[col] = df[col].clip(lower=0)

print(f"  Working dataframe shape after reshape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")


# ═══════════════════════════════════════════════════════════════
# STEP 2 — DATA SIMULATION
#   Simulate patient count and OPD wait time realistically.
#   Wait time is derived from a NON-LINEAR formula with noise
#   so the model cannot simply memorise patients / doctors.
# ═══════════════════════════════════════════════════════════════
print("\n[STEP 2] Simulating patient count and OPD wait time...")

np.random.seed(42)

# ── 2a. Patient count ──────────────────────────────────────────
# Patients arrive as 30%–95% of total bed capacity.
# Use a right-skewed beta distribution so most hospitals
# are moderately busy, not always at the extremes.
bed_capacity    = df['Total_Beds'].clip(lower=1).values
beta_samples    = np.random.beta(a=2.5, b=1.5, size=len(df))   # skewed toward higher load
occupancy_frac  = 0.30 + beta_samples * (0.95 - 0.30)          # map [0,1] → [0.30, 0.95]
df['Patients']  = np.clip((occupancy_frac * bed_capacity).astype(int), 1, None)

# ── 2b. Non-linear wait time formula ───────────────────────────
#
#   Core idea:
#     base_wait = a * (patients / doctors)^1.3        ← non-linear load term
#     + b * sigmoid(utilization - 0.75) * 30          ← surge spike beyond 75%
#     + c * (1 / (1 + Emergency_Rooms))               ← fewer emergency rooms = worse
#     + d * (city_type_factor)                         ← location adjustment
#     + Gaussian noise                                 ← stochastic real-world variation
#     + log-normal noise                               ← heavy-tail outliers
#
doctors     = df['Total_Doctors'].clip(lower=1).values
patients    = df['Patients'].values
beds        = df['Total_Beds'].clip(lower=1).values
icu         = df['ICU_Beds'].clip(lower=0).values
emr         = df['Emergency_Rooms'].clip(lower=1).values
rooms       = df['Total_Rooms'].clip(lower=1).values

utilization = patients / beds                          # hospital load [0, 1]

def sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

# City type factor: urban hospitals slightly faster due to resources
city_type_map = {'Urban': 0.85, 'Semi-Urban': 1.0, 'Rural': 1.20}
city_factor   = df['City_Type'].map(city_type_map).fillna(1.0).values

# Non-linear base wait time
pd_ratio    = patients / doctors                        # patient-to-doctor ratio
base_wait   = (
    8.0 * (pd_ratio ** 1.3)                            # dominant non-linear driver
    + 30.0 * sigmoid(10 * (utilization - 0.75))        # surge spike above 75% load
    - 5.0 * np.log1p(emr)                              # more emergency rooms = relief
    + 4.0 * city_factor                                # location effect
    - 2.0 * np.log1p(icu / beds.clip(1) * 100)        # ICU ratio (better infra)
)

# Additive Gaussian noise (±5 min std) — simulate shift-to-shift variation
gaussian_noise   = np.random.normal(loc=0, scale=5.0, size=len(df))

# Multiplicative log-normal noise — simulate rare spikes (emergencies, staff absences)
lognormal_noise  = np.random.lognormal(mean=0, sigma=0.12, size=len(df))

# Combine
wait_time_raw = (base_wait + gaussian_noise) * lognormal_noise

# Clip to realistic OPD range: 5–120 minutes, then round to integer
df['Wait_Time'] = np.clip(wait_time_raw, 5, 120).round(1)

print(f"  Patients — min: {df['Patients'].min()}, "
      f"max: {df['Patients'].max()}, "
      f"mean: {df['Patients'].mean():.1f}")
print(f"  Wait_Time — min: {df['Wait_Time'].min()}, "
      f"max: {df['Wait_Time'].max()}, "
      f"mean: {df['Wait_Time'].mean():.1f} mins")


# ═══════════════════════════════════════════════════════════════
# STEP 3 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
print("\n[STEP 3] Engineering predictive features...")

EPS = 1e-6  # epsilon to prevent division by zero everywhere

def safe_divide(a, b, eps=EPS):
    """Divides a by b, clipping denominator to avoid zero-division."""
    return a / np.maximum(b, eps)

# ── Ratio features ─────────────────────────────────────────────
df['utilization']          = safe_divide(df['Patients'], df['Total_Beds'])
df['patient_doctor_ratio'] = safe_divide(df['Patients'], df['Total_Doctors'])
df['icu_ratio']            = safe_divide(df['ICU_Beds'],         df['Total_Beds'])
df['room_density']         = safe_divide(df['Total_Rooms'],      df['Total_Beds'])
df['emergency_ratio']      = safe_divide(df['Emergency_Rooms'],  df['Total_Rooms'])
df['ventilator_ratio']     = safe_divide(df['Ventilators'],      df['Total_Beds'])
df['bed_availability']     = safe_divide(df['Non_ICU_Beds'],     df['Total_Beds'])

# ── Non-linear transforms ──────────────────────────────────────
# Log transforms compress skewed distributions and expose non-linearity
df['log_pd_ratio']     = np.log1p(df['patient_doctor_ratio'])
df['log_patients']     = np.log1p(df['Patients'])
df['log_beds']         = np.log1p(df['Total_Beds'])
df['log_doctors']      = np.log1p(df['Total_Doctors'])

# Interaction: load × ratio — captures compound pressure
df['load_x_pd']        = df['utilization'] * df['patient_doctor_ratio']

# Surge flag: 1 when hospital is over 80% capacity
df['high_load_flag']   = (df['utilization'] > 0.80).astype(int)

# Clip derived ratios to [0, 1] where sensible
for col in ['utilization', 'icu_ratio', 'room_density', 'emergency_ratio',
            'ventilator_ratio', 'bed_availability']:
    df[col] = df[col].clip(0, 1)

print(f"  Features engineered. Dataframe now has {df.shape[1]} columns.")


# ═══════════════════════════════════════════════════════════════
# STEP 4 — HANDLE MISSING VALUES
# ═══════════════════════════════════════════════════════════════
print("\n[STEP 4] Handling missing values...")

before = df.isnull().sum().sum()

# Fill numeric NaNs with column median (robust to outliers)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical NaNs with 'Unknown'
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna('Unknown')

after = df.isnull().sum().sum()
print(f"  Missing values: {before} → {after} (all resolved)")


# ═══════════════════════════════════════════════════════════════
# STEP 5 — CATEGORICAL ENCODING
# ═══════════════════════════════════════════════════════════════
print("\n[STEP 5] Encoding categorical features with LabelEncoder...")

CATEGORICAL_COLS = ['City', 'District', 'State', 'City_Type']
# Only encode columns that actually exist in the dataframe
CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c in df.columns]

encoders = {}  # store encoders for reuse in prediction function

for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    # Add a sentinel 'Unknown' class so unseen labels don't crash at inference
    all_values = list(df[col].astype(str).unique()) + ['Unknown']
    le.fit(all_values)
    df[col + '_enc'] = le.transform(df[col].astype(str))
    encoders[col] = le
    print(f"  {col}: {le.classes_[:5].tolist()}{'...' if len(le.classes_) > 5 else ''}")

print(f"  Encoders saved for {list(encoders.keys())}")


# ═══════════════════════════════════════════════════════════════
# STEP 6 — DEFINE FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════
print("\n[STEP 6] Defining feature matrix X and target y...")

# Raw infrastructure features
RAW_FEATURES = [
    'Total_Beds', 'ICU_Beds', 'Non_ICU_Beds', 'Total_Doctors',
    'Ventilators', 'Total_Rooms', 'Emergency_Rooms', 'Patients',
]

# Engineered features
ENG_FEATURES = [
    'utilization', 'patient_doctor_ratio', 'icu_ratio',
    'room_density', 'emergency_ratio', 'ventilator_ratio',
    'bed_availability', 'log_pd_ratio', 'log_patients',
    'log_beds', 'log_doctors', 'load_x_pd', 'high_load_flag',
]

# Encoded categorical features
ENC_FEATURES = [c + '_enc' for c in CATEGORICAL_COLS]

ALL_FEATURES = RAW_FEATURES + ENG_FEATURES + ENC_FEATURES
TARGET       = 'Wait_Time'

# Verify all features exist
missing_feats = [f for f in ALL_FEATURES if f not in df.columns]
if missing_feats:
    print(f"  WARNING — Missing features (will be dropped): {missing_feats}")
    ALL_FEATURES = [f for f in ALL_FEATURES if f in df.columns]

X = df[ALL_FEATURES].copy()
y = df[TARGET].copy()

# Final NaN guard
X.fillna(X.median(numeric_only=True), inplace=True)
y.fillna(y.median(), inplace=True)

print(f"  Feature matrix X: {X.shape}")
print(f"  Target y — mean: {y.mean():.2f} mins, std: {y.std():.2f} mins")
print(f"  Features used ({len(ALL_FEATURES)}): {ALL_FEATURES}")


# ═══════════════════════════════════════════════════════════════
# STEP 7 — TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════
print("\n[STEP 7] Splitting data 80/20 (stratified by high_load_flag)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    # Stratify ensures high-load and low-load hospitals are
    # represented equally in both train and test sets.
    stratify=df['high_load_flag']
)

print(f"  Train: {X_train.shape[0]:,} rows  |  Test: {X_test.shape[0]:,} rows")


# ═══════════════════════════════════════════════════════════════
# STEP 8 — MODEL TRAINING
#
#   We train TWO models:
#     (A) GradientBoostingRegressor — sequential boosting, handles
#         non-linearity well, less prone to overfitting than a deep tree.
#     (B) RandomForestRegressor — parallel ensemble, fast, robust
#         baseline for comparison.
#
#   We pick the best model by test RMSE.
# ═══════════════════════════════════════════════════════════════
print("\n[STEP 8] Training models (GradientBoosting + RandomForest)...")

# ── Model A: Gradient Boosting ────────────────────────────────
gbr = GradientBoostingRegressor(
    n_estimators    = 300,   # number of boosting stages
    learning_rate   = 0.05,  # shrinkage — smaller = more robust, needs more trees
    max_depth       = 5,     # shallow trees limit overfitting
    min_samples_leaf= 10,    # regularises leaf size
    subsample       = 0.80,  # stochastic gradient boosting (row sampling)
    max_features    = 0.75,  # column sampling per split
    loss            = 'huber',  # robust to outliers vs squared-error
    random_state    = 42,
    validation_fraction = 0.1,
    n_iter_no_change    = 20,   # early stopping
    tol             = 1e-4,
)
gbr.fit(X_train, y_train)
print("  ✓ GradientBoostingRegressor trained")

# ── Model B: Random Forest ─────────────────────────────────────
rfr = RandomForestRegressor(
    n_estimators    = 300,
    max_depth       = 12,
    min_samples_leaf= 8,
    max_features    = 'sqrt',
    random_state    = 42,
    n_jobs          = -1,
)
rfr.fit(X_train, y_train)
print("  ✓ RandomForestRegressor trained")


# ═══════════════════════════════════════════════════════════════
# STEP 9 — EVALUATION
# ═══════════════════════════════════════════════════════════════
print("\n[STEP 9] Evaluating models on held-out test set...")

def evaluate_model(model, X_test, y_test, name):
    """
    Compute and print MAE, RMSE, R² for a regression model.

    MAE  (Mean Absolute Error):
        Average absolute difference between predicted and actual wait
        time. Measured in minutes — easy to interpret.
        Lower is better.

    RMSE (Root Mean Squared Error):
        Square root of average squared errors. Penalises large errors
        more heavily than MAE. Also in minutes.
        Lower is better.

    R²   (Coefficient of Determination):
        Proportion of variance in wait time explained by the model.
        1.0 = perfect, 0.0 = no better than predicting the mean,
        <0 = worse than predicting the mean.
        Higher is better.
    """
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)   # negative predictions are physically impossible

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    print(f"\n  ── {name} ──────────────────────────────")
    print(f"  MAE  (Mean Abs Error)   : {mae:.3f} mins")
    print(f"  RMSE (Root Mean Sq Err) : {rmse:.3f} mins")
    print(f"  R²   (Explained Var)    : {r2:.4f}")
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'preds': preds}

res_gbr = evaluate_model(gbr, X_test, y_test, "GradientBoostingRegressor")
res_rfr = evaluate_model(rfr, X_test, y_test, "RandomForestRegressor")

# Cross-validation for GBR (primary model)
print("\n  ── 5-Fold Cross-Validation (GBR, R²) ──")
cv_scores = cross_val_score(gbr, X, y, cv=5, scoring='r2', n_jobs=-1)
print(f"  CV R² scores : {np.round(cv_scores, 4)}")
print(f"  Mean CV R²   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Pick best model
best_model = gbr if res_gbr['rmse'] < res_rfr['rmse'] else rfr
best_name  = "GradientBoostingRegressor" if best_model is gbr else "RandomForestRegressor"
best_preds = res_gbr['preds'] if best_model is gbr else res_rfr['preds']
print(f"\n  ✓ Selected best model: {best_name}")


# ═══════════════════════════════════════════════════════════════
# STEP 10 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════
print("\n[STEP 10] Computing feature importances...")

importances_raw = best_model.feature_importances_
feat_imp = pd.Series(importances_raw, index=ALL_FEATURES).sort_values(ascending=False)
print("\n  Top 15 features:")
for feat, val in feat_imp.head(15).items():
    bar = "█" * int(val * 300)
    print(f"  {feat:30s} {val:.4f}  {bar}")


# ═══════════════════════════════════════════════════════════════
# STEP 11 — VISUALISATIONS
# ═══════════════════════════════════════════════════════════════
print("\n[STEP 11] Generating evaluation plots...")

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)
fig.suptitle("Model 5: OPD Wait Time Prediction — Evaluation Dashboard",
             fontsize=15, fontweight='bold', y=0.98)

# ── Plot 1: Actual vs Predicted ────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(y_test, best_preds, alpha=0.35, color='#1D9E75', s=12, label='Predictions')
min_val = min(y_test.min(), best_preds.min())
max_val = max(y_test.max(), best_preds.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Perfect fit')
ax1.set_xlabel("Actual Wait Time (mins)")
ax1.set_ylabel("Predicted Wait Time (mins)")
ax1.set_title("Actual vs Predicted Wait Time")
ax1.legend(); ax1.grid(alpha=0.3)

# ── Plot 2: Residual distribution ─────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
residuals = y_test.values - best_preds
ax2.hist(residuals, bins=40, color='#534AB7', edgecolor='white', alpha=0.85)
ax2.axvline(0, color='red', linestyle='--', lw=1.5)
ax2.set_xlabel("Residual (mins)")
ax2.set_ylabel("Frequency")
ax2.set_title("Residual Distribution")
ax2.grid(alpha=0.3)

# ── Plot 3: Feature Importances ────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
top_feats = feat_imp.head(12)
colors    = ['#1D9E75' if any(k in f for k in ['ratio', 'util', 'load', 'log'])
             else '#534AB7' for f in top_feats.index]
ax3.barh(range(len(top_feats)), top_feats.values, color=colors, alpha=0.88)
ax3.set_yticks(range(len(top_feats)))
ax3.set_yticklabels(top_feats.index, fontsize=9)
ax3.invert_yaxis()
ax3.set_xlabel("Importance Score")
ax3.set_title("Top 12 Feature Importances")
import matplotlib.patches as mpatches
p1 = mpatches.Patch(color='#1D9E75', label='Engineered features')
p2 = mpatches.Patch(color='#534AB7', label='Raw features')
ax3.legend(handles=[p1, p2], fontsize=8)
ax3.grid(axis='x', alpha=0.3)

# ── Plot 4: Wait time distribution ────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.hist(df['Wait_Time'], bins=40, color='#D85A30', edgecolor='white', alpha=0.85)
ax4.set_xlabel("Wait Time (mins)")
ax4.set_ylabel("Count")
ax4.set_title("Simulated Wait Time Distribution")
ax4.grid(alpha=0.3)

# ── Plot 5: Prediction vs Patient-Doctor Ratio ─────────────────
ax5 = fig.add_subplot(gs[2, 0])
scatter = ax5.scatter(
    X_test['patient_doctor_ratio'], best_preds,
    c=y_test.values, cmap='RdYlGn_r', alpha=0.5, s=10
)
plt.colorbar(scatter, ax=ax5, label='Actual Wait (mins)')
ax5.set_xlabel("Patient / Doctor Ratio")
ax5.set_ylabel("Predicted Wait (mins)")
ax5.set_title("Predictions vs P/D Ratio")
ax5.grid(alpha=0.3)

# ── Plot 6: Predicted vs utilization ──────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
ax6.scatter(X_test['utilization'], best_preds, alpha=0.35, color='#185FA5', s=10)
ax6.set_xlabel("Hospital Utilization")
ax6.set_ylabel("Predicted Wait (mins)")
ax6.set_title("Predictions vs Utilization")
ax6.axvline(0.8, color='red', linestyle='--', lw=1.2, label='80% load')
ax6.legend(); ax6.grid(alpha=0.3)

# ── Plot 7: Model comparison bar chart ────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
model_names = ['GBR', 'RandomForest']
maes  = [res_gbr['mae'], res_rfr['mae']]
rmses = [res_gbr['rmse'], res_rfr['rmse']]
x_pos = np.arange(len(model_names))
width = 0.35
ax7.bar(x_pos - width/2, maes,  width, label='MAE',  color='#1D9E75', alpha=0.85)
ax7.bar(x_pos + width/2, rmses, width, label='RMSE', color='#D85A30', alpha=0.85)
ax7.set_xticks(x_pos); ax7.set_xticklabels(model_names)
ax7.set_ylabel("Error (mins)")
ax7.set_title("Model Comparison (MAE vs RMSE)")
ax7.legend(); ax7.grid(axis='y', alpha=0.3)

os.makedirs('/home/claude/model_outputs', exist_ok=True)
plt.savefig('/home/claude/model_outputs/opd_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: model_outputs/opd_evaluation.png")


# ═══════════════════════════════════════════════════════════════
# STEP 12 — SAVE MODEL ARTIFACTS
# ═══════════════════════════════════════════════════════════════
print("\n[STEP 12] Saving model artifacts...")

artifacts = {
    'model':        best_model,
    'model_name':   best_name,
    'encoders':     encoders,
    'all_features': ALL_FEATURES,
    'raw_features': RAW_FEATURES,
    'eng_features': ENG_FEATURES,
    'enc_features': ENC_FEATURES,
}
with open('/home/claude/model_outputs/opd_model.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print(f"  ✓ Saved: model_outputs/opd_model.pkl")


# ═══════════════════════════════════════════════════════════════
# STEP 13 — PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════

def predict_wait_time(input_dict: dict) -> dict:
    """
    Predict OPD wait time from hospital infrastructure inputs.

    Parameters
    ----------
    input_dict : dict
        Must contain:
          Total_Beds, Available_Beds, ICU_Beds, Non_ICU_Beds,
          Total_Doctors, Ventilators, Total_Rooms, Emergency_Rooms,
          City, District, State, City_Type

        Patients is optional — if not provided, it is estimated
        as 70% of Total_Beds (typical moderate load).

    Returns
    -------
    dict
        {
          "Predicted Wait Time (mins)": float,
          "Confidence Band (±mins)":    float,
          "Risk Level":                 str,
          "Key Drivers":                dict
        }

    Raises
    ------
    ValueError  — if required keys are missing or values are invalid.
    """

    # ── Validate required keys ──────────────────────────────────
    REQUIRED_KEYS = [
        'Total_Beds', 'ICU_Beds', 'Non_ICU_Beds', 'Total_Doctors',
        'Ventilators', 'Total_Rooms', 'Emergency_Rooms',
        'City', 'District', 'State', 'City_Type'
    ]
    missing = [k for k in REQUIRED_KEYS if k not in input_dict]
    if missing:
        raise ValueError(f"Missing required input keys: {missing}")

    # ── Build a single-row DataFrame ────────────────────────────
    row = {k: v for k, v in input_dict.items()}

    # Estimate Patients if not provided
    total_beds = max(float(row.get('Total_Beds', 100)), 1.0)
    if 'Patients' not in row or row['Patients'] is None:
        row['Patients'] = int(total_beds * 0.70)   # default 70% load

    # Sanitise numeric values
    for key in ['Total_Beds','ICU_Beds','Non_ICU_Beds','Total_Doctors',
                'Ventilators','Total_Rooms','Emergency_Rooms','Patients']:
        val = float(row.get(key, 0))
        if val < 0:
            raise ValueError(f"'{key}' must be non-negative, got {val}")
        row[key] = max(val, EPS)   # floor at epsilon to avoid /0

    row_df = pd.DataFrame([row])

    # ── Feature engineering (mirrors training) ──────────────────
    row_df['utilization']          = safe_divide(row_df['Patients'],        row_df['Total_Beds'])
    row_df['patient_doctor_ratio'] = safe_divide(row_df['Patients'],        row_df['Total_Doctors'])
    row_df['icu_ratio']            = safe_divide(row_df['ICU_Beds'],        row_df['Total_Beds'])
    row_df['room_density']         = safe_divide(row_df['Total_Rooms'],     row_df['Total_Beds'])
    row_df['emergency_ratio']      = safe_divide(row_df['Emergency_Rooms'], row_df['Total_Rooms'])
    row_df['ventilator_ratio']     = safe_divide(row_df['Ventilators'],     row_df['Total_Beds'])
    row_df['bed_availability']     = safe_divide(row_df['Non_ICU_Beds'],    row_df['Total_Beds'])

    row_df['log_pd_ratio']  = np.log1p(row_df['patient_doctor_ratio'])
    row_df['log_patients']  = np.log1p(row_df['Patients'])
    row_df['log_beds']      = np.log1p(row_df['Total_Beds'])
    row_df['log_doctors']   = np.log1p(row_df['Total_Doctors'])
    row_df['load_x_pd']     = row_df['utilization'] * row_df['patient_doctor_ratio']
    row_df['high_load_flag']= (row_df['utilization'] > 0.80).astype(int)

    # Clip ratios to [0, 1]
    for col in ['utilization','icu_ratio','room_density','emergency_ratio',
                'ventilator_ratio','bed_availability']:
        row_df[col] = row_df[col].clip(0, 1)

    # ── Encode categoricals (handle unseen labels gracefully) ────
    for col, le in encoders.items():
        raw_val = str(row.get(col, 'Unknown'))
        # Map unseen label to 'Unknown' sentinel added during fit
        if raw_val not in le.classes_:
            raw_val = 'Unknown'
        row_df[col + '_enc'] = le.transform([raw_val])

    # ── Select features in training order ───────────────────────
    X_input = row_df[ALL_FEATURES].fillna(0)

    # ── Predict ─────────────────────────────────────────────────
    raw_prediction = best_model.predict(X_input)[0]

    # Clip to physically valid range
    prediction = float(np.clip(raw_prediction, 5.0, 120.0))

    # Confidence band: derived from test-set MAE (model uncertainty proxy)
    confidence_band = round(
        res_gbr['mae'] if best_model is gbr else res_rfr['mae'], 1
    )

    # Risk level based on predicted wait
    if prediction < 20:
        risk = "Low 🟢"
    elif prediction < 45:
        risk = "Moderate 🟡"
    elif prediction < 75:
        risk = "High 🟠"
    else:
        risk = "Critical 🔴"

    # Key driver values for interpretability
    key_drivers = {
        "Patients":             int(row['Patients']),
        "Utilization (%)":      round(float(row_df['utilization'].iloc[0]) * 100, 1),
        "Patient/Doctor Ratio": round(float(row_df['patient_doctor_ratio'].iloc[0]), 2),
        "High Load Flag":       int(row_df['high_load_flag'].iloc[0]),
    }

    return {
        "Predicted Wait Time (mins)": round(prediction, 1),
        "Confidence Band (±mins)":    confidence_band,
        "Risk Level":                 risk,
        "Key Drivers":                key_drivers,
    }


# ═══════════════════════════════════════════════════════════════
# STEP 14 — EXAMPLE PREDICTIONS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 14: EXAMPLE PREDICTIONS")
print("=" * 65)

test_cases = [
    {
        "name":  "Large Urban Hospital — Moderate Load",
        "input": {
            "Total_Beds": 150, "Available_Beds": 40, "ICU_Beds": 30,
            "Non_ICU_Beds": 120, "Total_Doctors": 5, "Ventilators": 10,
            "Total_Rooms": 80, "Emergency_Rooms": 10,
            "City": "Mumbai", "District": "Mumbai",
            "State": "Maharashtra", "City_Type": "Urban",
            "Patients": 80,
        }
    },
    {
        "name":  "Small Rural Clinic — Overloaded",
        "input": {
            "Total_Beds": 30, "Available_Beds": 2, "ICU_Beds": 3,
            "Non_ICU_Beds": 27, "Total_Doctors": 1, "Ventilators": 2,
            "Total_Rooms": 15, "Emergency_Rooms": 2,
            "City": "Bhimtal", "District": "Nainital",
            "State": "Uttarakhand", "City_Type": "Rural",
            "Patients": 28,
        }
    },
    {
        "name":  "Mid-size Semi-Urban Hospital — Low Load",
        "input": {
            "Total_Beds": 80, "Available_Beds": 50, "ICU_Beds": 10,
            "Non_ICU_Beds": 70, "Total_Doctors": 8, "Ventilators": 6,
            "Total_Rooms": 45, "Emergency_Rooms": 5,
            "City": "Nagpur", "District": "Nagpur",
            "State": "Maharashtra", "City_Type": "Semi-Urban",
            "Patients": 25,
        }
    },
    {
        "name":  "Unseen City (tests graceful unknown label handling)",
        "input": {
            "Total_Beds": 100, "Available_Beds": 20, "ICU_Beds": 15,
            "Non_ICU_Beds": 85, "Total_Doctors": 4, "Ventilators": 8,
            "Total_Rooms": 60, "Emergency_Rooms": 7,
            "City": "Atlantis_XYZ",   # ← unseen city
            "District": "Unknown_District",
            "State": "Fantasy_State",
            "City_Type": "Urban",
            "Patients": 70,
        }
    },
]

for tc in test_cases:
    print(f"\n  Scenario: {tc['name']}")
    try:
        result = predict_wait_time(tc['input'])
        print(f"  ┌─ Predicted Wait Time : {result['Predicted Wait Time (mins)']} mins")
        print(f"  ├─ Confidence Band     : ±{result['Confidence Band (±mins)']} mins")
        print(f"  ├─ Risk Level          : {result['Risk Level']}")
        for k, v in result['Key Drivers'].items():
            print(f"  │    {k}: {v}")
        print(f"  └─ Done ✓")
    except (ValueError, KeyError) as e:
        print(f"  ✗ Error handled gracefully: {e}")

# ── Error handling demo ────────────────────────────────────────
print("\n  ── Error Handling Demos ──────────────────────────────")

# Demo 1: missing key
try:
    predict_wait_time({"Total_Beds": 100})
except ValueError as e:
    print(f"  Missing key caught   : {str(e)[:80]}")

# Demo 2: negative value
try:
    bad_input = test_cases[0]['input'].copy()
    bad_input['Total_Doctors'] = -5
    predict_wait_time(bad_input)
except ValueError as e:
    print(f"  Negative value caught: {e}")

print("\n" + "=" * 65)
print("  PIPELINE COMPLETE")
print("=" * 65)
print(f"\n  Best Model : {best_name}")
print(f"  Test MAE   : {(res_gbr if best_model is gbr else res_rfr)['mae']:.3f} mins")
print(f"  Test RMSE  : {(res_gbr if best_model is gbr else res_rfr)['rmse']:.3f} mins")
print(f"  Test R²    : {(res_gbr if best_model is gbr else res_rfr)['r2']:.4f}")
print(f"  CV R²      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("\n  Artifacts saved:")
print("    model_outputs/opd_model.pkl")
print("    model_outputs/opd_evaluation.png")
print("=" * 65)
