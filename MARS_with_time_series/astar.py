import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ============================================================================
# DATA
# ============================================================================

df_raw_unfilter = pd.read_csv("data/final_economic_data_regional.csv")
df_raw = df_raw_unfilter[df_raw_unfilter['Region'] == 1].reset_index(drop=True)
df_raw = df_raw[(df_raw['Year'] > 2010) & (df_raw['Year'] < 2022)]
# ============================================================================
# CONFIG
# ============================================================================

TARGET_VARIABLE = 'GDP per capita (euro at current prices)'  
FEATURE_VARIABLES = [
    'A Agriculture, forestry and fishing (TP)',
    'H Transportation and storage (TP)', 
    'Unemployed jobseekers',
    'Mean of debt for all debts',
    'With education, total (population)',
    'Gross value added (millions of euro), F Construction (41-43)',
    # 'ImportGrowth',
    # 'TradeBalance',
    # 'DebtToGDP',
    # 'Productivity',
]
MAX_LAGS = 2 # How many year are lagged for the training
TRAIN_FRACTION = 0.6  
ADD_CYCLICAL_FEATURES = False  # Set to True if you think there's a regular cycle
CYCLE_LENGTH = 10  # If cyclical features enabled, specify the cycle length in years

# Model hyperparameters (tuned for small datasets)
MAX_TERMS = 3  # Maximum number of basis functions
Q_KNOTS = 4  # Number of candidate knot points
PENALTY = 1.4  # GCV penalty (higher = simpler model)
MIN_IMPROVE = 1e-2  # Minimum improvement to add a term

# ============================================================================
# MARS MODEL 
# ============================================================================

def hinge(vec: np.ndarray, c: float, direction: int) -> np.ndarray:
    if direction == 1:
        return np.maximum(0.0, vec - c)
    else:
        return np.maximum(0.0, c - vec)

@dataclass
class BasisSpec:
    kind: str
    feat_idx: int
    knot: Optional[float]
    direction: Optional[int]

@dataclass
class MarsLiteTS:
    basis: List[BasisSpec]
    beta: np.ndarray
    feature_names: List[str]

def design_matrix(X: np.ndarray, basis_list: List[BasisSpec]) -> np.ndarray:
    parts = [np.ones((X.shape[0], 1))]
    for b in basis_list:
        col = X[:, b.feat_idx]
        if b.kind == "linear":
            vec = col
        else:
            vec = hinge(col, b.knot, b.direction)
        parts.append(vec.reshape(-1, 1))
    return np.hstack(parts)

def gcv(y_true: np.ndarray, y_hat: np.ndarray, m_terms: int, penalty: float = 2.0) -> float:
    n = len(y_true)
    rss = np.sum((y_true - y_hat) ** 2)
    c_eff = 1 + penalty * m_terms
    denom = max(1e-3, (1 - c_eff / n))
    return rss / (n * denom ** 2)

def fit_mars_lite_ts(X: np.ndarray, y: np.ndarray, max_terms: int = 20, q_knots: int = 15,
                     allow_linear: bool = True, penalty: float = 2.0, min_improve: float = 1e-3) -> MarsLiteTS:
    n, p = X.shape
    quantiles = np.linspace(0.05, 0.95, q_knots)
    cand_knots = [np.quantile(X[:, j], quantiles) for j in range(p)]

    basis_list: List[BasisSpec] = []
    D = design_matrix(X, basis_list)
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    yhat = D @ beta
    best_score = gcv(y, yhat, m_terms=0, penalty=penalty)

    for _ in range(max_terms):
        best_add = None
        best_add_score = best_score
        if allow_linear:
            for j in range(p):
                spec = BasisSpec(kind="linear", feat_idx=j, knot=None, direction=None)
                D_try = design_matrix(X, basis_list + [spec])
                beta_try, *_ = np.linalg.lstsq(D_try, y, rcond=None)
                yhat_try = D_try @ beta_try
                score = gcv(y, yhat_try, m_terms=len(basis_list) + 1, penalty=penalty)
                if score < best_add_score - 1e-12:
                    best_add_score = score
                    best_add = (spec, beta_try, yhat_try)
        for j in range(p):
            for c in cand_knots[j]:
                for d in (1, -1):
                    spec = BasisSpec(kind="hinge", feat_idx=j, knot=float(c), direction=d)
                    D_try = design_matrix(X, basis_list + [spec])
                    beta_try, *_ = np.linalg.lstsq(D_try, y, rcond=None)
                    yhat_try = D_try @ beta_try
                    score = gcv(y, yhat_try, m_terms=len(basis_list) + 1, penalty=penalty)
                    if score < best_add_score - 1e-12:
                        best_add_score = score
                        best_add = (spec, beta_try, yhat_try)
        if best_add is None or (best_score - best_add_score) < min_improve:
            break
        spec, beta, yhat = best_add
        basis_list.append(spec)
        best_score = best_add_score

    improved = True
    while improved and len(basis_list) > 0:
        improved = False
        current_best = best_score
        drop_idx = None
        for j in range(len(basis_list)):
            trial = [b for i, b in enumerate(basis_list) if i != j]
            D_try = design_matrix(X, trial)
            beta_try, *_ = np.linalg.lstsq(D_try, y, rcond=None)
            yhat_try = D_try @ beta_try
            score = gcv(y, yhat_try, m_terms=len(trial), penalty=penalty)
            if score < current_best - 1e-12:
                current_best = score
                drop_idx = j
                best_beta_try = beta_try
                best_yhat_try = yhat_try
        if drop_idx is not None:
            basis_list.pop(drop_idx)
            beta = best_beta_try
            yhat = best_yhat_try
            best_score = current_best
            improved = True

    D_final = design_matrix(X, basis_list)
    beta_final, *_ = np.linalg.lstsq(D_final, y, rcond=None)
    return MarsLiteTS(basis=basis_list, beta=beta_final, feature_names=[])

def predict(model: MarsLiteTS, X: np.ndarray) -> np.ndarray:
    D = design_matrix(X, model.basis)
    return D @ model.beta

# ============================================================================
# DATA PREPARATION
# ============================================================================

def make_lag_features(df: pd.DataFrame, target_col: str, feature_cols: List[str], 
                     max_lag: int, add_cyclical: bool = False, cycle_length: int = 10) -> pd.DataFrame:
    """Create lagged features for time series prediction"""
    result = pd.DataFrame()
    
    # Add lags of the target variable
    for lag in range(1, max_lag + 1):
        result[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
    
    # Add lags of feature variables
    for col in feature_cols:
        for lag in range(0, max_lag + 1):  # Include lag 0 (current value)
            result[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    # Add time index
    result['time_idx'] = np.arange(len(df))
    
    # Add cyclical features if requested
    if add_cyclical:
        result[f'sin_{cycle_length}'] = np.sin(2 * np.pi * np.arange(len(df)) / cycle_length)
        result[f'cos_{cycle_length}'] = np.cos(2 * np.pi * np.arange(len(df)) / cycle_length)
    
    # Add target
    result['target'] = df[target_col].values
    
    return result

# Prepare the data
print(f"Target variable: {TARGET_VARIABLE}")
print(f"Feature variables: {FEATURE_VARIABLES}")
print(f"Using {MAX_LAGS} lags")


df = make_lag_features(df_raw, TARGET_VARIABLE, FEATURE_VARIABLES, 
                       MAX_LAGS, ADD_CYCLICAL_FEATURES, CYCLE_LENGTH)
df = df.dropna().reset_index(drop=True)

print(f"Rows after creating lag features: {len(df)}")



split_idx = max(int(len(df) * TRAIN_FRACTION), len(df) - 3)
    
X_train = df.drop(columns=['target']).iloc[:split_idx].to_numpy()
y_train = df['target'].iloc[:split_idx].to_numpy()
X_test = df.drop(columns=['target']).iloc[split_idx:].to_numpy()
y_test = df['target'].iloc[split_idx:].to_numpy()

feature_names = df.drop(columns=['target']).columns.tolist()

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Number of features: {X_train.shape[1]}")

# ============================================================================
# TRAINING
# ============================================================================

print(f"\nTraining MARS model")
model = fit_mars_lite_ts(X_train, y_train, max_terms=MAX_TERMS, q_knots=Q_KNOTS, 
                         allow_linear=True, penalty=PENALTY, min_improve=MIN_IMPROVE)
model.feature_names = feature_names

print(f"Model trained with {len(model.basis)} basis functions")


print(f"Intercept: {model.beta[0]:.4f}")
for i, basis in enumerate(model.basis):
    coef = model.beta[i + 1]
    fname = feature_names[basis.feat_idx]
    if basis.kind == "linear":
        print(f"  {coef:+.4f} × {fname}")
    else:
        direction = ">" if basis.direction == 1 else "<"
        print(f"  {coef:+.4f} × max(0, {fname} {direction} {basis.knot:.4f})")

# ============================================================================
# EVALUATE
# ============================================================================

y_train_pred = predict(model, X_train)
y_test_pred = predict(model, X_test)

train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
train_mape = np.mean(np.abs((y_train - y_train_pred) / np.maximum(1e-6, np.abs(y_train)))) * 100
test_mape = np.mean(np.abs((y_test - y_test_pred) / np.maximum(1e-6, np.abs(y_test)))) * 100

print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE:  {test_rmse:.4f}")
print(f"Train MAPE: {train_mape:.2f}%")
print(f"Test MAPE:  {test_mape:.2f}%")

# ============================================================================
# VISUALIZATION
# ============================================================================

# Get the actual years for plotting
years_with_lags = df_raw['Year'].iloc[MAX_LAGS:].values
train_years = years_with_lags[:split_idx]
test_years = years_with_lags[split_idx:]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Full time series
ax1.plot(train_years, y_train, 'o-', label='Training Actual', alpha=0.7, markersize=4)
ax1.plot(test_years, y_test, 's-', label='Test Actual', alpha=0.7, markersize=4)
ax1.plot(train_years, y_train_pred, '--', label='Training Predicted', alpha=0.8, linewidth=2)
ax1.plot(test_years, y_test_pred, '--', label='Test Predicted', alpha=0.8, linewidth=2)
ax1.axvline(train_years[-1], color='red', linestyle=':', label='Train/Test Split', linewidth=2)
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel(TARGET_VARIABLE, fontsize=11)
ax1.set_title(f'MARS Time Series Forecast: {TARGET_VARIABLE}', fontsize=13, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

ax2.scatter(train_years, train_residuals, label='Train Residuals', alpha=0.6)
ax2.scatter(test_years, test_residuals, label='Test Residuals', alpha=0.6, marker='s')
ax2.axhline(0, color='black', linestyle='-', linewidth=1)
ax2.axvline(train_years[-1], color='red', linestyle=':', linewidth=2)
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel('Residual', fontsize=11)
ax2.set_title('Prediction Residuals', fontsize=12, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Optional: Feature importance (based on coefficient magnitudes)
print("\n" + "="*60)
print("FEATURE IMPORTANCE (by coefficient magnitude)")
print("="*60)
feature_importance = {}
for i, basis in enumerate(model.basis):
    fname = feature_names[basis.feat_idx]
    coef = abs(model.beta[i + 1])
    if fname not in feature_importance:
        feature_importance[fname] = 0
    feature_importance[fname] += coef

sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
for fname, importance in sorted_importance[:10]:  # Top 10
    print(f"  {fname}: {importance:.4f}")