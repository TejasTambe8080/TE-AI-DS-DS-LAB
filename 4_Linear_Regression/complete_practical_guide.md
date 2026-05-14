# PRACTICAL 4: LINEAR REGRESSION

## Practical Title
Linear Regression - Predicting House Prices with Boston Housing Dataset

## Aim
To build and evaluate a linear regression model to predict house prices using multiple features from the Boston Housing dataset.

## Objective
1. Understand univariate and multivariate linear regression
2. Implement linear regression using scikit-learn
3. Split data into training and testing sets
4. Calculate performance metrics (MSE, RMSE, R²)
5. Interpret model coefficients
6. Make predictions on new data

## Problem Statement
Given the Boston Housing dataset (506 samples, 13 features), build a linear regression model to predict house prices (MEDV) based on features like number of rooms (RM), crime rate (CRIM), distance to employment (DIS), etc.

---

## THEORY

### 1. Linear Regression: Definition

Linear regression models relationship between dependent variable (Y) and independent variables (X) using linear equation:
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n + \epsilon$$

Where:
- Y = target (dependent) variable
- X = features (independent) variables
- β₀ = intercept
- βᵢ = slopes (coefficients)
- ε = error term

### 2. Univariate vs Multivariate Regression

**Univariate:** One feature
$$Y = \beta_0 + \beta_1 X + \epsilon$$

**Multivariate:** Multiple features
$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n + \epsilon$$

### 3. Least Squares Method

Find β₀ and β₁ that minimize sum of squared residuals:
$$\text{Minimize: } \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2$$

Solutions:
$$\beta_1 = \frac{\sum(X_i - \bar{X})(Y_i - \bar{Y})}{\sum(X_i - \bar{X})^2}$$

$$\beta_0 = \bar{Y} - \beta_1 \bar{X}$$

### 4. Performance Metrics

**Mean Squared Error (MSE)**
$$MSE = \frac{1}{n}\sum(Y_{actual} - Y_{predicted})^2$$
- Units: Squared (hard to interpret)
- Lower is better
- Sensitive to large errors

**Root Mean Squared Error (RMSE)**
$$RMSE = \sqrt{MSE}$$
- Same units as Y
- Easier to interpret
- Lower is better

**R-Squared (Coefficient of Determination)**
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = \frac{SS_{reg}}{SS_{tot}}$$

Where:
- SSᵣₑₛ = sum of squared residuals
- SS_tot = total sum of squares

Interpretation:
- Range: 0 to 1
- R² = 1: Perfect fit
- R² = 0: Model no better than mean
- R² > 0.7: Good model
- R² < 0.3: Poor model

### 5. Training vs Testing

**Training Set (70-80%):**
- Used to fit model
- Model learns from this data
- Training error (Ein) typically low

**Testing Set (20-30%):**
- Used to evaluate model
- Model never sees this data
- Testing error (Eout) realistic performance indicator
- Usually Eout > Ein

### 6. Boston Housing Dataset

- **Samples:** 506 houses
- **Features:** 13 (CRIM, INDUS, RM, AGE, DIS, etc.)
- **Target:** MEDV (Median Value in $1000s)
- **Price Range:** $5,000 - $50,000

---

## ALGORITHM

**Step 1:** Import libraries and load dataset

**Step 2:** Check for missing values and perform preprocessing

**Step 3:** Separate features (X) and target (y)

**Step 4:** Normalize features (optional but recommended)

**Step 5:** Split data: 80% training, 20% testing

**Step 6:** Create linear regression model

**Step 7:** Train model on training data

**Step 8:** Predict on both training and testing data

**Step 9:** Calculate MSE, RMSE, R² for both sets

**Step 10:** Visualize actual vs predicted values

**Step 11:** Interpret model coefficients

---

## COMPLETE CODE

```python
# ===== PRACTICAL 4: LINEAR REGRESSION =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("=" * 80)
print("PRACTICAL 4: LINEAR REGRESSION - BOSTON HOUSING PRICE PREDICTION")
print("=" * 80)

# ===== STEP 1: LOAD DATASET =====
print("\nSTEP 1: LOAD AND EXPLORE BOSTON HOUSING DATASET")
print("-" * 80)

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name='PRICE')

df = X.copy()
df['PRICE'] = y

print(f"Dataset shape: {df.shape}")
print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nFeature Names:")
for i, feat in enumerate(boston.feature_names, 1):
    print(f"  {i}. {feat.strip()}")

print(f"\nDataset Statistics:")
print(df.describe())

# ===== STEP 2: CHECK DATA QUALITY =====
print("\n" + "=" * 80)
print("STEP 2: DATA QUALITY CHECK")
print("-" * 80)

print(f"\nMissing Values: {df.isnull().sum().sum()}")
print(f"Data Types:\n{df.dtypes}")

# ===== STEP 3: SEPARATE FEATURES AND TARGET =====
print("\n" + "=" * 80)
print("STEP 3: SEPARATE FEATURES AND TARGET")
print("-" * 80)

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"\nTarget Variable (Price) Statistics:")
print(f"  Mean: ${y.mean()*1000:.2f}")
print(f"  Median: ${y.median()*1000:.2f}")
print(f"  Std Dev: ${y.std()*1000:.2f}")
print(f"  Range: ${y.min()*1000:.2f} - ${y.max()*1000:.2f}")

# ===== STEP 4: NORMALIZE FEATURES =====
print("\n" + "=" * 80)
print("STEP 4: FEATURE NORMALIZATION")
print("-" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print(f"\nBefore Scaling:")
print(f"  X mean: {X.mean().mean():.4f}, X std: {X.std().mean():.4f}")

print(f"\nAfter Scaling:")
print(f"  X_scaled mean: {X_scaled_df.mean().mean():.6f}, X_scaled std: {X_scaled_df.std().mean():.6f}")

# ===== STEP 5: TRAIN-TEST SPLIT =====
print("\n" + "=" * 80)
print("STEP 5: SPLIT DATA INTO TRAINING (80%) AND TESTING (20%)")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples ({(len(X_train)/len(X))*100:.1f}%)")
print(f"Testing set size: {X_test.shape[0]} samples ({(len(X_test)/len(X))*100:.1f}%)")

print(f"\nTraining Target Statistics:")
print(f"  Mean: ${y_train.mean()*1000:.2f}")
print(f"  Std: ${y_train.std()*1000:.2f}")

print(f"\nTesting Target Statistics:")
print(f"  Mean: ${y_test.mean()*1000:.2f}")
print(f"  Std: ${y_test.std()*1000:.2f}")

# ===== STEP 6: CREATE MODEL =====
print("\n" + "=" * 80)
print("STEP 6: CREATE LINEAR REGRESSION MODEL")
print("-" * 80)

model = LinearRegression()
print("Linear Regression model created")
print(f"Model parameters: LinearRegression()")

# ===== STEP 7: TRAIN MODEL =====
print("\n" + "=" * 80)
print("STEP 7: TRAIN MODEL ON TRAINING DATA")
print("-" * 80)

model.fit(X_train, y_train)
print("Model trained successfully!")

print(f"\nModel Coefficients (β):")
print(f"  Intercept (β₀): {model.intercept_:.6f}")
print(f"\n  Feature Coefficients (β₁...β₁₃):")

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

print(coef_df.to_string(index=False))

print(f"\nInterpretation:")
print(f"  Most positive impact: {coef_df.iloc[0]['Feature']} ({coef_df.iloc[0]['Coefficient']:.4f})")
print(f"  Most negative impact: {coef_df.iloc[-1]['Feature']} ({coef_df.iloc[-1]['Coefficient']:.4f})")

# ===== STEP 8: MAKE PREDICTIONS =====
print("\n" + "=" * 80)
print("STEP 8: MAKE PREDICTIONS ON TRAINING AND TESTING DATA")
print("-" * 80)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print(f"Predictions generated for both sets")
print(f"\nFirst 5 Training Predictions vs Actual:")
comparison_train = pd.DataFrame({
    'Actual': y_train.iloc[:5].values * 1000,
    'Predicted': y_train_pred[:5] * 1000,
    'Error': (y_train.iloc[:5].values - y_train_pred[:5]) * 1000
})
print(comparison_train.round(2))

print(f"\nFirst 5 Testing Predictions vs Actual:")
comparison_test = pd.DataFrame({
    'Actual': y_test.iloc[:5].values * 1000,
    'Predicted': y_test_pred[:5] * 1000,
    'Error': (y_test.iloc[:5].values - y_test_pred[:5]) * 1000
})
print(comparison_test.round(2))

# ===== STEP 9: EVALUATE MODEL =====
print("\n" + "=" * 80)
print("STEP 9: MODEL EVALUATION - PERFORMANCE METRICS")
print("-" * 80)

# Training metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Testing metrics
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nTRAINING SET METRICS:")
print(f"  MSE (Mean Squared Error): {train_mse:.6f}")
print(f"  RMSE (Root Mean Squared Error): {train_rmse:.6f} (${train_rmse*1000:.2f})")
print(f"  MAE (Mean Absolute Error): {train_mae:.6f} (${train_mae*1000:.2f})")
print(f"  R² (Coefficient of Determination): {train_r2:.6f}")

print(f"\nTESTING SET METRICS:")
print(f"  MSE (Mean Squared Error): {test_mse:.6f}")
print(f"  RMSE (Root Mean Squared Error): {test_rmse:.6f} (${test_rmse*1000:.2f})")
print(f"  MAE (Mean Absolute Error): {test_mae:.6f} (${test_mae*1000:.2f})")
print(f"  R² (Coefficient of Determination): {test_r2:.6f}")

print(f"\nMODEL INTERPRETATION:")
print(f"  R² Train: {train_r2:.4f} → Model explains {train_r2*100:.2f}% of training variance")
print(f"  R² Test: {test_r2:.4f} → Model explains {test_r2*100:.2f}% of testing variance")
print(f"  RMSE Train: ${train_rmse*1000:.2f} → Average prediction error (train)")
print(f"  RMSE Test: ${test_rmse*1000:.2f} → Average prediction error (test)")

if abs(train_r2 - test_r2) < 0.05:
    print(f"\n  → Model is well-generalized (training ≈ testing)")
elif train_r2 > test_r2:
    print(f"\n  → Possible overfitting (training better than testing)")
else:
    print(f"\n  → Possible underfitting (test better than training)")

# ===== STEP 10: VISUALIZATION =====
print("\n" + "=" * 80)
print("STEP 10: VISUALIZATION")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Actual vs Predicted (Training)
axes[0, 0].scatter(y_train, y_train_pred, alpha=0.5, s=20, color='blue')
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Price ($1000s)')
axes[0, 0].set_ylabel('Predicted Price ($1000s)')
axes[0, 0].set_title(f'Training Data (R² = {train_r2:.4f})', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted (Testing)
axes[0, 1].scatter(y_test, y_test_pred, alpha=0.5, s=20, color='green')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Price ($1000s)')
axes[0, 1].set_ylabel('Predicted Price ($1000s)')
axes[0, 1].set_title(f'Testing Data (R² = {test_r2:.4f})', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Residuals (Training)
residuals_train = y_train - y_train_pred
axes[1, 0].scatter(y_train_pred, residuals_train, alpha=0.5, s=20, color='blue')
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Price ($1000s)')
axes[1, 0].set_ylabel('Residuals ($1000s)')
axes[1, 0].set_title('Residuals Plot - Training Data', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Residuals (Testing)
residuals_test = y_test - y_test_pred
axes[1, 1].scatter(y_test_pred, residuals_test, alpha=0.5, s=20, color='green')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Predicted Price ($1000s)')
axes[1, 1].set_ylabel('Residuals ($1000s)')
axes[1, 1].set_title('Residuals Plot - Testing Data', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_results.png', dpi=150, bbox_inches='tight')
print("Visualization saved as 'linear_regression_results.png'")
plt.show()

# ===== STEP 11: SUMMARY =====
print("\n" + "=" * 80)
print("STEP 11: MODEL SUMMARY AND CONCLUSION")
print("=" * 80)

print(f"\nFinal Model Equation:")
print(f"PRICE = {model.intercept_:.4f}", end='')
for feat, coef in zip(X.columns, model.coef_):
    sign = '+' if coef >= 0 else '-'
    print(f" {sign} {abs(coef):.4f} × {feat.strip()}", end='')
print()

print(f"\nModel Performance Summary:")
print(f"{'Metric':<20} {'Training':<15} {'Testing':<15} {'Status':<15}")
print("-" * 65)
print(f"{'R²':<20} {train_r2:<15.4f} {test_r2:<15.4f} {'Good' if test_r2 > 0.7 else 'Fair' if test_r2 > 0.5 else 'Poor':<15}")
print(f"{'RMSE':<20} ${train_rmse*1000:<14.2f} ${test_rmse*1000:<14.2f} {'' if abs(train_rmse-test_rmse)<0.05 else 'Gap detected':<15}")
print(f"{'MAE':<20} ${train_mae*1000:<14.2f} ${test_mae*1000:<14.2f}")

print(f"\n✓ LINEAR REGRESSION PRACTICAL COMPLETED!")
print("=" * 80)
```

---

## FORMULAS EXPLAINED

### Linear Regression Equation
$$Y = \beta_0 + \beta_1 X + \epsilon$$

### Least Squares Estimates
$$\beta_1 = \frac{\sum(X_i - \bar{X})(Y_i - \bar{Y})}{\sum(X_i - \bar{X})^2}$$

$$\beta_0 = \bar{Y} - \beta_1 \bar{X}$$

### MSE
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(Y_i - \hat{Y}_i)^2$$

### RMSE
$$RMSE = \sqrt{MSE}$$

### R²
$$R^2 = 1 - \frac{\sum(Y_i - \hat{Y}_i)^2}{\sum(Y_i - \bar{Y})^2}$$

---

## IMPORTANT FUNCTIONS

| Function | Purpose |
|----------|---------|
| `train_test_split()` | Split data into train/test |
| `StandardScaler()` | Normalize features |
| `LinearRegression()` | Create model |
| `fit()` | Train model |
| `predict()` | Make predictions |
| `mean_squared_error()` | Calculate MSE |
| `r2_score()` | Calculate R² |

---

## EXTERNAL VIVA QUESTIONS (Select 20)

### Q1: What is linear regression used for?
**A:** Predict continuous values based on one or more features using linear relationship.

### Q2: Explain the difference between univariate and multivariate regression.
**A:** Univariate: one feature. Multivariate: multiple features.

### Q3: What does intercept (β₀) represent?
**A:** Value of Y when all X=0. Where regression line crosses Y-axis.

### Q4: How to interpret coefficient β₁ = 2.5?
**A:** For each unit increase in X, Y increases by 2.5 units (holding others constant).

### Q5: Why split data into training and testing?
**A:** Prevent overfitting, get realistic performance estimate on unseen data.

### Q6: What ratio should training/testing be?
**A:** Typically 70-80% training, 20-30% testing. More data = better training.

### Q7: Explain MSE vs RMSE.
**A:** MSE: squared error (hard interpret). RMSE: square root (same units as Y, easier interpret).

### Q8: When is R² = 1?
**A:** Perfect fit - model explains 100% of variance. All actual = predicted.

### Q9: Is R² = 0.5 good?
**A:** Depends on context - explains 50% variance. Usually acceptable for complex data.

### Q10: Why normalize features before regression?
**A:** Features on different scales - prevents large-scale features from dominating.

### Q11: What does residual mean?
**A:** Difference between actual and predicted: Residual = Y_actual - Y_predicted.

### Q12: How to detect overfitting?
**A:** Training R² much higher than testing R². Model memorizes training, fails on test.

### Q13: Can negative R² occur?
**A:** Yes, if model worse than predicting mean Y. Indicates very poor model.

### Q14: What assumptions does linear regression make?
**A:** (1) Linearity, (2) No multicollinearity, (3) Homoscedasticity, (4) Normal residuals.

### Q15: How to improve poor model?
**A:** (1) Add features, (2) Remove outliers, (3) Feature engineering, (4) Regularization.

### Q16: What's multicollinearity problem?
**A:** Features highly correlated with each other - makes coefficients unstable, hard interpret.

### Q17: Can you predict outside data range?
**A:** Yes, but risky (extrapolation) - assumptions may not hold beyond observed range.

### Q18: Why use StandardScaler before regression?
**A:** Makes features comparable scale, improves convergence, easier interpretation.

### Q19: What does high residual indicate?
**A:** Prediction far from actual - outlier or model doesn't fit that point well.

### Q20: Explain train-test gap in R².
**A:** If gap small: good generalization. If gap large: possible overfitting.

---

## CONCLUSION

Linear Regression is fundamental ML algorithm:
- **Simple** yet powerful
- **Interpretable** coefficients
- **Foundation** for advanced methods
- **R²** shows goodness of fit
- **Proper evaluation** requires train-test split

---

**Ready for External Examination and Journal Submission**
