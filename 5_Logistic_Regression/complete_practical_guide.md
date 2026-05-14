# PRACTICAL 5: LOGISTIC REGRESSION

## Practical Title
Logistic Regression - Binary Classification on Social Network Ads Dataset

## Aim
To build a logistic regression model for binary classification to predict whether a user will purchase a product based on social network ad data.

## Objective
1. Understand logistic regression and sigmoid function
2. Implement binary classification
3. Calculate confusion matrix (TP, FP, TN, FN)
4. Compute accuracy, precision, recall, F1-score
5. Interpret classification results

## Problem Statement
Given Social Network Ads dataset, predict purchase (1) vs no purchase (0) based on age and estimated salary.

---

## THEORY

### 1. Logistic Regression: Overview
Logistic regression predicts probability of binary outcome (0/1):
$$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}} = \frac{e^z}{1 + e^z}$$

Where z = β₀ + β₁X

### 2. Sigmoid Function
- Transforms linear output to probability [0,1]
- S-shaped curve
- At z=0, P=0.5 (decision boundary)
- Smooth, differentiable

### 3. Log Odds and Logit
**Odds:** $\frac{P}{1-P}$

**Log Odds:** $\log(\frac{P}{1-P}) = \beta_0 + \beta_1 X$

### 4. Decision Rule
- If P ≥ 0.5: Predict Class 1
- If P < 0.5: Predict Class 0

### 5. Confusion Matrix (2×2)

|  | Predicted: 0 | Predicted: 1 |
|---|---|---|
| **Actual: 0** | TN | FP |
| **Actual: 1** | FN | TP |

Where:
- TP (True Positive): Correct, predicted 1
- FP (False Positive): Type I error, predicted 1 but actual 0
- TN (True Negative): Correct, predicted 0
- FN (False Negative): Type II error, predicted 0 but actual 1

### 6. Performance Metrics

**Accuracy**
$$\text{Accuracy} = \frac{TP + TN}{TP + FP + FN + TN}$$
Overall correctness

**Precision** (Positive Predictive Value)
$$\text{Precision} = \frac{TP}{TP + FP}$$
Of predicted positives, how many actually positive?

**Recall** (Sensitivity, True Positive Rate)
$$\text{Recall} = \frac{TP}{TP + FN}$$
Of actual positives, how many did we find?

**F1-Score**
$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
Harmonic mean of precision and recall

**Specificity** (True Negative Rate)
$$\text{Specificity} = \frac{TN}{TN + FP}$$

### 7. When to Use Which Metric?

- **Accuracy:** Balanced dataset
- **Precision:** When false positives costly (spam detection)
- **Recall:** When false negatives costly (disease detection)
- **F1:** Overall balance, imbalanced data

---

## COMPLETE CODE

```python
# ===== PRACTICAL 5: LOGISTIC REGRESSION =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, classification_report,
                             roc_curve, auc)

print("=" * 80)
print("PRACTICAL 5: LOGISTIC REGRESSION - SOCIAL NETWORK ADS PREDICTION")
print("=" * 80)

# ===== STEP 1: LOAD AND EXPLORE DATA =====
print("\nSTEP 1: LOAD AND EXPLORE SOCIAL NETWORK ADS DATA")
print("-" * 80)

df = pd.read_csv('Social_Network_Ads.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 10 rows:")
print(df.head(10))

print(f"\nColumn Info:")
print(df.info())

print(f"\nTarget Variable Distribution:")
print(df['Purchased'].value_counts())
print(f"\nClass Distribution:")
print(df['Purchased'].value_counts(normalize=True) * 100)

# ===== STEP 2: DATA PREPROCESSING =====
print("\n" + "=" * 80)
print("STEP 2: DATA PREPROCESSING")
print("-" * 80)

# Drop ID column
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

print(f"\nFeature Statistics:")
print(X.describe())

# Check missing values
print(f"\nMissing values: {X.isnull().sum().sum()}")

# ===== STEP 3: NORMALIZE FEATURES =====
print("\n" + "=" * 80)
print("STEP 3: FEATURE NORMALIZATION")
print("-" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=['Age', 'EstimatedSalary'])

print(f"Before scaling - Mean: {X.mean().mean():.2f}, Std: {X.std().mean():.2f}")
print(f"After scaling - Mean: {X_scaled_df.mean().mean():.6f}, Std: {X_scaled_df.std().mean():.6f}")

# ===== STEP 4: TRAIN-TEST SPLIT =====
print("\n" + "=" * 80)
print("STEP 4: SPLIT DATA (80% TRAIN, 20% TEST)")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

print(f"\nTraining target distribution:")
print(y_train.value_counts())

print(f"\nTesting target distribution:")
print(y_test.value_counts())

# ===== STEP 5: CREATE AND TRAIN MODEL =====
print("\n" + "=" * 80)
print("STEP 5: CREATE AND TRAIN LOGISTIC REGRESSION MODEL")
print("-" * 80)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

print("Logistic Regression model trained!")

print(f"\nModel Parameters:")
print(f"  Intercept: {model.intercept_[0]:.6f}")
print(f"  Coefficients: Age={model.coef_[0][0]:.6f}, Salary={model.coef_[0][1]:.6f}")

print(f"\nModel equation:")
print(f"  P(Purchase) = 1 / (1 + e^(-({model.intercept_[0]:.4f} + {model.coef_[0][0]:.4f}×Age + {model.coef_[0][1]:.4f}×Salary)))")

# ===== STEP 6: MAKE PREDICTIONS =====
print("\n" + "=" * 80)
print("STEP 6: MAKE PREDICTIONS")
print("-" * 80)

# Binary predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Probability predictions
y_train_pred_proba = model.predict_proba(X_train)
y_test_pred_proba = model.predict_proba(X_test)

print(f"Training predictions shape: {y_train_pred.shape}")
print(f"Testing predictions shape: {y_test_pred.shape}")

print(f"\nFirst 10 Testing Predictions:")
test_comparison = pd.DataFrame({
    'Actual': y_test.iloc[:10].values,
    'Predicted': y_test_pred[:10],
    'P(No Purchase)': y_test_pred_proba[:10, 0].round(4),
    'P(Purchase)': y_test_pred_proba[:10, 1].round(4)
})
print(test_comparison)

# ===== STEP 7: CALCULATE CONFUSION MATRIX =====
print("\n" + "=" * 80)
print("STEP 7: CONFUSION MATRIX")
print("-" * 80)

# Training confusion matrix
cm_train = confusion_matrix(y_train, y_train_pred)
print(f"\nTraining Confusion Matrix:")
print(cm_train)
print(f"\nInterpretation (Training):")
print(f"  TN (Correct No Purchase): {cm_train[0,0]}")
print(f"  FP (False Positive): {cm_train[0,1]}")
print(f"  FN (False Negative): {cm_train[1,0]}")
print(f"  TP (Correct Purchase): {cm_train[1,1]}")

# Testing confusion matrix
cm_test = confusion_matrix(y_test, y_test_pred)
print(f"\nTesting Confusion Matrix:")
print(cm_test)
print(f"\nInterpretation (Testing):")
print(f"  TN (Correct No Purchase): {cm_test[0,0]}")
print(f"  FP (False Positive): {cm_test[0,1]}")
print(f"  FN (False Negative): {cm_test[1,0]}")
print(f"  TP (Correct Purchase): {cm_test[1,1]}")

# ===== STEP 8: CALCULATE PERFORMANCE METRICS =====
print("\n" + "=" * 80)
print("STEP 8: PERFORMANCE METRICS")
print("-" * 80)

# Training metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# Testing metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"\nMETRICS COMPARISON:")
print(f"{'Metric':<15} {'Training':<15} {'Testing':<15}")
print("-" * 45)
print(f"{'Accuracy':<15} {train_accuracy:<15.4f} {test_accuracy:<15.4f}")
print(f"{'Precision':<15} {train_precision:<15.4f} {test_precision:<15.4f}")
print(f"{'Recall':<15} {train_recall:<15.4f} {test_recall:<15.4f}")
print(f"{'F1-Score':<15} {train_f1:<15.4f} {test_f1:<15.4f}")

print(f"\nCLASSIFICATION REPORT (Testing Data):")
print(classification_report(y_test, y_test_pred, target_names=['No Purchase', 'Purchase']))

# ===== STEP 9: ROC-AUC CURVE =====
print("\n" + "=" * 80)
print("STEP 9: ROC-AUC ANALYSIS")
print("-" * 80)

fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba[:, 1])
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba[:, 1])

auc_train = auc(fpr_train, tpr_train)
auc_test = auc(fpr_test, tpr_test)

print(f"AUC-ROC (Training): {auc_train:.4f}")
print(f"AUC-ROC (Testing): {auc_test:.4f}")

# ===== STEP 10: VISUALIZATION =====
print("\n" + "=" * 80)
print("STEP 10: VISUALIZATION")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Confusion Matrix (Testing)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['No Purchase', 'Purchase'],
            yticklabels=['No Purchase', 'Purchase'])
axes[0, 0].set_title('Confusion Matrix - Testing Data', fontweight='bold')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')

# Plot 2: Precision, Recall, F1 Comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_vals = [train_accuracy, train_precision, train_recall, train_f1]
test_vals = [test_accuracy, test_precision, test_recall, test_f1]

x = np.arange(len(metrics))
width = 0.35
axes[0, 1].bar(x - width/2, train_vals, width, label='Training', alpha=0.8)
axes[0, 1].bar(x + width/2, test_vals, width, label='Testing', alpha=0.8)
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_title('Metric Comparison', fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(metrics, rotation=15)
axes[0, 1].legend()
axes[0, 1].set_ylim([0, 1.1])
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: ROC Curve
axes[1, 0].plot(fpr_train, tpr_train, 'b-', label=f'Training (AUC={auc_train:.3f})', linewidth=2)
axes[1, 0].plot(fpr_test, tpr_test, 'g-', label=f'Testing (AUC={auc_test:.3f})', linewidth=2)
axes[1, 0].plot([0, 1], [0, 1], 'r--', label='Random', linewidth=2)
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC-AUC Curve', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Probability Distribution
axes[1, 1].hist(y_test_pred_proba[y_test == 0, 1], bins=20, alpha=0.5, label='No Purchase', edgecolor='black')
axes[1, 1].hist(y_test_pred_proba[y_test == 1, 1], bins=20, alpha=0.5, label='Purchase', edgecolor='black')
axes[1, 1].axvline(x=0.5, color='r', linestyle='--', linewidth=2, label='Decision Boundary')
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Probability Distribution', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('logistic_regression_results.png', dpi=150, bbox_inches='tight')
print("Visualization saved as 'logistic_regression_results.png'")
plt.show()

# ===== STEP 11: SUMMARY =====
print("\n" + "=" * 80)
print("STEP 11: MODEL SUMMARY")
print("=" * 80)

print(f"\nMODEL PERFORMANCE:")
print(f"  Test Accuracy: {test_accuracy:.4f} → {test_accuracy*100:.2f}% correct predictions")
print(f"  Test Precision: {test_precision:.4f} → {test_precision*100:.2f}% of predicted purchases correct")
print(f"  Test Recall: {test_recall:.4f} → {test_recall*100:.2f}% of actual purchases found")
print(f"  Test F1-Score: {test_f1:.4f} → Balanced measure")
print(f"  AUC-ROC: {auc_test:.4f} → Discrimination ability")

print(f"\n✓ LOGISTIC REGRESSION PRACTICAL COMPLETED!")
print("=" * 80)
```

---

## KEY FORMULAS

### Sigmoid Function
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### Log Odds
$$\log\left(\frac{P}{1-P}\right) = \beta_0 + \beta_1 X$$

### Accuracy
$$\text{Accuracy} = \frac{TP + TN}{Total}$$

### Precision & Recall
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

### F1-Score
$$F1 = 2 \times \frac{P \times R}{P + R}$$

---

## IMPORTANT FUNCTIONS

| Function | Purpose |
|----------|---------|
| `LogisticRegression()` | Create model |
| `fit()` | Train model |
| `predict()` | Binary predictions |
| `predict_proba()` | Probability predictions |
| `confusion_matrix()` | Calculate CM |
| `accuracy_score()` | Calculate accuracy |
| `precision_score()` | Calculate precision |
| `recall_score()` | Calculate recall |
| `f1_score()` | Calculate F1 |
| `classification_report()` | Full report |
| `roc_curve()` | ROC curve data |
| `auc()` | Calculate AUC |

---

## EXTERNAL VIVA QUESTIONS (15 Key)

### Q1: What does logistic regression predict?
**A:** Probability of binary outcome (0/1), not continuous value like linear regression.

### Q2: Explain sigmoid function.
**A:** Transforms any value to [0,1]. S-shaped curve. At z=0, P=0.5.

### Q3: What's decision boundary?
**A:** Threshold (usually 0.5) - if P≥0.5, predict 1; if P<0.5, predict 0.

### Q4: Define TP, FP, TN, FN.
**A:** TP=correct 1, FP=wrong 1, TN=correct 0, FN=wrong 0.

### Q5: What is accuracy?
**A:** (TP+TN)/Total. Percentage of correct predictions.

### Q6: When is precision important?
**A:** When false positives costly (spam detection - don't flag legitimate emails).

### Q7: When is recall important?
**A:** When false negatives costly (disease detection - don't miss patients).

### Q8: What's relationship between precision and recall?
**A:** Usually inverse - increasing one decreases other. F1 balances both.

### Q9: Explain AUC-ROC.
**A:** Area under ROC curve. 0.5=random, 1.0=perfect, >0.7=good.

### Q10: Why normalize before logistic regression?
**A:** Features on different scales - normalization makes optimization stable.

### Q11: Can you use decision threshold other than 0.5?
**A:** Yes - adjust trade-off between precision/recall based on requirements.

### Q12: What if dataset imbalanced?
**A:** Use F1, precision, recall instead of accuracy. Consider class weights.

### Q13: How to handle multiclass classification?
**A:** One-vs-Rest approach - create multiple binary classifiers.

### Q14: What assumptions does logistic regression make?
**A:** (1) Binary outcome, (2) Linear relationship log-odds, (3) Independence.

### Q15: How does logistic regression differ from linear regression?
**A:** Linear: continuous output. Logistic: probability output (0-1).

---

**Ready for External Examination and Journal Submission**
