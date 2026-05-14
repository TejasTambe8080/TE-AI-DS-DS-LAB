# PRACTICAL 6: NAIVE BAYES CLASSIFICATION

## Practical Title
Naive Bayes Classifier - Probabilistic Classification on Iris Dataset

## Aim
To implement Naive Bayes classifier using Bayes' theorem for probabilistic classification on the Iris dataset.

## Objective
1. Understand Bayes' theorem and conditional probability
2. Implement GaussianNB classifier
3. Calculate posterior probabilities
4. Evaluate using confusion matrix and metrics
5. Compare with other classifiers

## Problem Statement
Classify Iris flowers into species (Setosa, Versicolor, Virginica) using sepal and petal measurements using Naive Bayes.

---

## THEORY

### 1. Bayes' Theorem
$$P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$$

Where:
- P(A|B) = Posterior: Probability of A given B
- P(B|A) = Likelihood: Probability of B given A
- P(A) = Prior: Initial probability of A
- P(B) = Evidence: Total probability of B

### 2. Naive Bayes Classification
For classification with features X₁, X₂, ..., Xₙ:
$$P(Y|X_1, X_2, ..., X_n) = \frac{P(X_1|Y) \times P(X_2|Y) \times ... \times P(X_n|Y) \times P(Y)}{P(X_1, X_2, ..., X_n)}$$

**Naive assumption:** Features are conditionally independent given class Y

### 3. GaussianNB Algorithm
Assumes each feature follows Gaussian (normal) distribution:
$$P(X_i|Y) = \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left(-\frac{(X_i - \mu_i)^2}{2\sigma_i^2}\right)$$

Where μᵢ and σᵢ are mean and std dev of feature i in class Y

### 4. Training: Calculate Statistics
For each class and feature:
1. Calculate mean (μ)
2. Calculate variance (σ²)

### 5. Prediction: Calculate Probabilities
For new sample:
1. Calculate P(feature|class) for each feature
2. Calculate P(class|features) = Product of P(feature|class) × P(class)
3. Predict class with highest probability

### 6. Advantages & Disadvantages

**Advantages:**
- Fast, simple, scalable
- Works well with small datasets
- Probabilistic (gives confidence)
- Good baseline

**Disadvantages:**
- Assumes feature independence (rarely true)
- Sensitive to feature scaling
- Zero frequency problem

---

## COMPLETE CODE

```python
# ===== PRACTICAL 6: NAIVE BAYES CLASSIFICATION =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

print("=" * 80)
print("PRACTICAL 6: NAIVE BAYES CLASSIFICATION - IRIS DATASET")
print("=" * 80)

# ===== STEP 1: LOAD DATA =====
print("\nSTEP 1: LOAD IRIS DATASET")
print("-" * 80)

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='Species')
species_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
y_names = y.map(species_names)

df = X.copy()
df['Species'] = y_names

print(f"Dataset shape: {df.shape}")
print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
print(f"\nClass Distribution:")
print(y_names.value_counts())

# ===== STEP 2: DATA PREPROCESSING =====
print("\n" + "=" * 80)
print("STEP 2: DATA PREPROCESSING")
print("-" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print(f"Features normalized: Yes")
print(f"Feature statistics (scaled):")
print(X_scaled_df.describe())

# ===== STEP 3: TRAIN-TEST SPLIT =====
print("\n" + "=" * 80)
print("STEP 3: SPLIT DATA (70% TRAIN, 30% TEST)")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"\nTraining class distribution:")
print(y_train.value_counts())

# ===== STEP 4: CREATE AND TRAIN MODEL =====
print("\n" + "=" * 80)
print("STEP 4: CREATE AND TRAIN NAIVE BAYES MODEL")
print("-" * 80)

model = GaussianNB()
model.fit(X_train, y_train)

print("Naive Bayes model trained!")

# Calculate class priors
print(f"\nClass Priors P(Y):")
for class_idx, class_name in species_names.items():
    prior = (y_train == class_idx).sum() / len(y_train)
    print(f"  P({class_name}) = {prior:.4f}")

# ===== STEP 5: ANALYZE MODEL PARAMETERS =====
print("\n" + "=" * 80)
print("STEP 5: MODEL PARAMETERS (MEAN AND VARIANCE)")
print("-" * 80)

feature_names = iris.feature_names
for class_idx, class_name in species_names.items():
    print(f"\n{class_name}:")
    for feat_idx, feat_name in enumerate(feature_names):
        mean = model.theta_[class_idx, feat_idx]
        var = model.var_[class_idx, feat_idx]
        print(f"  {feat_name}:")
        print(f"    Mean: {mean:.4f}, Variance: {var:.4f}")

# ===== STEP 6: MAKE PREDICTIONS =====
print("\n" + "=" * 80)
print("STEP 6: MAKE PREDICTIONS")
print("-" * 80)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

y_train_pred_proba = model.predict_proba(X_train)
y_test_pred_proba = model.predict_proba(X_test)

print(f"Training predictions shape: {y_train_pred.shape}")
print(f"Testing predictions shape: {y_test_pred.shape}")

print(f"\nFirst 10 Testing Predictions:")
test_comp = pd.DataFrame({
    'Actual': y_test.iloc[:10].map(species_names).values,
    'Predicted': [species_names[p] for p in y_test_pred[:10]],
    'Setosa': y_test_pred_proba[:10, 0].round(3),
    'Versicolor': y_test_pred_proba[:10, 1].round(3),
    'Virginica': y_test_pred_proba[:10, 2].round(3)
})
print(test_comp)

# ===== STEP 7: CONFUSION MATRIX =====
print("\n" + "=" * 80)
print("STEP 7: CONFUSION MATRIX")
print("-" * 80)

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

print(f"\nTraining Confusion Matrix:")
print(cm_train)

print(f"\nTesting Confusion Matrix:")
print(cm_test)

# ===== STEP 8: PERFORMANCE METRICS =====
print("\n" + "=" * 80)
print("STEP 8: PERFORMANCE METRICS")
print("-" * 80)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

train_precision = precision_score(y_train, y_train_pred, average='weighted')
test_precision = precision_score(y_test, y_test_pred, average='weighted')

train_recall = recall_score(y_train, y_train_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')

train_f1 = f1_score(y_train, y_train_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"\nMETRICS COMPARISON:")
print(f"{'Metric':<15} {'Training':<15} {'Testing':<15}")
print("-" * 45)
print(f"{'Accuracy':<15} {train_accuracy:<15.4f} {test_accuracy:<15.4f}")
print(f"{'Precision':<15} {train_precision:<15.4f} {test_precision:<15.4f}")
print(f"{'Recall':<15} {train_recall:<15.4f} {test_recall:<15.4f}")
print(f"{'F1-Score':<15} {train_f1:<15.4f} {test_f1:<15.4f}")

print(f"\nCLASSIFICATION REPORT (Testing):")
print(classification_report(y_test, y_test_pred, 
                           target_names=list(species_names.values())))

# ===== STEP 9: VISUALIZATION =====
print("\n" + "=" * 80)
print("STEP 9: VISUALIZATION")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Confusion Matrix
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=list(species_names.values()),
            yticklabels=list(species_names.values()))
axes[0, 0].set_title('Confusion Matrix - Testing Data', fontweight='bold')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')

# Plot 2: Accuracy Comparison
models = ['Training', 'Testing']
accuracies = [train_accuracy, test_accuracy]
colors = ['blue', 'green']
axes[0, 1].bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Accuracy Comparison', fontweight='bold')
axes[0, 1].set_ylim([0, 1.1])
for i, v in enumerate(accuracies):
    axes[0, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Metrics Comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_vals = [train_accuracy, train_precision, train_recall, train_f1]
test_vals = [test_accuracy, test_precision, test_recall, test_f1]

x = np.arange(len(metrics))
width = 0.35
axes[1, 0].bar(x - width/2, train_vals, width, label='Training', alpha=0.8)
axes[1, 0].bar(x + width/2, test_vals, width, label='Testing', alpha=0.8)
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Metrics Comparison', fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(metrics, rotation=15)
axes[1, 0].legend()
axes[1, 0].set_ylim([0, 1.1])
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Class-wise Accuracy
class_names = list(species_names.values())
class_acc = []
for class_idx in range(3):
    mask = y_test == class_idx
    if mask.sum() > 0:
        acc = (y_test_pred[mask] == class_idx).sum() / mask.sum()
        class_acc.append(acc)
    else:
        class_acc.append(0)

axes[1, 1].bar(class_names, class_acc, color=['red', 'green', 'blue'], alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Class-wise Accuracy (Testing)', fontweight='bold')
axes[1, 1].set_ylim([0, 1.1])
for i, v in enumerate(class_acc):
    axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('naive_bayes_results.png', dpi=150, bbox_inches='tight')
print("Visualization saved as 'naive_bayes_results.png'")
plt.show()

# ===== STEP 10: SUMMARY =====
print("\n" + "=" * 80)
print("STEP 10: MODEL SUMMARY")
print("=" * 80)

print(f"\nNAIVE BAYES CLASSIFIER PERFORMANCE:")
print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Test Precision: {test_precision:.4f}")
print(f"  Test Recall: {test_recall:.4f}")
print(f"  Test F1-Score: {test_f1:.4f}")

print(f"\nMODEL CHARACTERISTICS:")
print(f"  Assumes feature independence: Yes (Naive assumption)")
print(f"  Distribution: Gaussian (normal)")
print(f"  Training time: Very fast")
print(f"  Interpretability: High (probability-based)")

print(f"\n✓ NAIVE BAYES PRACTICAL COMPLETED!")
print("=" * 80)
```

---

## KEY FORMULAS

### Bayes' Theorem
$$P(Y|X) = \frac{P(X|Y) \times P(Y)}{P(X)}$$

### Gaussian Probability
$$P(X_i|Y) = \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left(-\frac{(X_i - \mu_i)^2}{2\sigma_i^2}\right)$$

### Naive Bayes Classification
$$P(Y|X_1,...,X_n) \propto P(Y) \prod_{i=1}^{n} P(X_i|Y)$$

---

## IMPORTANT FUNCTIONS

| Function | Purpose |
|----------|---------|
| `GaussianNB()` | Create model |
| `fit()` | Train on data |
| `predict()` | Make predictions |
| `predict_proba()` | Get probabilities |
| `confusion_matrix()` | Calculate CM |
| `accuracy_score()` | Calculate accuracy |
| `classification_report()` | Full report |

---

## EXTERNAL VIVA QUESTIONS (10 Key)

### Q1: What does Naive Bayes assume?
**A:** Features are conditionally independent given the class (rarely true, but works).

### Q2: Explain Bayes' theorem in context of classification.
**A:** P(Class|Features) = P(Features|Class)×P(Class) / P(Features). Gives posterior probability of class.

### Q3: What is prior probability?
**A:** P(Y) - initial probability of class before seeing features. Calculated from training data.

### Q4: What is likelihood?
**A:** P(X|Y) - probability of features given class. Estimated from training data.

### Q5: Why is it called "Naive"?
**A:** Because it assumes feature independence, which is usually false but simplifies calculation.

### Q6: What distribution does GaussianNB assume?
**A:** Gaussian (normal) distribution for each feature in each class.

### Q7: How does training work in Naive Bayes?
**A:** Calculate mean and variance for each feature in each class from training data.

### Q8: How does prediction work?
**A:** For new sample, calculate P(feature|class) for each feature, multiply by P(class).

### Q9: Advantages of Naive Bayes?
**A:** Fast, simple, good baseline, works with small data, probabilistic.

### Q10: When to use Naive Bayes?
**A:** Baseline classifier, text classification, spam detection, when interpretability important.

---

**Ready for External Examination and Journal Submission**
