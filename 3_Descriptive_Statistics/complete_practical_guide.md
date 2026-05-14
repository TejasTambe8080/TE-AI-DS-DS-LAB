# PRACTICAL 3: DESCRIPTIVE STATISTICS

## Practical Title
Descriptive Statistics - Measures of Central Tendency and Variability

## Aim
To analyze and demonstrate measures of central tendency (mean, median, mode) and dispersion (variance, standard deviation, percentiles) for statistical data analysis and decision-making.

## Objective
1. Calculate measures of central tendency (mean, median, mode)
2. Calculate measures of dispersion (std dev, variance, range, percentiles)
3. Group data by categorical variables and calculate statistics
4. Interpret statistical results for business decisions
5. Compare distributions across different species/groups

## Problem Statement
Given the Iris dataset with 150 samples across 3 species, perform descriptive statistical analysis to understand:
1. Summary statistics (mean, median, std) grouped by species
2. Statistical measures for each Iris species
3. Comparative analysis: How do measurements vary across species?
4. Percentiles and quartile analysis

---

## THEORY

### 1. Measures of Central Tendency

**Mean (Average)**
- Formula: Mean = Σ X / N
- Advantages: Uses all data, mathematically convenient
- Disadvantages: Affected by outliers, may not be actual value
- Use: Normal, symmetric distributions

**Median (Middle Value)**
- For odd N: Middle value when sorted
- For even N: Average of two middle values
- Advantages: Resistant to outliers, better for skewed data
- Disadvantages: Ignores extreme values
- Use: Skewed distributions, presence of outliers

**Mode (Most Frequent)**
- Most occurring value
- Advantages: Works for categorical data, resistant to outliers
- Disadvantages: May not exist, multiple modes possible
- Use: Categorical data, discrete distributions

### 2. Measures of Dispersion

**Variance (σ²)**
- Formula: Variance = Σ(X - Mean)² / N
- Measures spread of data
- Higher variance = more spread
- Units: Squared (hard to interpret)

**Standard Deviation (σ)**
- Formula: Std Dev = √Variance
- Square root of variance
- Same units as original data
- 68-95-99.7 rule (normal distribution)
  - 68% within ±1σ
  - 95% within ±2σ
  - 99.7% within ±3σ

**Range**
- Formula: Range = Max - Min
- Simple measure of spread
- Affected by outliers
- Doesn't show distribution pattern

**Percentiles**
- P25 (Q1): 25th percentile, 25% below
- P50 (Q2/Median): 50th percentile
- P75 (Q3): 75th percentile, 75% below
- IQR = Q3 - Q1: Interquartile range

### 3. Statistical Comparisons

**Coefficient of Variation (CV)**
- Formula: CV = (Std Dev / Mean) × 100%
- Standardized measure for comparing distributions
- Removes scale dependency
- Use: Compare spread across different units

**Skewness**
- Measures asymmetry
- Positive: Right tail, Mean > Median
- Negative: Left tail, Mean < Median
- Zero: Symmetric distribution

**Kurtosis**
- Measures tail heaviness
- High kurtosis: Heavy tails, peaked center
- Low kurtosis: Light tails, flat center

### 4. Grouped Statistics

**Why Group Data?**
- Identify patterns within categories
- Compare groups
- Segment analysis
- Business intelligence

---

## LIBRARIES USED

### 1. **pandas**
- **Functions:** describe(), groupby(), agg(), quantile()
- **Purpose:** Grouped analysis, statistical calculations

### 2. **numpy**
- **Functions:** mean(), median(), std(), percentile()
- **Purpose:** Numerical calculations

### 3. **scipy.stats**
- **Functions:** describe(), skew(), kurtosis()
- **Purpose:** Advanced statistical measures

### 4. **matplotlib/seaborn**
- **Purpose:** Visualize distributions by group

---

## ALGORITHM

**Step 1:** Load Iris dataset

**Step 2:** Calculate basic statistics (mean, median, std, min, max)

**Step 3:** Calculate advanced statistics (var, percentiles, skewness)

**Step 4:** Group data by Species

**Step 5:** For each species, calculate grouped statistics

**Step 6:** Create comparison table

**Step 7:** Visualize distributions by species

**Step 8:** Interpret results

---

## COMPLETE CODE

```python
# ===== PRACTICAL 3: DESCRIPTIVE STATISTICS =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import describe, skew, kurtosis

print("=" * 80)
print("PRACTICAL 3: DESCRIPTIVE STATISTICS - IRIS DATASET ANALYSIS")
print("=" * 80)

# ===== STEP 1: LOAD DATASET =====
print("\nSTEP 1: LOAD IRIS DATASET")
print("-" * 80)

from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = iris.target
df['Species_Name'] = df['Species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 10 rows:")
print(df.head(10))

# ===== STEP 2: BASIC STATISTICS =====
print("\n" + "=" * 80)
print("STEP 2: BASIC DESCRIPTIVE STATISTICS (FULL DATASET)")
print("=" * 80)

print("\nDataset Statistics (Using describe()):")
print(df.describe())

# Manual calculation
print("\n\nManual Calculation - Sepal Length:")
sepal_length = df['sepal length (cm)']
print(f"Count: {len(sepal_length)}")
print(f"Mean: {sepal_length.mean():.4f}")
print(f"Median: {sepal_length.median():.4f}")
print(f"Mode: {sepal_length.mode().values[0]:.4f}")
print(f"Std Dev: {sepal_length.std():.4f}")
print(f"Variance: {sepal_length.var():.4f}")
print(f"Min: {sepal_length.min():.4f}")
print(f"Max: {sepal_length.max():.4f}")
print(f"Range: {sepal_length.max() - sepal_length.min():.4f}")

# ===== STEP 3: ADVANCED STATISTICS =====
print("\n" + "=" * 80)
print("STEP 3: ADVANCED STATISTICAL MEASURES")
print("=" * 80)

print("\nPercentiles for Sepal Length:")
percentiles = [10, 25, 50, 75, 90]
for p in percentiles:
    val = np.percentile(sepal_length, p)
    print(f"  {p}th percentile: {val:.4f}")

print(f"\nQuartiles:")
print(f"  Q1 (25th): {sepal_length.quantile(0.25):.4f}")
print(f"  Q2 (50th/Median): {sepal_length.quantile(0.50):.4f}")
print(f"  Q3 (75th): {sepal_length.quantile(0.75):.4f}")
print(f"  IQR: {sepal_length.quantile(0.75) - sepal_length.quantile(0.25):.4f}")

print(f"\nSkewness: {skew(sepal_length):.4f}")
print(f"  → {'Right-skewed (positive)' if skew(sepal_length) > 0 else 'Left-skewed (negative)' if skew(sepal_length) < 0 else 'Symmetric'}")

print(f"\nKurtosis: {kurtosis(sepal_length):.4f}")
print(f"  → {'Heavy tails (leptokurtic)' if kurtosis(sepal_length) > 0 else 'Light tails (platykurtic)'}")

# Coefficient of Variation
cv = (sepal_length.std() / sepal_length.mean()) * 100
print(f"\nCoefficient of Variation: {cv:.2f}%")

# ===== STEP 4: GROUPED STATISTICS - BY SPECIES =====
print("\n" + "=" * 80)
print("STEP 4: GROUPED STATISTICS BY SPECIES")
print("=" * 80)

# Group by species and calculate statistics
grouped_stats = df.groupby('Species_Name').describe()
print("\nGrouped Descriptive Statistics:")
print(grouped_stats)

# Detailed grouped statistics
print("\n\nDETAILED ANALYSIS BY SPECIES:")
print("-" * 80)

for species in ['Setosa', 'Versicolor', 'Virginica']:
    print(f"\n{'='*40}")
    print(f"Species: {species}")
    print(f"{'='*40}")
    
    species_data = df[df['Species_Name'] == species]
    
    for col in iris.feature_names:
        col_data = species_data[col]
        print(f"\n{col}:")
        print(f"  Count: {len(col_data)}")
        print(f"  Mean: {col_data.mean():.4f}")
        print(f"  Median: {col_data.median():.4f}")
        print(f"  Std Dev: {col_data.std():.4f}")
        print(f"  Min: {col_data.min():.4f}, Max: {col_data.max():.4f}")
        print(f"  Q1: {col_data.quantile(0.25):.4f}, Q3: {col_data.quantile(0.75):.4f}")
        print(f"  Skewness: {skew(col_data):.4f}")

# ===== STEP 5: MEAN AND STD BY SPECIES AND FEATURE =====
print("\n" + "=" * 80)
print("STEP 5: MEAN AND STD DEVIATION BY SPECIES")
print("=" * 80)

# Create summary table
features = iris.feature_names
species_list = ['Setosa', 'Versicolor', 'Virginica']

mean_values = []
std_values = []

for species in species_list:
    species_data = df[df['Species_Name'] == species]
    means = [species_data[feat].mean() for feat in features]
    stds = [species_data[feat].std() for feat in features]
    mean_values.append(means)
    std_values.append(stds)

mean_df = pd.DataFrame(mean_values, columns=features, index=species_list)
std_df = pd.DataFrame(std_values, columns=features, index=species_list)

print("\nMean Values:")
print(mean_df.round(4))

print("\nStandard Deviation:")
print(std_df.round(4))

# ===== STEP 6: COEFFICIENT OF VARIATION =====
print("\n" + "=" * 80)
print("STEP 6: COEFFICIENT OF VARIATION (NORMALIZED SPREAD)")
print("=" * 80)

cv_values = []
for species in species_list:
    species_data = df[df['Species_Name'] == species]
    cv_row = []
    for feat in features:
        col_data = species_data[feat]
        cv = (col_data.std() / col_data.mean()) * 100
        cv_row.append(cv)
    cv_values.append(cv_row)

cv_df = pd.DataFrame(cv_values, columns=features, index=species_list)
print("\nCoefficient of Variation (%):")
print(cv_df.round(2))
print("\n→ Lower CV = More consistent, Higher CV = More variable")

# ===== STEP 7: VISUALIZATION =====
print("\n" + "=" * 80)
print("STEP 7: VISUALIZATION - DISTRIBUTIONS BY SPECIES")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

feature_cols = iris.feature_names

for idx, feat in enumerate(feature_cols):
    ax = axes[idx // 2, idx % 2]
    
    # Create boxplot
    species_groups = [df[df['Species_Name'] == s][feat].values for s in species_list]
    
    bp = ax.boxplot(species_groups, labels=species_list, patch_artist=True)
    
    # Color boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_title(f'{feat} by Species', fontweight='bold', fontsize=12)
    ax.set_ylabel(feat)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('iris_statistics_by_species.png', dpi=150, bbox_inches='tight')
print("\nBoxplot saved as 'iris_statistics_by_species.png'")
plt.show()

# ===== STEP 8: DISTRIBUTION PLOTS =====
print("\n" + "=" * 80)
print("STEP 8: HISTOGRAM AND DISTRIBUTION PLOTS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, feat in enumerate(feature_cols):
    ax = axes[idx // 2, idx % 2]
    
    for species, color in zip(species_list, ['blue', 'green', 'red']):
        data = df[df['Species_Name'] == species][feat]
        ax.hist(data, bins=10, alpha=0.5, label=species, color=color, edgecolor='black')
    
    ax.set_title(f'Distribution of {feat}', fontweight='bold', fontsize=12)
    ax.set_xlabel(feat)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('iris_distributions.png', dpi=150, bbox_inches='tight')
print("\nHistogram saved as 'iris_distributions.png'")
plt.show()

# ===== STEP 9: CORRELATION ANALYSIS =====
print("\n" + "=" * 80)
print("STEP 9: CORRELATION ANALYSIS")
print("=" * 80)

# Overall correlation
print("\nCorrelation Matrix (All Species Combined):")
corr_matrix = df[feature_cols].corr()
print(corr_matrix.round(4))

# Correlation by species
for species in species_list:
    print(f"\n{species} Correlation Matrix:")
    species_corr = df[df['Species_Name'] == species][feature_cols].corr()
    print(species_corr.round(4))

# ===== STEP 10: INTERPRETATION AND SUMMARY =====
print("\n" + "=" * 80)
print("STEP 10: INTERPRETATION AND SUMMARY")
print("=" * 80)

print("\nKEY FINDINGS:")
print("-" * 80)

# Compare means
print("\nSpecies Comparison (Mean Values):")
for feat in feature_cols[:2]:  # Show first 2 features
    print(f"\n{feat}:")
    values = {species: df[df['Species_Name'] == species][feat].mean() for species in species_list}
    max_species = max(values, key=values.get)
    min_species = min(values, key=values.get)
    print(f"  Highest: {max_species} ({values[max_species]:.4f})")
    print(f"  Lowest: {min_species} ({values[min_species]:.4f})")
    print(f"  Difference: {values[max_species] - values[min_species]:.4f}")

# Variability comparison
print("\n\nVariability Comparison (Std Dev):")
for feat in feature_cols[:2]:
    print(f"\n{feat}:")
    values = {species: df[df['Species_Name'] == species][feat].std() for species in species_list}
    most_var = max(values, key=values.get)
    least_var = min(values, key=values.get)
    print(f"  Most Variable: {most_var} ({values[most_var]:.4f})")
    print(f"  Least Variable: {least_var} ({values[least_var]:.4f})")

print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS PRACTICAL COMPLETED!")
print("=" * 80)

# Save results
mean_df.to_csv('iris_means.csv')
std_df.to_csv('iris_std_dev.csv')
cv_df.to_csv('iris_coefficient_variation.csv')
print("\nResults saved:")
print("  - iris_means.csv")
print("  - iris_std_dev.csv")
print("  - iris_coefficient_variation.csv")
```

---

## FORMULAS EXPLAINED

### Mean
$$\bar{X} = \frac{\sum X}{N}$$

### Median
- Odd N: Middle value when sorted
- Even N: Average of two middle values

### Variance
$$\sigma^2 = \frac{\sum(X - \bar{X})^2}{N}$$

### Standard Deviation
$$\sigma = \sqrt{\sigma^2}$$

### Coefficient of Variation
$$CV = \frac{\sigma}{\bar{X}} \times 100\%$$

### Percentile Position
$$Position = \frac{P}{100} \times (N+1)$$

---

## IMPORTANT FUNCTIONS

| Function | Purpose |
|----------|---------|
| `describe()` | Get all basic stats |
| `mean()` | Calculate average |
| `median()` | Get middle value |
| `std()` | Standard deviation |
| `var()` | Variance |
| `quantile(p)` | Get pth percentile |
| `groupby()` | Group by categories |
| `agg()` | Apply multiple functions |
| `skew()` | Calculate skewness |
| `kurtosis()` | Calculate kurtosis |

---

## EXTERNAL VIVA QUESTIONS (Select 30)

### Q1: Define mean and when to use it.
**A:** Mean = sum of values / count. Use for symmetric, normally distributed data without extreme outliers.

### Q2: When is median preferred over mean?
**A:** When data is skewed or has outliers - median is resistant (position-based) while mean is affected by extreme values.

### Q3: Can mode be used for continuous data?
**A:** Rarely - continuous data has many unique values. Mode useful for categorical or discrete data.

### Q4: Explain standard deviation conceptually.
**A:** Average distance of values from mean. Higher std = more spread, lower std = values clustered near mean.

### Q5: What does 68-95-99.7 rule mean?
**A:** In normal distribution: 68% within ±1σ, 95% within ±2σ, 99.7% within ±3σ.

### Q6: How do variance and std dev differ?
**A:** Variance is squared (hard to interpret), Std Dev is square root (same units as data, easier to interpret).

### Q7: What is coefficient of variation used for?
**A:** Compare spread across variables with different units. Standardized measure: CV = (Std/Mean)×100%.

### Q8: Explain quartiles and IQR.
**A:** Q1=25th percentile, Q2=median, Q3=75th percentile. IQR = Q3-Q1, middle 50% of data.

### Q9: What indicates right skew?
**A:** Mean > Median, tail extends right. Positive skewness coefficient. Example: income distribution.

### Q10: How to interpret kurtosis > 0?
**A:** Leptokurtic distribution - heavy tails, more peaked center. More extreme values than normal distribution.

### Q11: Why group data by categories?
**A:** Identify patterns within groups, compare groups, segment analysis, understand group-specific behavior.

### Q12: What is the relationship between mean, median, mode in symmetric distribution?
**A:** All three equal: Mean = Median = Mode. Distribution perfectly balanced.

### Q13: Can you have negative variance?
**A:** No, variance is always ≥ 0 (sum of squared deviations). Zero means all values identical.

### Q14: How many modes can dataset have?
**A:** Can have no mode, one mode (unimodal), or multiple modes (bimodal, multimodal).

### Q15: What does range not tell you about data?
**A:** Doesn't show distribution pattern, affected by outliers, ignores values between min and max.

### Q16: Explain 90th percentile practically.
**A:** 90% of data below this value, 10% above. Used for benchmarking, performance analysis.

### Q17: Why is grouped analysis important?
**A:** Reveals within-group patterns masked in overall data. Example: Iris setosa different from virginica.

### Q18: What's the effect of adding constant to all values?
**A:** Adds same constant to mean, median, doesn't affect std dev or variance (unchanged spread).

### Q19: How does outlier affect mean vs median?
**A:** Mean: Pulled dramatically toward outlier. Median: Minimal effect (position-based).

### Q20: Explain negative skew with example.
**A:** Left tail, Mean < Median. Example: Test scores with high ceiling (many students score near max).

### Q21: When to use CV instead of standard deviation?
**A:** Comparing variability across different units/scales. Makes std dev unitless for fair comparison.

### Q22: What does bimodal distribution suggest?
**A:** Two distinct groups in data. Example: Mixed age distribution (children + elderly peaks).

### Q23: How to detect outliers using quartiles?
**A:** Outliers: Values < Q1-1.5×IQR or > Q3+1.5×IQR. Same as boxplot whisker definition.

### Q24: Why divide by (N-1) for sample variance?
**A:** Bessel's correction - provides unbiased estimator of population variance (not N).

### Q25: What do whiskers represent in boxplot?
**A:** Extend from Q1-1.5×IQR to Q3+1.5×IQR. Show data range excluding outliers.

### Q26: How grouped statistics support decision-making?
**A:** Compare group performance, identify best/worst groups, target interventions, understand segments.

### Q27: Can median be useful in business?
**A:** Yes - median salary more realistic than mean (not pulled by extreme earners), median house price.

### Q28: What's the relationship between variance and correlation?
**A:** No direct relationship - variance measures spread, correlation measures relationship between variables.

### Q29: How does sample size affect standard deviation?
**A:** Doesn't directly affect std dev calculation, but larger samples give better estimates.

### Q30: Summarize how to describe a distribution completely.
**A:** Use: (1) Center: mean/median, (2) Spread: std/variance, (3) Shape: skewness/kurtosis, (4) Range: min/max.

---

## CONCLUSION

Descriptive Statistics provides foundation for data analysis:
- **Central tendency** measures where data centers
- **Dispersion** measures how spread out data is
- **Grouped analysis** reveals patterns within categories
- **Statistical insights** enable informed decision-making

This practical connects to all subsequent ML work - understanding data distribution is crucial before modeling.

---

**Ready for External Examination and Journal Submission**
