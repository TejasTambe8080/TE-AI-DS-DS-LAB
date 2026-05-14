# PRACTICAL 2: DATA WRANGLING II

## Practical Title
Data Wrangling II - Outlier Detection, Handling & Data Transformations

## Aim
To identify and handle outliers in datasets, apply data transformations to reduce skewness, and transform non-normal distributions into normal distributions.

## Objective
1. Understand what outliers are and their impact on models
2. Detect outliers using boxplot, IQR, and Z-score methods
3. Handle outliers using trimming, capping, and imputation
4. Identify and reduce skewness in data
5. Apply log transformations to normalize distributions
6. Document preprocessing decisions and their rationale

## Problem Statement
Real-world datasets (like student performance data) often contain outliers that can distort statistical analysis and reduce model accuracy. This practical focuses on identifying outliers in student performance metrics (Math, Reading, Writing scores), handling them appropriately, and transforming skewed distributions into normal ones using logarithmic and other transformations.

---

## THEORY

### 1. Outliers: Definition and Impact

**What are Outliers?**
Outliers are data points that lie far from the rest of observations - vastly larger or smaller than remaining values. They violate expected patterns.

**Why do they occur?**
- Variability in data (legitimate extremes)
- Measurement errors (faulty sensors)
- Human error (wrong data entry)
- Data transmission errors
- Genuine rare events (fraud, anomalies)

**How do they affect statistics?**
- Mean: Highly affected (pulls toward outliers)
- Median: Minimally affected (middle value)
- Std Dev: Increases (spread increases)
- Variance: Increases significantly
- Linear Regression: Pulls regression line toward outliers

**Example:** Dataset [15, 101, 18, 7, 13, 16, 11, 21, 5, 15, 10, 9]
- Without outlier 101: Mean=12.8, Median=13
- With outlier 101: Mean=23.5, Median=13
- Outlier increases mean by 83%!

### 2. Detecting Outliers

**Method 1: Boxplot**
- Quartile-based approach
- Q1 (25th percentile), Q2 (median), Q3 (75th percentile)
- IQR = Q3 - Q1
- Outliers: Values < Q1-1.5×IQR or > Q3+1.5×IQR
- Visual: Easy to spot outliers as dots beyond whiskers

**Method 2: Z-Score**
- Formula: Z = (X - mean) / std_deviation
- Interpretation: |Z| > 3 = strong outlier
- |Z| > 2.5 = moderate outlier
- Simple and statistically sound
- Assumes normal distribution

**Method 3: IQR Method**
- Lower Bound: Q1 - 1.5×IQR
- Upper Bound: Q3 + 1.5×IQR
- Outliers outside bounds
- More robust than Z-score for non-normal data

**Method 4: Isolation Forest**
- Unsupervised ML algorithm
- Creates random trees to isolate anomalies
- Good for multivariate outliers
- Less affected by dimensionality

### 3. Handling Outliers

**Method 1: Trimming (Deletion)**
- Remove outlier rows from dataset
- Best: <1% of data are outliers
- Advantage: Simple, removes problematic data
- Disadvantage: Lose information
- Risk: May remove legitimate extreme values

**Method 2: Capping (Winsorization)**
- Replace outliers with upper/lower bounds
- Formula: Replace values > Q3+1.5×IQR with Q3+1.5×IQR
- Advantage: Keeps data points, reduces extreme values
- Disadvantage: Creates artificial values

**Method 3: Transformation**
- Log transformation: X → log(X)
- Square root: X → √X
- Box-Cox transformation
- Reduces spread of extreme values
- Advantage: Preserves all data
- Disadvantage: Harder to interpret

**Method 4: Imputation**
- Replace with mean/median/mode
- Median for outlier-prone data
- Mean for normally distributed data

### 4. Data Skewness

**Definition:** Asymmetry in distribution - how much distribution deviates from symmetry.

**Positive Skew (Right Skew):**
- Tail on right side
- Mean > Median > Mode
- More extreme values on right
- Examples: Income, house prices, reaction time

**Negative Skew (Left Skew):**
- Tail on left side
- Mode > Median > Mean
- More extreme values on left
- Examples: Test scores (ceiling effect)

**Skewness Coefficient:**
- 0: Perfectly symmetric
- -0.5 to 0.5: Fairly symmetric
- -1 to -0.5 or 0.5 to 1: Moderately skewed
- < -1 or > 1: Highly skewed

### 5. Data Transformations

**Why Transform?**
- Convert non-normal to normal distribution
- Reduce skewness
- Stabilize variance
- Linearize relationships
- Improve model performance

**Common Transformations:**

a) **Log Transformation (Most Common)**
- Formula: X_transformed = log(X) or ln(X)
- Best for: Right-skewed data, exponential relationships
- Cannot be applied to: Zero or negative values
- Effect: Compresses large values, spreads small values

b) **Square Root Transformation**
- Formula: X_transformed = √X
- Weaker than log (less compression)
- Can handle zeros
- Good for: Moderately right-skewed data

c) **Box-Cox Transformation**
- Automatic optimal transformation
- Finds lambda that maximizes normality
- Formula: X^λ (varies lambda)
- Most sophisticated approach
- Computationally intensive

d) **Reciprocal Transformation**
- Formula: X_transformed = 1/X
- Strong effect, rarely used
- Reverses order
- Good for: Strongly right-skewed data

### 6. Real-Time Examples

**Example 1: Student Performance Dataset**
- Issue: Math scores [60-80], one student scored 150 (impossible)
- Detection: Boxplot shows dot beyond whisker
- Solution: Delete or cap at 100

**Example 2: Income Data**
- Issue: Income highly right-skewed (most earn $30-50K, few earn millions)
- Solution: Log transformation converts to approximately normal
- Benefit: Linear regression performs better

**Example 3: House Price Prediction**
- Issue: Prices range $100K-$5M, high variance
- Solution: Log transform prices
- Benefit: Predictions more accurate

---

## LIBRARIES USED

### 1. **pandas**
- **Functions:** DataFrame manipulation, describe(), boxplot()
- **Purpose:** Explore data, calculate statistics

### 2. **numpy**
- **Functions:** sqrt(), log(), exp(), where()
- **Purpose:** Mathematical transformations

### 3. **scipy.stats**
- **Functions:** boxcox(), skew(), kurtosis()
- **Purpose:** Statistical tests and transformations

### 4. **matplotlib & seaborn**
- **Functions:** hist(), boxplot(), distplot()
- **Purpose:** Visualize before/after transformations

### 5. **scikit-learn**
- **Functions:** Isolation Forest
- **Purpose:** Detect outliers using ML

---

## ALGORITHM

**Step 1:** Load student performance dataset

**Step 2:** Explore and calculate statistics

**Step 3:** Detect outliers using boxplot
- Visualize data distribution
- Identify values beyond whiskers

**Step 4:** Detect outliers using Z-score
- Calculate Z = (X - mean) / std
- Identify values with |Z| > 3

**Step 5:** Handle outliers
- Option A: Delete rows with outliers
- Option B: Cap outliers at Q3 + 1.5×IQR
- Option C: Replace with median

**Step 6:** Check and reduce skewness
- Calculate skewness coefficient
- If skewness > 0.5: Moderately right-skewed

**Step 7:** Apply log transformation
- X_transformed = np.log(X)
- Create new column with transformed values

**Step 8:** Apply alternative transformations
- Square root, Box-Cox for comparison

**Step 9:** Verify transformation
- Compare skewness before/after
- Visualize histograms
- Calculate new skewness coefficient

**Step 10:** Document decisions and rationale

---

## COMPLETE CODE

```python
# ===== PRACTICAL 2: DATA WRANGLING II =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import boxcox, skew

# ===== STEP 1: CREATE/LOAD STUDENT PERFORMANCE DATASET =====
print("=" * 70)
print("STEP 1: CREATE STUDENT PERFORMANCE DATASET")
print("=" * 70)

# Create sample dataset
np.random.seed(42)
data = {
    'Student_ID': range(1, 31),
    'Math_Score': np.random.randint(60, 85, 30),
    'Reading_Score': np.random.randint(75, 95, 30),
    'Writing_Score': np.random.randint(60, 80, 30),
    'Placement_Score': np.random.randint(75, 100, 30),
}

# Add some outliers intentionally
data['Math_Score'][5] = 150  # Outlier
data['Math_Score'][15] = 20  # Outlier
data['Reading_Score'][8] = 200  # Outlier

df = pd.DataFrame(data)
print("\nDataset created with outliers:")
print(df.head(10))

# ===== STEP 2: EXPLORE DATASET =====
print("\n" + "=" * 70)
print("STEP 2: DATASET EXPLORATION")
print("=" * 70)

print("\nDataset Shape:", df.shape)
print("\nDescriptive Statistics:")
print(df.describe())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# ===== STEP 3: DETECT OUTLIERS - BOXPLOT METHOD =====
print("\n" + "=" * 70)
print("STEP 3: OUTLIER DETECTION - BOXPLOT METHOD")
print("=" * 70)

# Function to detect outliers using IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    print(f"\nColumn: {column}")
    print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
    print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Outlier values: {outliers[column].values}")
    
    return outliers, lower_bound, upper_bound

# Detect outliers in each column
outliers_math, lower_m, upper_m = detect_outliers_iqr(df, 'Math_Score')
outliers_reading, lower_r, upper_r = detect_outliers_iqr(df, 'Reading_Score')

# Visualize using boxplot
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
df.boxplot(column='Math_Score', ax=axes[0])
axes[0].set_title('Math Score - Boxplot', fontweight='bold')
df.boxplot(column='Reading_Score', ax=axes[1])
axes[1].set_title('Reading Score - Boxplot', fontweight='bold')
df.boxplot(column='Writing_Score', ax=axes[2])
axes[2].set_title('Writing Score - Boxplot', fontweight='bold')
df.boxplot(column='Placement_Score', ax=axes[3])
axes[3].set_title('Placement Score - Boxplot', fontweight='bold')
plt.tight_layout()
plt.savefig('boxplot_outliers.png', dpi=150, bbox_inches='tight')
print("\nBoxplot saved as 'boxplot_outliers.png'")
plt.show()

# ===== STEP 4: DETECT OUTLIERS - Z-SCORE METHOD =====
print("\n" + "=" * 70)
print("STEP 4: OUTLIER DETECTION - Z-SCORE METHOD")
print("=" * 70)

# Function to detect outliers using Z-score
def detect_outliers_zscore(data, column, threshold=3):
    mean = data[column].mean()
    std = data[column].std()
    z_scores = np.abs((data[column] - mean) / std)
    
    outliers_index = np.where(z_scores > threshold)[0]
    outliers_data = data.iloc[outliers_index]
    
    print(f"\nColumn: {column}")
    print(f"Mean: {mean:.2f}, Std: {std:.2f}")
    print(f"Threshold (|Z| > {threshold})")
    print(f"Number of outliers: {len(outliers_data)}")
    print(f"Outlier values: {outliers_data[column].values}")
    print(f"Z-scores: {z_scores[outliers_index].values}")
    
    return outliers_data

outliers_z_math = detect_outliers_zscore(df, 'Math_Score', threshold=2)

# ===== STEP 5: HANDLE OUTLIERS =====
print("\n" + "=" * 70)
print("STEP 5: HANDLE OUTLIERS - THREE APPROACHES")
print("=" * 70)

# Approach 1: Deletion
print("\n--- Approach 1: DELETION ---")
df_no_outliers = df[(df['Math_Score'] >= lower_m) & (df['Math_Score'] <= upper_m)]
print(f"Original rows: {len(df)}, After deletion: {len(df_no_outliers)}")
print(f"Rows deleted: {len(df) - len(df_no_outliers)}")

# Approach 2: Capping (Winsorization)
print("\n--- Approach 2: CAPPING ---")
df_capped = df.copy()
df_capped['Math_Score'] = df_capped['Math_Score'].clip(lower=lower_m, upper=upper_m)
print(f"Min Math Score: {df_capped['Math_Score'].min()}")
print(f"Max Math Score: {df_capped['Math_Score'].max()}")

# Approach 3: Replace with median
print("\n--- Approach 3: REPLACE WITH MEDIAN ---")
df_median = df.copy()
median_math = df_median['Math_Score'].median()
outlier_mask = (df_median['Math_Score'] < lower_m) | (df_median['Math_Score'] > upper_m)
df_median.loc[outlier_mask, 'Math_Score'] = median_math
print(f"Replaced {outlier_mask.sum()} outliers with median: {median_math}")

print("\nComparison of handling methods:")
print(f"Original Math Mean: {df['Math_Score'].mean():.2f}")
print(f"After Deletion Mean: {df_no_outliers['Math_Score'].mean():.2f}")
print(f"After Capping Mean: {df_capped['Math_Score'].mean():.2f}")
print(f"After Median Replace Mean: {df_median['Math_Score'].mean():.2f}")

# ===== STEP 6: CHECK SKEWNESS =====
print("\n" + "=" * 70)
print("STEP 6: ANALYZE SKEWNESS")
print("=" * 70)

def analyze_skewness(data, column):
    skewness = skew(data[column])
    mean = data[column].mean()
    median = data[column].median()
    mode = data[column].mode()[0] if len(data[column].mode()) > 0 else None
    
    print(f"\nColumn: {column}")
    print(f"Skewness: {skewness:.4f}")
    print(f"Mean: {mean:.2f}, Median: {median:.2f}, Mode: {mode}")
    
    if skewness > 0.5:
        print("→ Positively (right) skewed - has longer tail on right")
    elif skewness < -0.5:
        print("→ Negatively (left) skewed - has longer tail on left")
    else:
        print("→ Approximately symmetric")
    
    return skewness

print("\nSkewness Analysis - Original Data:")
skew_original_math = analyze_skewness(df, 'Math_Score')
analyze_skewness(df, 'Reading_Score')
analyze_skewness(df, 'Writing_Score')
analyze_skewness(df, 'Placement_Score')

# ===== STEP 7: APPLY TRANSFORMATIONS =====
print("\n" + "=" * 70)
print("STEP 7: DATA TRANSFORMATIONS")
print("=" * 70)

# Create DataFrame for transformations
df_transform = df_capped.copy()

# Log Transformation (only for positive values)
print("\n--- Log Transformation ---")
# Shift values to be positive if needed
min_val = df_transform['Math_Score'].min()
if min_val <= 0:
    shift = abs(min_val) + 1
    df_transform['Math_Score_log'] = np.log(df_transform['Math_Score'] + shift)
    print(f"Shifted by {shift} to make values positive")
else:
    df_transform['Math_Score_log'] = np.log(df_transform['Math_Score'])

skew_log = skew(df_transform['Math_Score_log'])
print(f"Skewness after log transformation: {skew_log:.4f}")
print(f"Original skewness: {skew_original_math:.4f}")
print(f"Improvement: {abs(skew_original_math) - abs(skew_log):.4f}")

# Square Root Transformation
print("\n--- Square Root Transformation ---")
df_transform['Math_Score_sqrt'] = np.sqrt(df_transform['Math_Score'])
skew_sqrt = skew(df_transform['Math_Score_sqrt'])
print(f"Skewness after sqrt transformation: {skew_sqrt:.4f}")

# ===== STEP 8: BOX-COX TRANSFORMATION =====
print("\n" + "=" * 70)
print("STEP 8: BOX-COX TRANSFORMATION")
print("=" * 70)

# Ensure all values are positive
if (df_transform['Math_Score'] > 0).all():
    df_transform['Math_Score_boxcox'], lambda_param = boxcox(df_transform['Math_Score'])
    skew_boxcox = skew(df_transform['Math_Score_boxcox'])
    print(f"Optimal lambda: {lambda_param:.4f}")
    print(f"Skewness after Box-Cox: {skew_boxcox:.4f}")

# ===== STEP 9: VISUALIZATION - BEFORE AND AFTER =====
print("\n" + "=" * 70)
print("STEP 9: VISUALIZE TRANSFORMATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: Original and Capped
axes[0, 0].hist(df['Math_Score'], bins=10, color='red', alpha=0.7, edgecolor='black')
axes[0, 0].set_title(f'Original (Skew: {skew_original_math:.3f})', fontweight='bold')
axes[0, 0].set_xlabel('Math Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(df['Math_Score'].mean(), color='green', linestyle='--', label='Mean')
axes[0, 0].axvline(df['Math_Score'].median(), color='blue', linestyle='--', label='Median')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(df_capped['Math_Score'], bins=10, color='orange', alpha=0.7, edgecolor='black')
axes[0, 1].set_title(f'After Capping (Skew: {skew(df_capped["Math_Score"]):.3f})', fontweight='bold')
axes[0, 1].set_xlabel('Math Score')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].hist(df_transform['Math_Score_log'], bins=10, color='green', alpha=0.7, edgecolor='black')
axes[0, 2].set_title(f'Log Transform (Skew: {skew_log:.3f})', fontweight='bold')
axes[0, 2].set_xlabel('Log(Math Score)')
axes[0, 2].grid(True, alpha=0.3)

# Row 2: sqrt and Box-Cox and comparison
axes[1, 0].hist(df_transform['Math_Score_sqrt'], bins=10, color='purple', alpha=0.7, edgecolor='black')
axes[1, 0].set_title(f'Sqrt Transform (Skew: {skew_sqrt:.3f})', fontweight='bold')
axes[1, 0].set_xlabel('Sqrt(Math Score)')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(df_transform['Math_Score_boxcox'], bins=10, color='cyan', alpha=0.7, edgecolor='black')
axes[1, 1].set_title(f'Box-Cox Transform (Skew: {skew_boxcox:.3f})', fontweight='bold')
axes[1, 1].set_xlabel('Box-Cox(Math Score)')
axes[1, 1].grid(True, alpha=0.3)

# Summary comparison
transformations = ['Original', 'Capped', 'Log', 'Sqrt', 'Box-Cox']
skewness_values = [skew_original_math, skew(df_capped['Math_Score']), skew_log, skew_sqrt, skew_boxcox]
colors_bar = ['red', 'orange', 'green', 'purple', 'cyan']

axes[1, 2].bar(transformations, [abs(s) for s in skewness_values], color=colors_bar, alpha=0.7, edgecolor='black')
axes[1, 2].set_title('Skewness Comparison (Lower is Better)', fontweight='bold')
axes[1, 2].set_ylabel('|Skewness|')
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('transformations_comparison.png', dpi=150, bbox_inches='tight')
print("\nTransformation comparison saved as 'transformations_comparison.png'")
plt.show()

# ===== STEP 10: SUMMARY AND RECOMMENDATIONS =====
print("\n" + "=" * 70)
print("STEP 10: SUMMARY AND RECOMMENDATIONS")
print("=" * 70)

print("\nTransformation Results:")
print(f"{'Method':<15} {'Skewness':<15} {'Improvement':<15} {'Best For':<20}")
print("-" * 65)
print(f"{'Original':<15} {skew_original_math:<15.4f} {'---':<15} {'---':<20}")
print(f"{'Capped':<15} {skew(df_capped['Math_Score']):<15.4f} {abs(skew_original_math)-abs(skew(df_capped['Math_Score'])):<15.4f} {'Interpretability':<20}")
print(f"{'Log':<15} {skew_log:<15.4f} {abs(skew_original_math)-abs(skew_log):<15.4f} {'Economic data':<20}")
print(f"{'Sqrt':<15} {skew_sqrt:<15.4f} {abs(skew_original_math)-abs(skew_sqrt):<15.4f} {'Moderate skew':<20}")
print(f"{'Box-Cox':<15} {skew_boxcox:<15.4f} {abs(skew_original_math)-abs(skew_boxcox):<15.4f} {'Optimal result':<20}")

print("\n✓ Data Wrangling II Practical Completed!")

# Save final processed data
df_transform.to_csv('student_performance_processed.csv', index=False)
print("\nProcessed data saved as 'student_performance_processed.csv'")
```

---

## LINE BY LINE CODE EXPLANATION

See Practical 1 for explanation of imports and basic pandas functions.

### Outlier Detection - IQR Method
```python
Q1 = data[column].quantile(0.25)
Q3 = data[column].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```
**Explanation:** Calculate quartiles and IQR. Outliers are values outside [lower_bound, upper_bound].

### Z-Score Detection
```python
z_scores = np.abs((data[column] - mean) / std)
outliers_index = np.where(z_scores > threshold)[0]
```
**Explanation:** Calculate Z-score for each value. Values with |Z| > threshold are outliers.

### Capping Outliers
```python
df_capped['Math_Score'] = df_capped['Math_Score'].clip(lower=lower_m, upper=upper_m)
```
**Explanation:** .clip() limits values to range [lower_m, upper_m]. Values outside range are replaced with bounds.

### Skewness Calculation
```python
skewness = skew(data[column])
```
**Explanation:** Calculates skewness coefficient. Positive = right skew, Negative = left skew, 0 = symmetric.

### Log Transformation
```python
df_transform['Math_Score_log'] = np.log(df_transform['Math_Score'])
```
**Explanation:** Natural log of values. Compresses large values, spreads small values. Reduces right skew.

### Box-Cox Transformation
```python
df_transform['Math_Score_boxcox'], lambda_param = boxcox(df_transform['Math_Score'])
```
**Explanation:** Automatically finds optimal transformation parameter. Returns transformed data and lambda value used.

---

## DATASET EXPLANATION

### Student Performance Dataset
- **Samples:** 30 students
- **Features:** 5 columns
  - Math_Score: 20-150 (with outliers)
  - Reading_Score: 75-200 (with outliers)
  - Writing_Score: 60-80 (clean)
  - Placement_Score: 75-100 (clean)
- **Target:** None (exploratory analysis)

---

## OUTPUT EXPLANATION

### Boxplot Output
- Outliers shown as dots beyond whiskers
- Identifies extreme values visually
- Whiskers at Q1-1.5×IQR and Q3+1.5×IQR

### Z-Score Output
- |Z| values showing deviation from mean
- Values > 3 are strong outliers
- Shows statistical magnitude of deviation

### Skewness Results
- Positive skew: Mean > Median (tail right)
- Negative skew: Mean < Median (tail left)
- Lower absolute value = more symmetric

### Transformation Improvement
- Original: High skewness value
- After Log: Reduced skewness (closer to 0)
- Box-Cox: Maximum reduction (most optimal)

---

## REAL TIME APPLICATIONS

1. **E-commerce Delivery Time:** Outliers (wrong entries), Log transformation normalizes
2. **Medical Records:** Outlier patients with extreme values, IQR-based handling
3. **House Price Prediction:** Right-skewed distribution, log transformation improves model
4. **Stock Price Analysis:** Extreme fluctuations, capping for stability
5. **Student Grades:** Outlier scores, median imputation preserves distribution

---

## EXTERNAL VIVA QUESTIONS

### Q1: What is an outlier?
**A:** A data point that lies far from other observations, vastly larger or smaller than remaining values.

### Q2: How does IQR method calculate outlier bounds?
**A:** Lower Bound = Q1 - 1.5×IQR, Upper Bound = Q3 + 1.5×IQR. Values outside are outliers.

### Q3: Explain Z-score for outlier detection.
**A:** Z = (X - mean) / std. |Z| > 3 = outlier. Shows how many standard deviations from mean.

### Q4: When to use deletion vs capping for outliers?
**A:** Deletion when <1% outliers (acceptable loss). Capping preserves data but creates artificial values.

### Q5: What is right skew and give an example.
**A:** Right skew: tail on right, Mean > Median. Example: Income distribution (most earn average, few earn millions).

### Q6: What does log transformation do?
**A:** Converts exponential to linear, compresses large values, spreads small values, reduces right skew.

### Q7: Why normalize before applying Box-Cox?
**A:** Box-Cox requires positive values. Transform: X_new = X + |min| + 1 to ensure X > 0.

### Q8: How does Box-Cox find optimal transformation?
**A:** Tests different lambda values, finds one maximizing normality (highest likelihood). Automatic optimization.

### Q9: What's difference between Median and Mean imputation?
**A:** Mean affected by outliers, pulls toward extreme values. Median is middle value, resistant to outliers.

### Q10: Can you reverse log transformation?
**A:** Yes. If X_log = log(X), then X = e^(X_log). Called inverse transformation.

### Q11: What are the three approaches to handle outliers?
**A:** (1) Deletion - remove rows, (2) Capping - replace with bounds, (3) Transformation - reduce spread.

### Q12: Which outlier detection method works best for non-normal data?
**A:** IQR method - doesn't assume normal distribution. Z-score assumes normality.

### Q13: Explain Winsorization.
**A:** Replace values beyond Q3+1.5×IQR with Q3+1.5×IQR. Preserves data count but reduces extreme values.

### Q14: What does negative skewness indicate?
**A:** Tail on left side, Mode > Median > Mean. More extreme values on left (like ceiling effect in tests).

### Q15: When is transformation inappropriate?
**A:** When interpretation is critical (e.g., dollar amounts), deletion/capping better than log.

### Q16: How many outliers are acceptable?
**A:** Generally <5% of data. If >5%, investigate if they're genuine or errors.

### Q17: Compare Box-Cox with Square Root transformation.
**A:** Box-Cox: Optimal but complex, requires positive values. Sqrt: Simpler, weaker effect, less aggressive.

### Q18: What happens if you cap outliers but don't document?
**A:** Risk of bias, hard to reproduce, misrepresentation of data, invalid conclusions.

### Q19: Why is skewness important in regression?
**A:** Violates normality assumption, residuals become non-normal, confidence intervals invalid.

### Q20: How to choose between multiple transformation methods?
**A:** Compare skewness values - lower is better. Validate with histogram and Q-Q plots.

### Q21: Can negative values be log-transformed?
**A:** No. Need to shift first: log(X + |min| + 1) to make all values positive.

### Q22: What's difference between outlier and anomaly?
**A:** Outlier: statistical extreme. Anomaly: unexpected pattern (fraud, error). Overlapping concepts.

### Q23: Why remove outliers before normalizing?
**A:** Outliers affect min/max (MinMax) and std (Standardization). Removing them first prevents skewed scaling.

### Q24: Can you have multivariate outliers?
**A:** Yes. Single variable seems normal, but combination unusual. Detected by Isolation Forest or Mahalanobis distance.

### Q25: What's effect of outliers on correlation?
**A:** Can dramatically change or create spurious correlations. Single outlier can reverse correlation sign.

### Q26: Explain robust statistics.
**A:** Statistical methods resistant to outliers: Median, IQR, Interquartile range. Vs non-robust: Mean, Std.

### Q27: When to keep outliers despite detection?
**A:** If legitimate (fraud detection needs anomalies), domain importance, small effect on model.

### Q28: How Box-Cox works mathematically.
**A:** Tests X_λ (varies lambda), calculates log-likelihood for each, selects lambda maximizing likelihood.

### Q29: What's advantage of quantile-based (IQR) vs statistical (Z-score) outlier detection?
**A:** Quantile: distribution-free (no assumptions), robust. Statistical: assumes normal distribution.

### Q30: Summarize complete outlier handling workflow.
**A:** (1) Detect using visual/statistical methods, (2) Understand cause, (3) Choose handling strategy, (4) Document decision, (5) Validate results, (6) Compare before/after.

---

## CONCLUSION

Data Wrangling II addresses outliers and skewness:
- **Outlier detection** using boxplot, Z-score, IQR
- **Outlier handling** via deletion, capping, transformation
- **Skewness reduction** using log, sqrt, Box-Cox transformations
- **Data validation** before and after processing

Key insight: Proper handling improves model performance and ensures reliable analysis.

---

**Ready for External Examination and Journal Submission**
