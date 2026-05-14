# PRACTICAL 1: DATA WRANGLING I

## Practical Title
Data Wrangling I - Missing Values, Preprocessing, Normalization & Label Encoding

## Aim
To learn and implement data preprocessing techniques including handling missing values, data normalization, and label encoding using Python and pandas library on real-world datasets.

## Objective
1. Understand the importance of data wrangling in machine learning
2. Identify and handle missing values using appropriate strategies
3. Apply data normalization techniques (Min-Max and Standardization)
4. Implement Label Encoding for categorical variables
5. Prepare datasets for machine learning model training
6. Understand pandas and scikit-learn preprocessing functions

## Problem Statement
Raw datasets from sources like Kaggle often contain missing values, inconsistent data types, and categorical variables. This practical focuses on cleaning and transforming the Iris dataset (and similar datasets) into a format suitable for machine learning models. Students will identify issues, handle missing values, normalize numerical features, and encode categorical variables.

---

## THEORY

### 1. What is Data Wrangling?
Data wrangling is the process of transforming raw, unstructured data into a clean, organized format suitable for analysis and machine learning. It includes:
- Data cleaning (handling errors, inconsistencies)
- Data transformation (converting formats)
- Data enrichment (adding useful information)
- Data reduction (removing redundant data)

**Why is it important?**
- 60-80% of data science work involves data wrangling
- Poor data quality leads to incorrect models
- Clean data is essential for reliable predictions
- Garbage in = Garbage out (GIGO principle)

### 2. Missing Values

**Definition:** Missing values (NaN/NA) are data points that are absent or not recorded in the dataset.

**Why do they occur?**
- Data collection errors
- Sensor failures
- Non-applicable values
- Intentional omission by respondents
- Data corruption during storage

**Impact:**
- Reduces dataset size
- Introduces bias
- Causes errors in calculations
- Reduces model accuracy
- Can lead to incorrect conclusions

**Types of Missing Data:**
1. **MCAR** (Missing Completely At Random): No pattern, safe to ignore
2. **MAR** (Missing At Random): Missing depends on other observed variables
3. **MNAR** (Missing Not At Random): Most problematic, depends on unobserved values

**Strategies to Handle Missing Values:**

a) **Deletion (Removal)**
   - Remove rows/columns with missing values
   - Best when: <5% data is missing
   - Advantages: Simple, removes problematic data
   - Disadvantages: Loss of information, reduces dataset size

b) **Imputation (Replacement)**
   - Mean: Sum/count, best for normal distributions
   - Median: Middle value, less affected by outliers
   - Mode: Most frequent value for categorical data
   - Forward/Backward Fill: For time series data
   - Interpolation: Estimate between known points
   
c) **Prediction**
   - Use ML models to predict missing values
   - More sophisticated but computationally expensive

### 3. Data Normalization

**Definition:** Normalization scales features to a specific range to ensure all features have similar scales.

**Why normalize?**
- Different features have different ranges
- Features with larger ranges dominate models
- Algorithms like KNN use distance metrics
- Gradient descent converges faster
- Prevents numerical instability

**Types of Normalization:**

a) **Min-Max Scaling (Normalization)**
```
Formula: X_normalized = (X - X_min) / (X_max - X_min)
Range: [0, 1]
```
- Preserves shape of distribution
- Sensitive to outliers
- Best for bounded data

b) **Z-Score Normalization (Standardization)**
```
Formula: X_standardized = (X - mean) / std_deviation
Range: [-3, 3] (typically)
```
- Less affected by outliers
- Produces mean=0, std=1
- Best for normal distributions

c) **Decimal Scaling**
```
Formula: X_scaled = X / 10^k
```
- Shifts decimal point

d) **Robust Scaling**
```
Formula: X_scaled = (X - median) / IQR
```
- Most resistant to outliers

### 4. Label Encoding

**Definition:** Converts categorical variables into numerical format by assigning unique integers to each category.

**Why encode?**
- ML algorithms work with numbers, not text
- Computers can't interpret "Red", "Green", "Blue"
- Encoding makes data machine-readable

**Types:**

a) **Label Encoding**
   - Assigns 0, 1, 2, ... to categories
   - Fast and simple
   - Problem: Creates implicit ordering

b) **One-Hot Encoding**
   - Creates binary columns for each category
   - No implicit ordering
   - Increases dimensionality

### 5. Real-Time Examples

**Example 1: E-commerce Dataset**
- Issue: Age 25, 30, 100 (outlier), 200-year-old person
- Missing gender values
- Solution: Normalize age with StandardScaler, fill missing gender with mode, remove outliers

**Example 2: Medical Data**
- Issue: Missing blood pressure readings, Age varies 5-100
- Blood type: "A", "B", "AB", "O"
- Solution: Fill BP with median of similar age group, normalize age, encode blood type

**Example 3: Stock Market Data**
- Issue: Missing daily prices, stock prices from $10-$1000
- Solution: Forward fill missing prices, normalize with StandardScaler

### 6. Advantages of Data Wrangling
✓ Improves model performance
✓ Reduces bias and variance
✓ Enables faster convergence
✓ Prevents runtime errors
✓ Facilitates data integration
✓ Ensures reliability

### 7. Disadvantages/Challenges
✗ Time-consuming (60-80% of project time)
✗ Information loss (deletion)
✗ Introduces assumptions (imputation)
✗ Requires domain knowledge
✗ Subjective decision-making
✗ Scalability issues with large datasets

### 8. Applications
- Machine learning model preprocessing
- Data analytics and reporting
- Business intelligence dashboards
- Financial analysis and fraud detection
- Healthcare patient data standardization
- Social media sentiment analysis

---

## LIBRARIES USED

### 1. **pandas**
- **Why:** Primary library for data manipulation and analysis
- **Functions used:** read_csv(), head(), info(), describe(), isnull(), fillna(), dropna()
- **Purpose:** Create DataFrames, explore data, handle missing values

### 2. **numpy**
- **Why:** Numerical computations and array operations
- **Functions used:** array(), mean(), std(), where()
- **Purpose:** Mathematical operations, array manipulation

### 3. **scikit-learn (sklearn)**
- **Why:** Machine learning preprocessing tools
- **Modules:** 
  - MinMaxScaler (Min-Max normalization)
  - StandardScaler (Z-score standardization)
  - LabelEncoder (Categorical encoding)
- **Purpose:** Feature scaling and encoding

### 4. **matplotlib**
- **Why:** Data visualization and plotting
- **Purpose:** Create graphs to visualize before/after transformations

### 5. **seaborn**
- **Why:** Statistical data visualization on top of matplotlib
- **Purpose:** Create attractive comparative plots

---

## ALGORITHM

**Step 1:** Import all required libraries (pandas, numpy, sklearn)

**Step 2:** Load the dataset using pd.read_csv()

**Step 3:** Explore the dataset:
- Check shape using df.shape
- View first few rows using df.head()
- Get info using df.info()
- Check for missing values using df.isnull().sum()

**Step 4:** Check for missing values:
- Use isnull() to identify NaN values
- Decide strategy: delete, impute, or predict

**Step 5:** Handle missing values:
- Option A: Drop rows (if <5% missing)
- Option B: Fill with mean/median
- Option C: Forward/backward fill (time series)

**Step 6:** Separate features (X) and target (y):
- X = df.iloc[:, :-1]  (all columns except last)
- y = df.iloc[:, -1]   (last column only)

**Step 7:** Apply Min-Max Normalization:
- Create MinMaxScaler object
- Fit and transform X using scaler.fit_transform(X)
- Features scaled to [0, 1]

**Step 8:** Apply Standardization:
- Create StandardScaler object
- Fit and transform X using scaler.fit_transform(X)
- Features scaled to mean=0, std=1

**Step 9:** Encode categorical target variable:
- Create LabelEncoder object
- Fit and transform target using encoder.fit_transform(y)

**Step 10:** Verify transformed data:
- Check min/max of normalized features
- Verify no NaN values remain
- Check mean/std of standardized features

**Step 11:** Save processed data to CSV for later use

**Step 12:** Create comparison visualization (before vs after)

---

## COMPLETE CODE

```python
# ===== PRACTICAL 1: DATA WRANGLING I =====

# STEP 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# STEP 2: Load the Iris dataset
# Method 1: From sklearn
from sklearn.datasets import load_iris
iris_data = load_iris()
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df['Species'] = iris_data.target

# Method 2: From CSV (if saved locally)
# df = pd.read_csv('iris.csv')

print("=" * 60)
print("DATASET EXPLORATION")
print("=" * 60)

# STEP 3: Explore the dataset
print("\nDataset Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nDataset Info:")
df.info()
print("\nDataset Statistics:\n", df.describe())

# STEP 4: Check for missing values
print("\n" + "=" * 60)
print("MISSING VALUES ANALYSIS")
print("=" * 60)
print("\nMissing values in each column:")
print(df.isnull().sum())
print("\nTotal missing values:", df.isnull().sum().sum())

# STEP 5: Handle missing values (if any exist)
# If missing values exist, uncomment the following:
# df = df.dropna()  # Option 1: Delete rows
# df.fillna(df.mean(), inplace=True)  # Option 2: Fill with mean

# STEP 6: Separate features (X) and target (y)
X = df.iloc[:, :-1]  # All columns except last (features)
y = df.iloc[:, -1]   # Last column (target)

print("\n" + "=" * 60)
print("FEATURE AND TARGET SEPARATION")
print("=" * 60)
print("\nFeature columns (X):", X.columns.tolist())
print("Target column (y): Species")
print("X shape:", X.shape)
print("y shape:", y.shape)

# STEP 7: Apply Min-Max Normalization (Scaling to [0,1])
print("\n" + "=" * 60)
print("MIN-MAX NORMALIZATION")
print("=" * 60)

minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)
X_minmax_df = pd.DataFrame(X_minmax, columns=X.columns)

print("\nOriginal X (first 5 rows):\n", X.head())
print("\nMin-Max Normalized X (first 5 rows):\n", X_minmax_df.head())
print("\nMin value in normalized X:", X_minmax_df.min().min())
print("Max value in normalized X:", X_minmax_df.max().max())

# STEP 8: Apply Standardization (Z-score normalization)
print("\n" + "=" * 60)
print("STANDARDIZATION (Z-SCORE NORMALIZATION)")
print("=" * 60)

standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X)
X_standard_df = pd.DataFrame(X_standard, columns=X.columns)

print("\nStandardized X (first 5 rows):\n", X_standard_df.head())
print("\nMean of standardized X:\n", X_standard_df.mean())
print("\nStandard Deviation of standardized X:\n", X_standard_df.std())

# STEP 9: Encode categorical target variable
print("\n" + "=" * 60)
print("LABEL ENCODING (TARGET VARIABLE)")
print("=" * 60)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded_df = pd.DataFrame({'Original': y, 'Encoded': y_encoded})

print("\nUnique classes before encoding:", label_encoder.classes_)
print("\nEncoding mapping:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"  {class_name} → {i}")

print("\nFirst 10 encoded values:", y_encoded[:10])

# STEP 10: Verify transformed data
print("\n" + "=" * 60)
print("VERIFICATION OF TRANSFORMED DATA")
print("=" * 60)

print("\nMin-Max Normalized Data:")
print("  Min:", X_minmax_df.min().min())
print("  Max:", X_minmax_df.max().max())
print("  Any NaN?", X_minmax_df.isnull().any().any())

print("\nStandardized Data:")
print("  Mean (approx 0):", X_standard_df.mean().mean())
print("  Std (approx 1):", X_standard_df.std().mean())
print("  Any NaN?", X_standard_df.isnull().any().any())

# STEP 11: Save processed data
print("\n" + "=" * 60)
print("SAVING PROCESSED DATA")
print("=" * 60)

X_minmax_df.to_csv('iris_normalized.csv', index=False)
X_standard_df.to_csv('iris_standardized.csv', index=False)
y_encoded_df.to_csv('iris_target_encoded.csv', index=False)
print("\nProcessed data saved successfully!")

# STEP 12: Visualization - Comparison of Normalization
print("\n" + "=" * 60)
print("CREATING COMPARISON VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Original Data
axes[0, 0].hist(X.iloc[:, 0], bins=20, color='blue', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Original Sepal Length', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Sepal Length (cm)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Min-Max Normalized
axes[0, 1].hist(X_minmax_df.iloc[:, 0], bins=20, color='green', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Min-Max Normalized Sepal Length', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Sepal Length (Normalized)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Standardized
axes[1, 0].hist(X_standard_df.iloc[:, 0], bins=20, color='orange', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Standardized Sepal Length', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Sepal Length (Standardized)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Side-by-side comparison
axes[1, 1].boxplot([X.iloc[:, 0], X_minmax_df.iloc[:, 0], X_standard_df.iloc[:, 0]],
                   labels=['Original', 'Min-Max', 'Standardized'])
axes[1, 1].set_title('Comparison of Normalization Methods', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Values')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_wrangling_comparison.png', dpi=150, bbox_inches='tight')
print("Comparison plot saved as 'data_wrangling_comparison.png'")
plt.show()

print("\n" + "=" * 60)
print("DATA WRANGLING PRACTICAL COMPLETED!")
print("=" * 60)
```

---

## LINE BY LINE CODE EXPLANATION

### Import Statements
```python
import pandas as pd
```
**Explanation:** Imports pandas library with alias 'pd'. Used for creating DataFrames and data manipulation.

```python
import numpy as np
```
**Explanation:** Imports numpy for numerical operations and array manipulations.

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
```
**Explanation:** Imports scaling and encoding classes from scikit-learn. MinMaxScaler for [0,1] scaling, StandardScaler for Z-score, LabelEncoder for categorical encoding.

```python
from sklearn.datasets import load_iris
```
**Explanation:** Imports the Iris dataset loader. Iris is a built-in dataset with 150 samples and 4 features.

### Data Loading
```python
iris_data = load_iris()
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df['Species'] = iris_data.target
```
**Explanation:** 
- load_iris() returns a dictionary-like object with 'data' and 'target' keys
- pd.DataFrame() creates a table from numpy array with proper column names
- Adds target variable (species) as the last column

### Data Exploration
```python
print("Dataset Shape:", df.shape)
```
**Explanation:** .shape returns tuple (rows, columns). Output: (150, 5)

```python
df.info()
```
**Explanation:** Displays column names, data types, and non-null counts. Helps identify missing values.

```python
df.describe()
```
**Explanation:** Provides statistical summary (mean, std, min, max, quartiles) for numeric columns.

### Missing Values Check
```python
print(df.isnull().sum())
```
**Explanation:** 
- .isnull() returns boolean True where value is NaN
- .sum() counts True values (total missing per column)
- Shows which columns have missing data

### Feature-Target Separation
```python
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
```
**Explanation:**
- .iloc[rows, columns] uses integer-based indexing
- [:, :-1] means all rows, all columns except last
- [:, -1] means all rows, last column only
- X contains features (independent variables)
- y contains target (dependent variable)

### Min-Max Normalization
```python
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)
```
**Explanation:**
- MinMaxScaler() creates a scaler object
- .fit_transform() learns min/max values and transforms data
- Formula: (X - min) / (max - min), scales to [0, 1]
- Returns numpy array (convert to DataFrame for readability)

```python
X_minmax_df = pd.DataFrame(X_minmax, columns=X.columns)
```
**Explanation:** Converts numpy array back to DataFrame with original column names for easier handling.

### Standardization
```python
standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X)
```
**Explanation:**
- StandardScaler() creates scaler object
- .fit_transform() learns mean/std and transforms data
- Formula: (X - mean) / std_deviation
- Result: mean ≈ 0, standard deviation ≈ 1
- Better for handling outliers than Min-Max

### Label Encoding
```python
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
```
**Explanation:**
- LabelEncoder() creates encoder object
- .fit() learns unique classes
- .transform() converts categories to numbers
- Example: ["setosa", "versicolor", "virginica"] → [0, 1, 2]

```python
print(label_encoder.classes_)
```
**Explanation:** Displays unique classes in sorted order. Shows mapping of labels.

### Verification
```python
print("Min:", X_minmax_df.min().min())
```
**Explanation:**
- .min().min() gets minimum of entire DataFrame (first .min() per column, second overall)
- For normalized data should be close to 0

```python
print("Mean:", X_standard_df.mean().mean())
```
**Explanation:** Should be close to 0 for properly standardized data.

### Saving Data
```python
X_minmax_df.to_csv('iris_normalized.csv', index=False)
```
**Explanation:** 
- .to_csv() saves DataFrame to CSV file
- index=False avoids writing row indices
- Saves preprocessed data for future use

### Visualization
```python
axes[0, 0].hist(X.iloc[:, 0], bins=20, color='blue', alpha=0.7, edgecolor='black')
```
**Explanation:**
- .hist() creates histogram
- X.iloc[:, 0] selects first column (Sepal Length)
- bins=20 creates 20 bars
- color='blue', alpha=0.7 for styling
- edgecolor='black' for visibility

---

## DATASET EXPLANATION

### Iris Dataset
- **Total samples:** 150 (50 per species)
- **Features:** 4 numerical
  1. **Sepal Length (cm):** 4.3 - 7.9
  2. **Sepal Width (cm):** 2.0 - 4.4
  3. **Petal Length (cm):** 1.0 - 6.9
  4. **Petal Width (cm):** 0.1 - 2.5
- **Target:** 3 species
  1. Iris-setosa
  2. Iris-versicolor
  3. Iris-virginica

### Data Characteristics
- **Data type:** Numerical (continuous)
- **Scale:** Different ranges for each feature
- **Missing values:** None (clean dataset)
- **Categorical:** Target variable only

---

## OUTPUT EXPLANATION

### Shape Output: (150, 5)
- 150 rows = 150 iris flowers
- 5 columns = 4 features + 1 target variable

### Missing Values Output: All zeros
- No null values in dataset
- Dataset is clean, no imputation needed

### Normalized Min-Max Output
```
Min value: 0.0
Max value: 1.0
```
- Successful scaling to [0, 1]
- All values between 0 and 1
- Relative distances preserved

### Standardized Output
```
Mean: ~0.0
Std: ~1.0
```
- Successful standardization
- Mean shifted to approximately 0
- Variance normalized to 1

### Encoded Target Output
```
Setosa → 0
Versicolor → 1
Virginica → 2
```
- Categorical classes converted to numbers
- Machine-readable format

---

## GRAPH EXPLANATION

### Plot 1: Original Sepal Length Distribution
- **Shape:** Slightly right-skewed
- **Range:** Approximately 4.3 to 7.9 cm
- **Peak:** Around 5.0-6.0 cm
- **Meaning:** Most iris flowers have sepal length in 5-6 cm range

### Plot 2: Min-Max Normalized Distribution
- **Shape:** Identical to original (shape preserved)
- **Range:** 0.0 to 1.0
- **Peak:** Around 0.3-0.4 (relative to original)
- **Meaning:** Same distribution, different scale

### Plot 3: Standardized Distribution
- **Shape:** Identical to original
- **Range:** Approximately -1.5 to +2.0
- **Center:** Around 0 (mean)
- **Meaning:** Centered at zero, spread by standard deviation

### Plot 4: Box Plot Comparison
- **Original:** Wide range (4-8)
- **Min-Max:** Compressed to 0-1
- **Standardized:** Centered at 0
- **Meaning:** Different scaling methods, same information

---

## FORMULA EXPLANATION

### Min-Max Normalization
$$X_{normalized} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

**Where:**
- X = original value
- X_min = minimum value in column
- X_max = maximum value in column
- Result: Range [0, 1]

**Example:** If age range is 20-80 and value is 50:
$$X_{normalized} = \frac{50 - 20}{80 - 20} = \frac{30}{60} = 0.5$$

### Standardization (Z-score)
$$X_{standardized} = \frac{X - \mu}{\sigma}$$

**Where:**
- X = original value
- μ = mean of column
- σ = standard deviation
- Result: Range ≈ [-3, +3]

**Example:** If age mean is 50, std is 10, and value is 60:
$$X_{standardized} = \frac{60 - 50}{10} = 1.0$$

---

## IMPORTANT FUNCTIONS USED

### pandas Functions

| Function | Purpose |
|----------|---------|
| `read_csv()` | Load CSV file into DataFrame |
| `head(n)` | Display first n rows |
| `shape` | Return (rows, columns) dimensions |
| `info()` | Display data types and info |
| `describe()` | Statistical summary |
| `isnull()` | Find missing values |
| `fillna()` | Fill missing values |
| `dropna()` | Remove missing values |
| `iloc[:]` | Integer-based indexing |
| `to_csv()` | Save to CSV file |

### sklearn Functions

| Function | Purpose |
|----------|---------|
| `MinMaxScaler()` | Create Min-Max scaler |
| `StandardScaler()` | Create Z-score scaler |
| `LabelEncoder()` | Create categorical encoder |
| `fit_transform()` | Learn parameters and transform |
| `fit()` | Learn parameters only |
| `transform()` | Apply learned parameters |

---

## COMMON ERRORS AND SOLUTIONS

### Error 1: "ModuleNotFoundError: No module named 'sklearn'"
**Solution:** Install scikit-learn
```bash
pip install scikit-learn
```

### Error 2: "ValueError: Input contains NaN, infinity or a value too large"
**Solution:** Handle missing values before scaling
```python
df = df.dropna()
# OR
df.fillna(df.mean(), inplace=True)
```

### Error 3: "KeyError: 'column_name'"
**Solution:** Column doesn't exist. Check column names:
```python
print(df.columns)
```

### Error 4: "TypeError: fit_transform() takes 1 positional argument but 2 were given"
**Solution:** fit_transform() doesn't work on Series, convert to DataFrame:
```python
X = df[['column']].values  # Convert to 2D array
X_scaled = scaler.fit_transform(X)
```

### Error 5: "AttributeError: 'numpy.ndarray' object has no attribute 'columns'"
**Solution:** fit_transform() returns numpy array, not DataFrame:
```python
X_scaled_df = pd.DataFrame(X_scaled, columns=original_columns)
```

---

## REAL TIME APPLICATIONS

### 1. E-commerce Recommendation System
- **Problem:** Product prices ($1-$10,000), ratings (1-5), reviews (1-100,000)
- **Solution:** Normalize all to [0,1] so no feature dominates
- **Benefit:** Algorithm treats all features equally

### 2. Medical Diagnosis System
- **Problem:** Age (0-100), blood pressure (80-200), cholesterol (100-300)
- **Solution:** Standardize all features
- **Benefit:** Neural networks converge faster

### 3. Student Performance Analysis
- **Problem:** Math (0-100), Science (0-100), English (0-100), Sports category (Beginner/Intermediate/Advanced)
- **Solution:** Normalize scores, encode sports category
- **Benefit:** Compare performance across subjects fairly

### 4. Credit Risk Assessment
- **Problem:** Income ($10,000-$500,000), Debt (varies), Credit score (300-850)
- **Solution:** Min-Max normalization to [0,1]
- **Benefit:** Features on same scale for decision trees

### 5. Stock Price Prediction
- **Problem:** Stock prices ($10-$1000+), Volume (thousands to millions)
- **Solution:** Standardize features
- **Benefit:** LSTM neural networks train better

### 6. Customer Segmentation
- **Problem:** Age, Income, Purchases vary widely
- **Solution:** Standardize before K-means clustering
- **Benefit:** Distance-based clustering works correctly

---

## DIFFERENCES BETWEEN IMPORTANT CONCEPTS

### Min-Max vs Standardization

| Aspect | Min-Max | Standardization |
|--------|---------|-----------------|
| Range | [0, 1] | [-3, +3] (unbounded) |
| Formula | (x-min)/(max-min) | (x-mean)/std |
| Outlier Effect | High (sensitive) | Low (resistant) |
| Distribution | Preserves shape | Preserves shape |
| Best For | Bounded data | Normal distribution |
| Algorithms | KNN, Neural Networks | Linear regression, SVM |

### Deletion vs Imputation

| Aspect | Deletion | Imputation |
|--------|----------|-----------|
| Data Loss | Yes (rows removed) | No (values estimated) |
| Best When | <5% missing | 5-30% missing |
| Complexity | Simple | More complex |
| Bias | Possible if MNAR | Reduces bias if MAR |
| Speed | Fast | Slower |

### Label Encoding vs One-Hot

| Aspect | Label | One-Hot |
|--------|-------|---------|
| Output | Single integer | Multiple binary columns |
| Ordering | Implicit | No ordering |
| Dimensionality | Same | Increased |
| Use Case | Tree models | Linear models |
| Memory | Low | High |

---

## EXTERNAL VIVA QUESTIONS AND ANSWERS

### Q1: What is data wrangling and why is it important?
**A:** Data wrangling is the process of transforming raw data into a clean, usable format. It's important because 60-80% of data science involves wrangling, and "garbage in = garbage out" - clean data is essential for accurate models.

### Q2: Name three types of missing data and explain MCAR.
**A:** Three types: MCAR (Missing Completely At Random - no pattern), MAR (Missing At Random - depends on other variables), MNAR (Missing Not At Random - depends on unobserved values). MCAR is safest to handle as it has no pattern.

### Q3: When would you use deletion vs imputation for missing values?
**A:** Use deletion when <5% of data is missing (acceptable information loss). Use imputation when 5-30% is missing (preserve data). If >30% missing, consider dropping the entire column.

### Q4: Explain Min-Max normalization with a formula and example.
**A:** Formula: X_normalized = (X - X_min) / (X_max - X_min), scales to [0,1]. Example: If age 25 with min=20, max=80: (25-20)/(80-20) = 5/60 = 0.083

### Q5: What is the difference between normalization and standardization?
**A:** Normalization (Min-Max) scales to [0,1] and is sensitive to outliers. Standardization (Z-score) centers at 0 with std=1 and is resistant to outliers.

### Q6: When would you use standardization over normalization?
**A:** Use standardization when: (1) data has outliers, (2) using algorithms like SVM or linear regression, (3) data approximately normally distributed. Use Min-Max for bounded features.

### Q7: What is label encoding and when should you use it?
**A:** Label encoding converts categorical values to integers (0, 1, 2...). Use it for: (1) target variables in classification, (2) tree-based models, (3) ordinal categories. Avoid for nominal categories in linear models.

### Q8: Why can't machine learning algorithms directly work with categorical data?
**A:** Because algorithms perform mathematical operations (multiplication, addition) on numerical values. Text like "Red", "Green" can't be used in equations, so encoding converts them to numbers.

### Q9: Explain the difference between fit() and transform() in sklearn.
**A:** fit() learns parameters (min, max, mean, std) from data. transform() applies those parameters. fit_transform() does both. Always fit on training data, then transform both train and test.

### Q10: What does MinMaxScaler.fit_transform() do in two steps?
**A:** Step 1: fit() finds min and max values of each column. Step 2: transform() applies formula (X-min)/(max-min) to scale all values to [0,1].

### Q11: What is StandardScaler and what are the resulting mean and std?
**A:** StandardScaler applies Z-score normalization: (X-mean)/std. Result: mean becomes ~0, standard deviation becomes ~1, allowing fair comparison of features.

### Q12: If you have 150 rows and apply fit_transform to X with shape (150, 4), what is output shape?
**A:** Output shape is (150, 4). Same as input - only values change, not dimensions.

### Q13: How do you convert a sklearn numpy array output back to DataFrame with column names?
**A:** Use `pd.DataFrame(array, columns=original_columns)` where original_columns is list of feature names.

### Q14: What is LabelEncoder.classes_ attribute used for?
**A:** It shows the mapping of encoded labels. For example, if classes_ = ['Setosa', 'Versicolor', 'Virginica'], then Setosa→0, Versicolor→1, Virginica→2.

### Q15: Explain why outliers affect Min-Max scaling more than standardization.
**A:** Min-Max uses min/max values - extreme outliers stretch the denominator, compressing all other values. Standardization uses mean/std - outliers have less dramatic effect on std.

### Q16: Can you apply normalization to only some features? How?
**A:** Yes, select specific columns: `scaler.fit_transform(df[['age', 'income']])` - this normalizes only age and income, not other columns.

### Q17: What happens if you fit scaler on training data but test data has values outside [0,1]?
**A:** Test values outside [0,1] indicate they're outside training range. Not necessarily wrong (extrapolation), but may indicate data shift or outliers in test set.

### Q18: How would you handle missing values in 'Income' column using mean imputation?
**A:** `df['Income'].fillna(df['Income'].mean(), inplace=True)` - replaces NaN with column's mean value.

### Q19: What's the difference between fillna() and dropna()?
**A:** fillna() replaces NaN with specified value (preserves rows). dropna() removes entire rows/columns with NaN (loses data). Choose based on how much data is missing.

### Q20: If you accidentally fit scaler on full data before splitting, what's the problem?
**A:** Data leakage! Test data influenced training parameters. Should always: (1) split data, (2) fit scaler on training only, (3) transform both using trained scaler.

### Q21: How do you prevent data leakage in preprocessing?
**A:** (1) Split data first, (2) fit all transformers on training set only, (3) apply same transformation to test set, (4) never fit on combined data.

### Q22: Why is pd.read_csv() first step before any analysis?
**A:** Because it loads raw data into DataFrame structure, enabling: (1) data exploration, (2) checking data types, (3) identifying issues, (4) planning preprocessing strategy.

### Q23: What does df.describe() NOT show?
**A:** It doesn't show: (1) missing values count, (2) data types, (3) categorical variables, (4) mode. Use df.info() and df.isnull().sum() for these.

### Q24: Explain why sklearn.preprocessing is preferred over manual normalization.
**A:** sklearn handles edge cases, prevents mistakes, works with multiple data types, provides inverse_transform(), easily integrates with pipelines, and is optimized for performance.

### Q25: If you normalize training data to [0,1], how do you normalize new single data point?
**A:** Use trained scaler: `scaler.transform([new_point])` - uses same min/max learned during training.

### Q26: What error occurs if you use transform() without fit()?
**A:** AttributeError - scaler not fitted. Always call fit_transform() or fit() before transform().

### Q27: Can you encode target variable before or after train-test split?
**A:** After split - encode separately. If before, risk data leakage (test data influences encoder).

### Q28: Why use LabelEncoder for target but OneHotEncoder for features in linear models?
**A:** LabelEncoder for target (naturally ordered - class 0, 1, 2). OneHotEncoder for features (avoids implicit ordering, prevents algorithm misinterpreting 0<1<2).

### Q29: If feature has zero variance (all values same), can you normalize it?
**A:** Technically yes, but pointless - already constant. May cause division by zero in Standardization. Better to drop zero-variance features first.

### Q30: What's best practice order for preprocessing pipeline?
**A:** (1) Handle missing values, (2) Remove outliers, (3) Feature engineering, (4) Encoding categorical, (5) Scaling/Normalization, (6) Feature selection.

---

## CONCLUSION

Data Wrangling I introduces the foundation of data preprocessing:
- **Missing value handling** ensures data completeness
- **Normalization/Standardization** ensures features on comparable scales
- **Label encoding** converts categorical to numerical format
- **Proper preprocessing** is essential for model success

**Key Takeaways:**
✓ 60-80% of data science is preprocessing
✓ Always explore data first
✓ Choose appropriate handling strategy for missing values
✓ Fit scalers on training data only (prevent data leakage)
✓ Encoding depends on data type and algorithm
✓ Proper preprocessing significantly improves model performance

This practical prepares you for Practical 2 (outlier handling and transformations) and all subsequent machine learning practicals.

---

**Ready for External Examination and Journal Submission**
DR. D.Y. PATIL INSTITUTE OF TECHNOLOGY, PUNE
DEPARTMENT

Lab

Academic
DR. D.Y. PATIL INSTITUTE OF TECHNOLOGY, PUNE
DEPARTMENT OF ARTIFICIAL INTELLIGENCE

SCIENCE

Lab Manual

Third Year Engineering
Semester-V
Software Laboratory III
Subject Code: 317534

Class : TEAI&DS

Academic Year 2024
DR. D.Y. PATIL INSTITUTE OF TECHNOLOGY, PUNE
INTELLIGENCE AND DATA

Manual

2024-25

Third Year of Artificial Intelligence and Data Science Software Laboratory III

3

Software Laboratory III
Subject Code: 317534

Guidelines for Instructor's Manual
The instructor‘s manual is to be developed as a reference and hands-on resource. It should include
prologue (about University/program/ institute/ department/foreword/ preface), curriculum of the
course, conduction and Assessment guidelines, topics under consideration, concept, objectives,
outcomes, set of typical applications/assignments/ guidelines, and references
Guidelines for Student Journal

The laboratory assignments are to be submitted by student in the form of journal. Journal consists of
Certificate, table of contents, and handwritten write-up of each assignment (Title, Date of Completion,
Objectives, Problem Statement, Software and Hardware requirements, Assessment grade/marks and
assessor's sign, Theory- Concept in brief, algorithm, flowchart, test cases, Test Data Set(if applicable),
mathematical model (if applicable), conclusion/analysis. Program codes with sample output of all
performed assignments are to be submitted as softcopy. As a conscious effort and little contribution
towards Green IT and environment awareness, attaching printed papers as part of write-ups and program
listing to journal must be avoided. Use of DVD containing students programs maintained by Laboratory
In-charge is highly encouraged. For reference one or two journals may be maintained with program
prints in the Laboratory.
Guidelines for Laboratory /Term Work Assessment
Continuous assessment of laboratory work should be based on overall performance of Laboratory
assignments by a student. Each Laboratory assignment assessment will assign grade/marks based on
parameters, such as timely completion, performance, innovation, efficient codes, and punctuality.

Guidelines for Practical Examination
Problem statements must be decided jointly by the internal examiner and external examiner. During
practical assessment, maximum weightage should be given to satisfactory implementation of the
problem statement. Relevant questions may be asked at the time of evaluation to test the student’s
Teaching Scheme Credit Examination Scheme
PR: 02 Hours/Week 02 PR: 25 Marks
TW: 50 Marks

Third Year of Artificial Intelligence and Data Science Software Laboratory III

4

understanding of the fundamentals, effective and efficient implementation. This will encourage,
transparent evaluation and fair approach, and hence will not create any uncertainty or doubt in the minds
of the students. So, adhering to these principles will consummate our team efforts to the promising start
of student's academics.
Guidelines for Laboratory Conduction
The instructor is expected to frame the assignments by understanding the prerequisites, technological
aspects, utility and recent trends related to the topic. The assignment framing policy need to address the
average students and inclusive of an element to attract and promote the intelligent students. Use of open
source software is encouraged. Based on the concepts learned. Instructor may also set one assignment
or mini-project that is suitable to AI & DS branch beyond the scope of the syllabus.

Third Year of Artificial Intelligence and Data Science Software Laboratory III

5

ASSIGNMENT 1

TITLE: DATA WRANGLING I

PROBLEM STATEMENT: -
Perform the following operations using Python on any open-source dataset (e.g., data.csv)
Import all the required Python Libraries.
1. Locate open-source data from the web (e.g. https://www.kaggle.com).
2. Provide a clear description of the data and its source (i.e., URL of the web site).
3. Load the Dataset into the pandas data frame.
4. Data Preprocessing: check for missing values in the data using pandas insult(), describe()
function to get some initial statistics. Provide variable descriptions. Types of variables etc.
Check the dimensions of the data frame.
5. Data Formatting and Data Normalization: Summarize the types of variables by checking the
data types (i.e., character, numeric, integer, factor, and logical) of the variables in the data set.
If variables are not in the correct data type, apply proper type conversions.
6. Turn categorical variables into quantitative variables in Python.
OBJECTIVE:
1. To Learn and understand the concepts of Python Libraries.
2. To learn and understand the Data Science for the analysis of real time problems
3. To understand and practice Data Preprocessing & Data Normalization.

PREREQUISITE: -
1 Basic of Python Programming
2 Concept of Data Preprocessing, Data Formatting, Data Normalization and Data Cleaning

THEORY:
1. Introduction to Big Data
Big data means really a big data, it is a collection of large datasets that cannot be processed using
traditional computing techniques. Big data is not merely a data, rather it has become a complete
subject, which involves various tools, techniques, and frameworks. Big data involves the data

Third Year of Artificial Intelligence and Data Science Software Laboratory III

6

produced by different devices and applications. Given below are some of the fields that come under
the umbrella of Big Data.

2. Introduction to Dataset
A dataset is a collection of records, similar to a relational database table. Records are similar to
table rows, but the columns can contain not only strings or numbers, but also nested data structures
such as lists, maps, and other records.
3. Python Libraries for Data Science
a. NumPy
One of the most fundamental packages in Python, NumPy is a general-purpose array-processing
package. It provides high-performance multidimensional array objects and tools to work with the
arrays. NumPy is an efficient container of generic multi-dimensional data. NumPy’s main object
is the homogeneous multidimensional array. It is a table of elements or numbers of the same
datatype, indexed by a tuple of positive integers. In NumPy, dimensions are called axes, and the
number of axes is called rank. NumPy’s array class is called ndarray aka array.
What can you do with NumPy?
1. Basic array operations: add, multiply, slice, flatten, reshape, index arrays
2. Advanced array operations: stack arrays, split into sections, broadcast arrays
3. Work with DateTime or Linear Algebra
4. Basic Slicing and Advanced Indexing in NumPy Python
b. Pandas
Pandas is an open-source Python package that provides high-performance, easy-to-use data
structures and data analysis tools for the labeled data in Python programming language.
What can you do with Pandas?
1. Indexing, manipulating, renaming, sorting, merging data frame
2. Update, Add, Delete columns from a data frame

Third Year of Artificial Intelligence and Data Science Software Laboratory III

7
3. Impute missing files; handle missing data or NANs
4. Plot data with histogram or box plot.
c. Scikit Learn
Introduced to the world as a Google Summer of Code project, Scikit Learn is a robust machine
learning library for Python. It features ML algorithms like SVMs, randomforests, k-means
clustering, spectral clustering, mean shift, cross-validation and more... Even NumPy, SciPy and
related scientific operations are supported by Scikit Learn with Scikit Learn being a part of the
SciPy Stack.
What can you do with Scikit Learn?
1. Classification: Spam detection, image recognition.
2. Clustering: Drug response, Stock price.
3. Regression: Customer segmentation, Grouping Assignment outcomes.
4. Dimensionality reduction: Visualization, Increased efficiency.
5. Model selection: Improved accuracy via parameter tuning.
6. Pre-processing: Preparing input data as a text for processing with machine learning algorithms.
4. Description of Dataset
The Iris dataset was used in R.A. Fisher's classic 1936 paper, The Use of Multiple
Measurements in Taxonomic Problems and can also be found on the UCI Machine Learning Repository.
It includes three iris species with 50 samples each as well as some properties about each flower. One
flower species is linearly separable from the other two, but the other two are not linearly separable from
each other.
Total Sample- 150
The columns in this dataset are:
1. Id
2. SepalLengthCm
3. SepalWidthCm
4. PetalLengthCm
5. PetalWidthCm
6. Species

Third Year of Artificial Intelligence and Data Science Software Laboratory III

8

Fig 1 Three Different Types of Species each contain 50 Sample
5. Panda Dataframe functions for Load Dataset
# The columns of the resulting DataFrame have different dtypes. iris.dtypes
1.The dataset is downloads from UCI repository.
csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
2. Now Read CSV File as a Dataframe in Python from from path where you saved the same The
Iris data set is stored in .csv format. ‘.csv’ stands for comma separated values. It is easier to load .csv
files in Pandas data frame and perform various analytical operations on it.
Load Iris.csv into a Pandas data frame —

Syntax-
iris = pd.read_csv(csv_url, header = None)

3.The csv file at the UCI repository does not contain the variable/column names. They are located
in a separate file.
col_names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species']
4.read in the dataset from the UCI Machine Learning Repository link and specify column names to
use
iris = pd.read_csv(csv_url, names = col_names)

Third Year of Artificial Intelligence and Data Science Software Laboratory III

9
Fig.2 Sample Dataset

6. Panda Data frame functions for Data Preprocessing:
Sr.
No

Data Frame Function Description

1 dataset.head(n=5) Return the first n rows.
2 dataset.tail(n=5) Return the last n rows.
3 dataset.index The index (row labels) of the Dataset.
4 dataset.columns The column labels of the Dataset.
5 dataset.shape Return a tuple representing the dimensionality of the

Dataset.

6 dataset.dtypes Return the dtypes in the Dataset.
7 dataset['Column name] Read the Data Column wise.
8 dataset.iloc[5] Purely integer-location based indexing for selection by

position.

9 dataset[0:3] Selecting via [], which slices the rows.
10 dataset.loc[:, ["Col_name1",
"col_name2"]]

Selection by label

11 dataset.iloc[:n, :] a subset of the first n rows of the original data
12 dataset.iloc[:, :n] a subset of the first n columns of the original data
13 dataset.iloc[:m, :n] a subset of the first m rows and the first n columns
Table 1. Panda Data frame functions for Data Preprocessing

Third Year of Artificial Intelligence and Data Science Software Laboratory III

10

Checking of Missing Values in Dataset:
● isnull() is the function that is used to check missing values or null values in pandas python.
● isna() function is also used to get the count of missing values of column and row wise count
of missing values
a. Is there any missing values in data frame as a whole.
Syntax: DataFrame.isnull()
b. Is there any missing values across each column.
Syntax: DataFrame.isnull().any()

c. count of missing values across each column using isna() and isnull()
In order to get the count of missing values of the entire dataframe isnull() function is used. sum() which does the
column wise sum first and doing another sum() will get the count of missing values of the entire dataframe.
Syntax: dataframe.isnull().sum().sum()
d. count row wise missing value using isnull()
Syntax: dataframe.isnull().sum(axis = 1)
7. Panda functions for Data Formatting and Normalization
The Transforming data stage is about converting the data set into a format that can be analyzed or modelled
effectively, and there are several techniques for this process.
A.Data Formatting: Ensuring all data formats are correct (e.g. object, text, floating number, integer, etc.) is
another part of this initial ‘cleaning’ process. If you are working with dates in Pandas, they also need to be stored
in the exact format to use special date-time functions.

Sr.
No
Data Frame
Function

Description Output

1. df.dtypes To check the data

type

Third Year of Artificial Intelligence and Data Science Software Laboratory III

11

B Data normalization:- Mapping all the nominal data values onto a uniform scale (e.g. from 0 to 1) is
involved in data normalization. Making the ranges consistent across variables helps with statistical
analysis and ensures better comparisonslater on.It is also known as Min-Max scaling.
Algorithm:
Step 1 : Import pandas and sklearn library for preprocessing
from sklearn import preprocessing
Step 2: Load the iris dataset in dataframe object df
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
Step 3: Print iris dataset.
df.head()
Step 4: Create x, where x the 'scores' column's values as floats
x = df[['score']].values.astype(float)
Step 5: Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()
Step 6: Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)
2. df['petal length
(cm)']= df['petal
length

To change the data
type (data type of
‘petal length
(cm)'changed to int)

(cm)'].astype("int
")

Third Year of Artificial Intelligence and Data Science Software Laboratory III

12
Step 7: Run the normalizer on the dataframe
df_normalized = pd.DataFrame(x_scaled)
Step 8: View the dataframe
df_normalized
8. Panda Functions for handling categorical variables
 Categorical variables have values that describe a ‘quality’ or ‘characteristic’of a data unit, like
‘what type’ or ‘which category’.
 Categorical variables fall into mutually exclusive (in one category or in another) and
exhaustive (include all possible options) categories. Therefore, categorical variables are
qualitative variables and tend to be represented by a non-numeric value.
 Categorical features refer to string type data and can be easily understood by human beings.
But in case of a machine, it cannot interpret the categorical data directly. Therefore, the
categorical data must be translated into numerical data that can be understood by machine.
 There are many ways to convert categorical data into numerical data. Here the most used
method is
Label Encoding: Label Encoding refers to converting the labels into a numeric form so as to
convert them into the machine-readable form. It is an important preprocessing step for the
structured dataset in supervised learning.
Example : Suppose we have a column Height in some dataset. After applying label
encoding, the Height column is converted into:

Where 0 is the label for tall, 1 is the label for medium, and 2 is a label for short height.
Label Encoding on iris dataset: For iris dataset the target column which is Species. It
contains three species Iris-setosa, Iris-versicolor, Iris-virginica.

Third Year of Artificial Intelligence and Data Science Software Laboratory III

13

Sklearn Functions for Label Encoding:
● preprocessing.LabelEncoder : It Encode labels with value between 0 and n_classes-1.
● fit_transform(y):
Parameters: yarray-like shape (n_samples,) Target values.
Returns: yarray-like of shape (n_samples,) Encoded labels.
This transformer should be used to encode target values, and not the input.
Algorithm:
Step 1 : Import pandas and sklearn library for preprocessing
from sklearn import preprocessing
Step 2: Load the iris dataset in dataframe object df
Step 3: Observe the unique values for the Species column.
df['Species'].unique()
output:array(['Iris-setosa','Iris-versicolor','Iris-virginica'], dtype=object)
Step 4: define label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
Step 5: Encode labels in column 'species'.
df['Species']= label_encoder.fit_transform(df['Species'])
Step 6: Observe the unique values for the Species column.
df['Species'].unique()
Output: array([0, 1, 2], dtype=int64)

CONCLUSION: In this way we have explored the functions of the python library for Data
Preprocessing, Data Wrangling Techniques and How to handle missing values on Iris Dataset.
ASSIGNMENT QUESTION
1. Explain Data Frame with Suitable example.
2. What is the limitation of the label encoding method?
3. What is the need of data normalization?
4. What are the different Techniques for Handling Missing Data?

Third Year of Artificial Intelligence and Data Science Software Laboratory III

14

ASSIGNMENT 2

TITLE: DATA WRANGLING II

PROBLEM STATEMENT: -
Create an “Academic performance” dataset of students and perform the following operations using Python.
1. Scan all variables for missing values and inconsistencies. If there are missing values and/or
inconsistencies, use any of the suitable techniques to deal with them.
2. Scan all numeric variables for outliers. If there are outliers, use any of the suitable techniques to deal
with them.
3. Apply data transformations on at least one of the variables. The purpose of this transformation should be
one of the following reasons: to change the scale for better understanding of the variable, to convert a
non-linear relation into a linear one, or to decrease the skewness and convert the distribution into a normal
distribution.
4. Reason and document your approach properly.
OBJECTIVE:
Students should be able to perform the data wrangling operation using Python on any open-source dataset.
PREREQUISITE: -
1. Basic of Python Programming
2. Concept of Data Preprocessing, Data Formatting, Data Normalization and Data Cleaning
THEORY:
1. Creation of Dataset using Microsoft Excel.
The dataset is created in “CSV” format.
● The name of dataset is Students Performance
● The features of the dataset are: Math_Score, Reading_Score, Writing_Score,
Placement_Score, Club_Join_Date .
● Number of Instances: 30
● The response variable is: Placement_Offer_Count .
● Range of Values:
Math_Score [60-80], Reading_Score[75-,95], ,Writing_Score [60,80],
Placement_Score[75-100], Club_Join_Date [2018-2021].
● The response variable is the number of placement offers facilitated to particular
students, which is largely depend on Placement_Score
To fill the values in the dataset the RANDBETWEEN is used. Returns a random integer number between
the numbers you specify.
Syntax: RANDBETWEEN (bottom, top)
Bottom The smallest integer and Top The largest integer RANDBETWEEN will return.

Third Year of Artificial Intelligence and Data Science Software Laboratory III

15
2 Identification and Handling of Null Values
Missing Data can occur when no information is provided for one or more items or for a whole
unit. Missing Data is a very big problem in real-life scenarios. Missing Data can also refer to as
NA(Not Available) values in pandas. In DataFrame sometimes many datasets simply arrive with
missing data, either because it exists and was not collected, or it never existed. For Example,
Suppose different users being surveyed may choose not to share their income, some users may
choose not to share the address in this way many datasets went missing.
In Pandas missing data is represented by two values:
1. None: None is a Python singleton object that is often used for missing data in Python code.
2. NaN: NaN (an acronym for Not a Number), is a special floating-point value recognized by all
systems that use the standard IEEE floating-point representation.
Pandas treat None and NaN as essentially interchangeable for indicating missing or null values.
To facilitate this convention, there are several useful functions fordetecting, removing, and
replacing null values in Pandas DataFrame :

● isnull()
● notnull()
● dropna()
● fillna()
● replace()

1. Checking for missing values using isnull() and notnull()
● Checking for missing values using isnull()

In order to check null values in Pandas DataFrame, isnull() function is used. This function
returns dataframe of Boolean values which are True for NaN values.
Algorithm:
Step 1 : Import pandas and numpy in order to check missing values in PandasDataFrame

import pandas as pd
import numpy as np
Step 2: Load the dataset in dataframe object df
df=pd.read_csv("/content/StudentsPerformanceTest1.csv")
Step 3: Display the data frame

df

Step 4: Use isnull() function to check null values in the dataset.

df.isnull()

Step 5:To create a series true for NaN values for specific columns. for examplemath score in
dataset and display data with only math score as NaN
series = pd.isnull(df["math score"])df[series]
● Checking for missing values using notnull()

Third Year of Artificial Intelligence and Data Science Software Laboratory III

16

In order to check null values in Pandas Dataframe, notnull() function is used. This
function returns dataframe of Boolean values which are False for NaN values.

1. Algorithm:
Step 1 : Import pandas and numpy in order to check missing values in
PandasDataFrame
import pandas as pd
import numpy as np
Step 2: Load the dataset in dataframe object df
df=pd.read_csv("/content/StudentsPerformanceTest1.csv")
Step 3: Display the data frame

df

Step 4: Use notnull() function to check null values in the dataset.
df.notnull()
Step 5: To create a series true for NaN values for specific columns. for
examplemath score in dataset and display data with only math score
as NaN
series1 = pd.notnull(df["math score"])df[series1]
See that there are also categorical values in the dataset, for this, you need to
useLabel Encoding
from sklearn.preprocessing import LabelEncoderle =
LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])newdf=df

1. Filling missing values using dropna(), fillna(), replace()
In order to fill null values in a datasets, fillna(), replace() functions are used. These functions
replace NaN values with some value of their own. All these functions help in filling null
values in datasets of a DataFrame.
For replacing null values with NaN
missing_values = ["Na", "na"]
df = pd.read_csv("StudentsPerformanceTest1.csv", na_values =missing_values)

df
● Filling null values with a single value

Step 1 : Import pandas and numpy in order to check missing values in
PandasDataFrame
import pandas as pd
import numpy as np

Third Year of Artificial Intelligence and Data Science Software Laboratory III

17

Step 2: Load the dataset in dataframe object df
df=pd.read_csv("/content/StudentsPerformanceTest1.csv")
Step 3: Display the data frame
df
Step 4: filling missing value using fillna()
ndf=df
ndf.fillna(0)
Step 5: filling missing values using mean, median and standard deviation of
thatcolumn.
data['math score'] = data['math score'].fillna(data['math score'].mean())
data[''math score''] = data[''math score''].fillna(data[''mathscore''].median())
data['math score''] = data[''math score''].fillna(data[''math score''].std())

replacing missing values in forenoon column with minimum/maximum numberof that column

data[''math score''] = data[''math score''].fillna(data[''math score''].min())
data[''math score''] = data[''math score''].fillna(data[''math score''].max())
● Filling null values in dataset

To fill null values in dataset use inplace=true
m_v=df['math score'].mean()
df['math score'].fillna(value=m_v, inplace=True)
df

● Filling a null values using replace() method

Following line will replace Nan value in dataframe with value -99
ndf.replace(to_replace = np.nan, value = -99)
● Deleting null values using dropna() method

In order to drop null values from a dataframe, dropna() function is used.
Thisfunction drops Rows/Columns of datasets with Null values in different
ways.
1. Dropping rows with at least 1 null value
2. Dropping rows if all values in that row are missing
3. Dropping columns with at least 1 null value.
4. Dropping Rows with at least 1 null value in CSV file

Algorithm:

Step 1 : Import pandas and numpy in order to check missing values in
PandasDataFrame

Third Year of Artificial Intelligence and Data Science Software Laboratory III

18

import pandas as pd
import numpy as np
Step 2: Load the dataset in dataframe object df
df=pd.read_csv("/content/StudentsPerformanceTest1.csv")
Step 3: Display the data frame

df

Step 4:To drop rows with at least 1 null value

ndf.dropna()

Step 5: To Drop rows if all values in that row are missing
ndf.dropna(how = 'all')
Step 6: To Drop columns with at least 1 null value.
ndf.dropna(axis = 1)
Step 7 : To drop rows with at least 1 null value in CSV
file.making new data frame with dropped NA values
new_data = ndf.dropna(axis = 0, how ='any')
new_data

3. Identification and Handling of Outliers
3.1 Identification of Outliers
One of the most important steps as part of data preprocessing is detecting and treating theoutliers as
they can negatively affect the statistical analysis and the training process of a machine learning
algorithm resulting in lower accuracy.
1. What are Outliers?
We all have heard of the idiom ‘odd one out' which means something unusual in comparison to the
others in a group.
Similarly, an Outlier is an observation in a given dataset that lies far from the rest of the observations.
That means an outlier is vastly larger or smaller than the remaining values in the set.
2. Why do they occur?
An outlier may occur due to the variability in the data, or due to Assignment al error/human error. They
may indicate an Assignment al error or heavy skewness in the data(heavy-tailed distribution).
3. What do they affect?
In statistics, we have three measures of central tendency namely Mean, Median, and Mode. They help us
describe the data.
Mean is the accurate measure to describe the data when we do not have any outliers present. Median is
used if there is an outlier in the dataset. Mode is used if thereis an outlier AND about 1⁄2 or more of the
data is the same.

Third Year of Artificial Intelligence and Data Science Software Laboratory III

19

‘Mean’ is the only measure of central tendency that is affected by the outliers which in turn impacts
Standard deviation.
Example:
Consider a small dataset, sample= [15, 101, 18, 7, 13, 16, 11, 21, 5, 15, 10, 9]. Bylooking at it, one
can quickly say ‘101’ is an outlier that is much larger than the other values.

Fig.1 Computation with and without outlier
From the above calculations, we can clearly say the Mean is more affected than the Median.
4. Detecting Outliers
If our dataset is small, we can detect the outlier by just looking at the dataset. But what
if we have a huge dataset, how do we identify the outliers then? We need to use visualization
and mathematical techniques.
Below are some of the techniques of detecting outliers
● Boxplots
● Scatterplots
● Z-score
● Inter Quantile Range(IQR)
Detecting outliers using Boxplot:
It captures the summary of the data effectively and efficiently with only a simple box and
whiskers. Boxplot summarizes sample data using 25th, 50th, and 75th percentiles. One can just
get insights(quartiles, median, and outliers) into the dataset by just looking at its boxplot.
2. Algorithm:

Step 1 : Import pandas and numpy libraries
import pandas as pd
import numpy as np
Step 2: Load the dataset in dataframe object df
df=pd.read_csv("/content/demo.csv")
Step 3: Display the data frame

Third Year of Artificial Intelligence and Data Science Software Laboratory III

20

df

Step 4:Select the columns for boxplot and draw the boxplot.

col = ['math score', 'reading score' , 'writingscore','placement
score']
df.boxplot(col)

Step 5: We can now print the outliers for each column with reference to the box plot.
print(np.where(df['math score']>90))
print(np.where(df['reading score']<25))
print(np.where(df['writing score']<30))

Handling of Outliers:

For removing the outlier, one must follow the same process of removing an
entry from the dataset using its exact position in the dataset because in all the above
methods ofdetecting the outliers end result is the list of all those data items that satisfy
the outlier definition according to the method used.
Below are some of the methods of treating the outliers
● Trimming/removing the outlier

Third Year of Artificial Intelligence and Data Science Software Laboratory III

21
● Quantile based flooring and capping
● Mean/Median imputation

4. Data Transformation for the purpose of:
Data transformation is the process of converting raw data into a format or structure that would
be more suitable for model building and also data discovery in general.The processof data
transformation can also be referred to as extract/transform/load (ETL). Theextraction phase
involves identifying and pulling data from the various source systems that create data and then
moving the data to a single repository. Next, the raw data is cleansed, if needed. It's then
transformed into a target format that can be fed into operational systems or into a data
warehouse, a date lake or another repository for use in business intelligence and analytics
applications. The transformation The data are transformed in ways that are ideal for mining the
data. The data transformation involves steps that are.
● Smoothing: It is a process that is used to remove noise from the dataset using
some algorithms It allows for highlighting important features present in the
dataset. It helps in predicting the patterns
● Aggregation: Data collection or aggregation is the method of storing and
presenting data in a summary format. The data may be obtained from multiple data
sources to integrate these data sources into a data analysis description. This is a
crucial step since the accuracy of data analysis insights is highly dependent on the
quantity and quality of the data used.
● Generalization:It converts low-level data attributes to high-level data attributes
using concept hierarchy. For Example Age initially in Numerical form (22, 25) is
converted into categorical value (young, old).
● Normalization: Data normalization involves converting all data variables into a
given range. Some of the techniques that are used for accomplishing normalization
are:
○ Min–max normalization: This transforms the original data linearly.
○ Z-score normalization: In z-score normalization (or zero-mean normalization)
the values of an attribute (A), are normalized based on the mean of A and its
standard deviation.
○ Normalization by decimal scaling: It normalizes the values
of an attribute by changing the position of their decimal points
● Attribute or feature construction.
○ New attributes constructed from the given ones: Where new attributes
are created & applied to assist the mining process from the given set of attributes.
This simplifies the original data & makes the mining more efficient.
In this assignment , The purpose of this transformation should be one of the following
reasons:

a. To change the scale for better understanding (Attribute or feature
construction)

Third Year of Artificial Intelligence and Data Science Software Laboratory III

22

Here the Club_Join_Date is transferred to Duration.

Algorithm:

Step 1 : Import pandas and numpy libraries
import pandas as pd
import numpy as np
Step 2: Load the dataset in dataframe object df
df=pd.read_csv("/content/demo.csv")
Step 3: Display the data frame
Step 3: Change the scale of Joining year to duration.

b. To decrease the skewness and convert distribution into normal distribution
(Normalization by decimal scaling)
Data Skewness: It is asymmetry in a statistical distribution, in which the curve appears distorted
or skewed either to the left or to the right. Skewness can be quantified to define the extent to which
a distribution differs from a normal distribution.
Normal Distribution: In a normal distribution, the graph appears as a classical, symmetrical
“bell-shaped curve.” The mean, or average, and the mode, or maximum point on the curve, are
equal.

Third Year of Artificial Intelligence and Data Science Software Laboratory III

23
Positively Skewed Distribution
A positively skewed distribution means that the extreme data results are larger. This skews the
data in that it brings the mean (average) up. The mean will be larger than the median in a Positively
skewed distribution.
A negatively skewed distribution means the opposite: that the extreme data results are smaller.
This means that the mean is brought down, and the median is larger than the mean in a negatively
skewed distribution.

Reducing skewness A data transformation may be used to reduce skewness. A distribution that
is symmetric or nearly so is often easier to handle and interpret than a skewed distribution. The
logarithm, x to log base 10 of x, or x to log base eof x (ln x), or x to log base 2 of x, is a strong
transformation with a major effecton distribution shape. It is commonly used for reducing right
skewness and is often appropriate for measured variables. It can not be applied to zero or negative
values.

1. Algorithm:

Step 1 : Detecting outliers using Z-Score for the Math_score variable

andremove the outliers.

Third Year of Artificial Intelligence and Data Science Software Laboratory III

24

Step 2: Observe the histogram for math_score variable.
import matplotlib.pyplot as plt new_df['math
score'].plot(kind = 'hist')

Step 3: Convert the variables to logarithm at the scale 10.

df['log_math'] = np.log10(df['math score'])
Step 4: Observe the histogram for math_score variable.
df['log_math'].plot(kind = 'hist')

It is observed that skewness is reduced at some level.

Conclusion: In this way we have explored the functions of the python library for Data
Identifying and handling the outliers. Data Transformations Techniques are explored with the
purpose of creating the new variable and reducing the skewness from datasets.
Assignment Question:
1. Explain the methods to detect the outlier.
2. Explain data transformation methods.
3. Write the algorithm to display the statistics of Null values present in the dataset.
4. Write an algorithm to replace the outlier value with the mean of the variable.
.

Third Year of Artificial Intelligence and Data Science Software Laboratory III

25
ASSIGNMENT 3

TITLE: DESCRIPTIVE STATISTICS- MEASURES OF CENTRAL TENDENCY AND

VARIABILITY

PROBLEM STATEMENT: -
Perform the following operations on any open-source dataset (e.g., data.csv)
1. Provide summary statistics (mean, median, minimum, maximum, standard deviation) for a dataset
(age, income etc.) with numeric variables grouped by one of the qualitative (categorical) variable.
For example, if your categorical variable is age groups and quantitative variable is income, then
provide summary statistics of income grouped bythe age groups. Create a list that contains a
numeric value for each response to the categorical variable.
2. Write a Python program to display some basic statistical details like percentile, mean, standard
deviation etc. of the species of ‘Iris-setosa’, ‘Iris-versicolor’ and ‘Iris- versicolor’ of iris.csv dataset.
Provide the codes with outputs and explain everything that you do in this step.
OBJECTIVE: To analyze and demonstrate knowledge of statistical data analysis techniques for
decision-making
PREREQUISITE: -
1. Basic of Python Programming
2. Concept of statistics mean, median, minimum, maximum, standard deviation

THEORY:
The data are summarized in some, but not all ways. We chose descriptives that are either most often
reported, or most often covered in introductory courses. These are as follows:
 Central Tendency
o Mean
o Median
o Mode
 Dispersion
o Standard deviation (Std. deviation)
o Minimum
o Maximum
 Central Tendency
The Mean, Median and Mode are the three measures of central tendency. Mean is the arithmetic average of
a data set. This is found by adding the numbers in a data set and dividing by the number of observations in
the data set. The median is the middle number in a data set when the numbers are listed in either ascending

Third Year of Artificial Intelligence and Data Science Software Laboratory III

26

or descending order. The mode is the value that occurs the most often in a data set and the range is the
difference between the highest and lowest values in a data set.
The Mean

Here,
∑ represents the summation
X represents observations
N represents the number of observations .
In the case where the data is presented in a tabular form, the following formula is used to compute the
mean
Mean = ∑f x / ∑f
Where ∑f = N
The Median
If the total number of observations (n) is an odd number, then the formula is given below:

If the total number of the observations (n) is an even number, then the formula is given below:

Consider the case where the data is continuous and presented in the form of a frequency distribution, the
median formula is as follows.
Find the median class, the total count of observations ∑f.
The median class consists of the class in which (n / 2) is present.

Here
l = lesser limit belonging to the median class
c = cumulative frequency value of the class before the median class
f = frequency possessed by the median class
h = size of the class
The Mode

Third Year of Artificial Intelligence and Data Science Software Laboratory III

27
The mode is the most frequently occurring observation or value.
Consider the case where the data is continuous, and the value of mode can be computed using the
following steps.
a] Determine the modal class that is the class possessing the maximum frequency.
b] Calculate the mode using the below formula.

l = lesser limit of modal class
fm = frequency possessed by the modal class
f1 = frequency possessed by the class before the modal class
f2 = frequency possessed by the class after the modal class
h = width of the class
Standard deviation

Steps for calculating the standard deviation.
The standard deviation is usually calculated automatically by whichever software you use for your
statistical analysis. But you can also calculate it by hand to better understand how the formula works.
There are six main steps for finding the standard deviation by hand. We’ll use a small data set of 6 scores
to walk through the steps.

Data set

4669 32605241

3. Step 1: Find the mean.
To find the mean, add up all the scores, then divide them by the number of scores.

Third Year of Artificial Intelligence and Data Science Software Laboratory III

28
Mean (x̅)

x̅= (46 + 69 + 32 + 60 + 52 + 41) ÷ 6 = 50

4. Step 2: Find each score’s deviation from the mean
Subtract the mean from each score to get the deviations from the mean.
Since x̅= 50, here we take away 50 from each score.

Score Deviation from the mean

46 46 – 50 = -4

69 69 – 50 = 19

32 32 – 50 = -18

60 60 – 50 = 10

52 52 – 50 = 2

41 41 – 50 = -9

5. Step 3: Square each deviation from the mean.
Multiply each deviation from the mean by itself. This will result in positive numbers.

Squared deviations from the mean
(-4)2 = 4 × 4 = 16
192 = 19 × 19 = 361

(-18)2 = -18 × -18 = 324
102 = 10 × 10 = 100

Third Year of Artificial Intelligence and Data Science Software Laboratory III

29

Squared deviations from the mean

2
2 = 2 × 2 = 4

(-9)2 = -9 × -9 = 81

6. Step 4: Find the sum of squares.
Add up all of the squared deviations. This is called the sum of squares.
Sum of squares

16 + 361 + 324 + 100 + 4 + 81 = 886

7. Step 5: Find the variance.
Divide the sum of the squares by n – 1 (for a sample) or N (for a population) – this is the variance.
Since we’re working with a sample size of 6, we will use n – 1, where n = 6.

Variance

886 ÷ (6 – 1) = 886 ÷ 5 = 177.2

8. Step 6: Find the square root of the variance.
To find the standard deviation, we take the square root of the variance.
Standard deviation
√177.2 = 13.31

From learning that SD = 13.31, we can say that each score deviates from the mean by 13.31 points on
average.
Consider HR dataset it contains fields like Age, Monthly Income, Attrition, Business Travel and so on so
following are steps to find statistics (mean, median, minimum, maximum, standard deviation)
#Mean of monthly income and age
print("The mean of monthly income is :",df.loc[:,"MonthlyIncome"].mean())
print("The mean of age is :",df.loc[:,"Age"].mean())

Third Year of Artificial Intelligence and Data Science Software Laboratory III

30

#Mode of monthly income and age
print("The median of monthly income is :",df.loc[:,"MonthlyIncome"].median())
print("The median of age is :",df.loc[:,"Age"].median())
#Median of monthly income and age
print("The mode of monthly income is :",df.loc[:,"MonthlyIncome"].mode())
print("The mode of age is :",df.loc[:,"Age"].mode())
#Standard deviation of monthly income and age
print("The standard deviation of monthly income is :",df.loc[:,"MonthlyIncome"].std())
print("The standard deviation of age is :",df.loc[:,"Age"].std())
#Storing age and monthly income in array and then finding maximum and minimum values
array1 = np.array(df['MonthlyIncome'])
array2=np.array(df["Age"])
print("Income",array1)
print("Age array",array2)
print("Maximum income among the employees is :",max(array1))
print("Minimum income among the employees is :",min(array1))
print("Maximum age among the employees is :",max(array2))
print("Minimum age among the employees is :",min(array2))
# Replacing the categorical values by numeric values
df.head()
df["BusinessTravel"].replace({"Travel_Rarely":1, "Travel_Frequently":0}, inplace=True)
df["Attrition"].replace({ "Yes":1, "No":0}, inplace=True)
df.head()

##Use of describe()
To display basic statistical details like percentile, mean, standard deviation etc. for Iris-Vigginica use
describe.

Third Year of Artificial Intelligence and Data Science Software Laboratory III

31

Conclusion: In this way we have explored the functions of the python library for calculating Data statistics
(mean, median, minimum, maximum, standard deviation).
Assignment Question:

1. How to find Mean Mode and Median.
2. How to find standard deviation.
3. Write a Python program to display some basic statistical details like percentile, mean,
standard deviation etc.
4. What is use of Describe () Command?

Third Year of Artificial Intelligence and Data Science Software Laboratory III

32

Assignment No: 4

PROBLEM STATEMENT: Create a Linear Regression Model using Python/R to predict
home prices using Boston Housing Dataset (https://www.kaggle.com/c/boston-housing). The
Boston Housing dataset contains information about various houses in Boston through different
parameters. There are 506 samples and 14 feature variables in this dataset.
The objective is to predict the value of prices of the house using the given features.

Objective of the Assignment: Students should be able to data analysis using liner regression
using Python for any open-source dataset.
Prerequisite:
1. Basic of Python Programming
2.Concept of Regression.
Contents for Theory:
1. Linear Regression: Univariate and Multivariate
2. Least Square Method for Linear Regression
3. Measuring Performance of Linear Regression
4. Example of Linear Regression
5. Training data set and Testing data set
1. Linear Regression: It is a machine learning algorithm based on supervised learning. It
targets prediction values on the basis of independent variables.
● It is preferred to find out the relationship between forecasting and variables.
● A linear relationship between a dependent variable (X) is continuous; while
independent variable(Y) relationship may be continuous or discrete. A linear
relationship should be available in between predictor and target variable so knownas
Linear Regression.
● Linear regression is popular because the cost function is Mean Squared Error (MSE)

Third Year of Artificial Intelligence and Data Science Software Laboratory III

33

which is equal to the average squared difference between an observation’s actual and
predicted values.
● It is shown as an equation of line like :Y =
m*X + b + e
Where : b is intercepted, m is slope of the line and e is error term.

This equation can be used to predict the value of target variable Y based on given predictor
variable(s) X, as shown in Fig. 1.

Fig. 1: geometry of linear regression

● Fig. 2 shown below is about the relation between weight (in Kg) and height (in cm), a
linear relation. It is an approach of studying in a statistical manner to summarise and
learn the relationships among continuous (quantitative) variables.
● Here a variable, denoted by ‘x’ is considered as the predictor, explanatory, or
independent variable.
● Another variable, denoted ‘y’, is considered as the response, outcome, or dependent
variable. While "predictor" and "response" used to refer to these variables.
● Simple linear regression technique concerned with the study of only one predictor
variable.

Fig.2 : Relation between weight (in Kg) and height (in cm)

Third Year of Artificial Intelligence and Data Science Software Laboratory III

34

MultiVariate Regression :It concerns the study of two or more predictor variables. Usually a
transformation of the original features into polynomial features from a given degree is preferred
and further Linear Regression is applied on it.
● A simple linear model Y = a + bX is in original feature will be transformed into
polynomial feature is transformed and further a linear regression applied to it and it will
be something like

Y=a + bX + cX2

● If a high degree value is used in transformation the curve becomes over-fitted as it
captures the noise from data as well.

2.Least Square Method for Linear Regression
● Linear Regression involves establishing linear relationships between dependent and
independent variables. Such a relationship is portrayed in the form of an equation also known
as the linear model.
● A simple linear model is the one which involves only one dependent and one independent
variable. Regression Models are usually denoted in Matrix Notations.
● However, for a simple univariate linear model, it can be denoted by the regression
equation

9.y = β+ β x
0 1

(1)

Third Year of Artificial Intelligence and Data Science Software Laboratory III

35

where y is the dependent or the response variable x is the independent or the input variable
β is the value of y when x=0 or the y intercept
0
β is the value of slope of the line ε is the error or the noise
1
● This linear equation represents a line also known as the ‘regression line’. The least square
estimation technique is one of the basic techniques used to guess the values of the
parameters and based on a sample set.
● This technique estimates parameters β
0
and β and by trying to minimize the square
1

of errors at all the points in the sample set. The error is the deviation of the actual sample
● data point from the regression line. The technique can be represented by the equation.

n

2
min ∑ (y − y)
i=0

(2)

Third Year of Artificial Intelligence and Data Science Software Laboratory III

36

Using differential calculus on equation 1 we can find the values of β
0
and β such

that the sum of squares (that is equation 2) is minimum.

n

n

2

β = ∑ (x
1 i
i=1
− x ) (y − y )/
i

∑ (x
i=1
i −
x ) (3)

β = y − β x
0 1

(4)

Third Year of Artificial Intelligence and Data Science Software Laboratory III

37

Once the Linear Model is estimated using equations (3) and (4), we can estimate thevalue
of the dependent variable in the given range only. Going outside the range is called
extrapolation which is inaccurate if simple regression techniques are used.
3.Measuring Performance of Linear RegressionMean Square Error:
The Mean squared error (MSE) represents the error of the estimator or predictive model
created based on the given set of observations in the sample. Two or more regression
models created using a given sample data can be compared based on their MSE. The lesser
the MSE, the better the regression model is. When the linear regression model is trained
using a given set of observations, the model with the least mean sum of squares error (MSE)
is selected as the best model. The Python or R packages select the best-fit model as the
model with the lowest MSE or lowest RMSE when training the linear regression models.
Mathematically, the MSE can be calculated as the average sum of the squared difference
between the actual value and the predicted or estimated value represented by the regression
model (line or plane).

An MSE of zero (0) represents the fact that the predictor is a perfect predictor.
RMSE:
Root Mean Squared Error method that basically calculates the least-squares error and takes a
root of the summed values.
Mathematically speaking, Root Mean Squared Error is the square root of the sum of all errors
divided by the total number of values. This is the formula to calculate RMSE

Third Year of Artificial Intelligence and Data Science Software Laboratory III

38

RMSE - Least Squares Regression Method – Edureka R-Squared :

R-Squared is the ratio of the sum of squares regression (SSR) and the sum of squares total
(SST).
SST : total sum of squares (SST), regression sum of squares (SSR), Sum of square of errors
(SSE) are all showing the variation with different measures.

Third Year of Artificial Intelligence and Data Science Software Laboratory III

39

A value of R-squared closer to 1 would mean that the regression model covers
most partof the variance of the values of the response variable and can be
termed as a good model.
One can alternatively use MSE or R-Squared based on what is appropriate and the need of the
hour. However, the disadvantage of using MSE rather than R-squared is that it will be difficult
to gauge the performance of the model using MSE as the value of MSE can vary from 0 toany
larger number. However, in the case of R-squared, the value is bounded between 0 and .
4.Example of Linear Regression
Consider the following data for 5 students.
Each Xi (i = 1 to 5) represents the score of i

th students in standard X and
correspondingYi (i = 1 to 5) represents the score of ith students in standard XII.
(i) Linear regression equation best predicts standard XIIth score.
(ii) Interpretation for the equation of Linear Regression
(iii) If a student's score is 80 in std X, then what is his expected score in XII standard?
Student Score in X standard (Xi) Score in XII standard (Yi)
1 95 85
2 85 95
3 80 70
4 70 65
5 60 70

(i) linear regression equation that best predicts standard XIIth score

Third Year of Artificial Intelligence and Data Science Software Laboratory III

40
(ii) Interpretation of the regression line.
Interpretation 1
For an increase in value of x by 0.644 units there is an increase in value of y in one unit.

Interpretation 2
Even if x = 0 value of independent variable, it is expected that value of y is 26.768 Score in XII
standard (Yi) is 0.644 units depending on Score in X standard (Xi) but otherfactors will also
contribute to the result of XII standard by 26.768 .
(iii) If a student's score is 65 in std X, then his expected score in XII standard is
78.288
For x = 80 the y value will be
5.Training data set and Testing data set
● Machine Learning algorithm has two phases
1. Training and 2. Testing.

● The input of the training phase is training data, which is passed to any machine learning
algorithm and machine learning model is generated as output of the training phase.
● The input of the testing phase is test data, which is passed to the machine learning model
and prediction is done to observe the correctness of mode.

Fig. 1.3.1 : Training and Testing Phase in Machine Learning
(a) Training Phase

Third Year of Artificial Intelligence and Data Science Software Laboratory III

41
● Training dataset is provided as input to this phase.
● Training dataset is a dataset having attributes and class labels and used for training
Machine Learning algorithms to prepare models.Machines can learn when they observe
enough relevant data. Using this one can model algorithms to find relationships, detect
patterns, understand complex problems and makedecisions.
● Training error is the error that occurs by applying the model to the same data from which
the model is trained.
● In a simple way the actual output of training data and predicted output of the model does
not match the training error Ein is said to have occurred.
● Training error is much easier to compute.
(b) Testing Phase
● Testing dataset is provided as input to this phase.
● Test dataset is a dataset for which class label is unknown. It is tested using model
● A test dataset used for assessment of the finally chosen model.
● Training and Testing dataset are completely different.
● Testing error is the error that occurs by assessing the model by providing the unknown
data to the model.
● In a simple way the actual output of testing data and predicted output of the model does
not match the testing error Eout is said to have occurred.
● E out is generally observed larger than Ein.
(c) Generalization
● Generalization is the prediction of the future based on the past system.
● It needs to generalize beyond the training data to some future data that it might not have
seen yet.
● The ultimate aim of the machine learning model is to minimize the generalization error.
● The generalization error is essentially the average error for data the model has never
seen.
● In general, the dataset is divided into two partition training and test sets.
● The fit method is called on the training set to build the model.
● This fit method is applied to the model on the test set to estimate the target value and
evaluate the model's performance.
● The reason the data is divided into training and test sets is to use the test set to estimate
how well the model trained on the training data and how well it would perform on the
unseen data.
Algorithm (Synthesis Dataset):
Step 1: Import libraries and create alias for Pandas, Numpy and Matplotlib
import pandas as pd
import numpy as np

Third Year of Artificial Intelligence and Data Science Software Laboratory III

42

import matplotlib.pyplot as plt
Step 2: Create a Dataframe with Dependent Variable(x) and independent variable y.
x=np.array([95,85,80,70,60])
y=np.array([85,95,70,65,70])
Step 3 : Create Linear Regression Model using Polyfit Function:
model= np.polyfit(x, y, 1)
Step 4: Observe the coefficients of the model.
model
Output:
array([ 0.64383562, 26.78082192])
Step 5: Predict the Y value for X and observe the output.
predict = np.poly1d(model)
predict(65)
Output:
68.63
Step 6: Predict the y_pred for all values of x.
y_pred= predict(x)
y_pred
Output:
array([81.50684932, 87.94520548, 71.84931507, 68.63013699, 71.84931507])
Step 7: Evaluate the performance of Model (R-Suare)
R squared calculation is not implemented in numpy... so that one should be borrowed
from sklearn.
from sklearn.metrics import r2_score
r2_score(y, y_pred)
Output:
0.4803218090889323
Step 8: Plotting the linear regression model
y_line = model[1] + model[0]* x
plt.plot(x, y_line, c = 'r') plt.scatter(x,
y_pred) plt.scatter(x,y,c='r')

Third Year of Artificial Intelligence and Data Science Software Laboratory III

43

Output:

Algorithm (Boston Dataset):
Step 1: Import libraries and create alias for Pandas, Numpy and Matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
Step 2: Import the Boston Housing dataset
from sklearn.datasets import load_bostonboston =
load_boston()
Step 3: Initialize the data frame
data = pd.DataFrame(boston.data)
Step 4: Add the feature names to the dataframe
data.columns = boston.feature_names data.head()
Step 5: Adding target variable to dataframe
data['PRICE'] = boston.target
Step 6: Perform Data Preprocessing( Check for missing values)
data.isnull().sum()
Step 7: Split dependent variable and independent variables
x = data.drop(['PRICE'], axis = 1)y =
data['PRICE']
Step 8: splitting data to training and testing dataset.
from sklearn.model_selection import train_test_splitxtrain, xtest,
ytrain, ytest =
train_test_split(x, y, test_size =0.2,random_state = 0)
Step 9: Use linear regression( Train the Machine ) to Create Model

Third Year of Artificial Intelligence and Data Science Software Laboratory III

44

import sklearn
from sklearn.linear_model import LinearRegressionlm =
LinearRegression()
model=lm.fit(xtrain, ytrain)
Step 10: Predict the y_pred for all values of train_x and test_x
ytrain_pred = lm.predict(xtrain)ytest_pred
= lm.predict(xtest)
Step 11:Evaluate the performance of Model for train_y and test_y
df=pd.DataFrame(ytrain_pred,ytrain)
df=pd.DataFrame(ytest_pred,ytest)
Step 12: Calculate Mean Square Paper for train_y and test_y
from sklearn.metrics import mean_squared_error, r2_scoremse =
mean_squared_error(ytest, ytest_pred)
print(mse)
mse = mean_squared_error(ytrain_pred,ytrain)print(mse)
Output:
33.44897999767638
mse = mean_squared_error(ytest, ytest_pred)print(mse)
Output:
19.32647020358573
Step 13: Plotting the linear regression model
lt.scatter(ytrain ,ytrain_pred,c='blue',marker='o',label='Training data')
plt.scatter(ytest,ytest_pred ,c='lightgreen',marker='s',label='Test data')plt.xlabel('True values')
plt.ylabel('Predicted')
plt.title("True value vs Predicted value")plt.legend(loc=
'upper left') #plt.hlines(y=0,xmin=0,xmax=50)
plt.plot()
plt.show()

Third Year of Artificial Intelligence and Data Science Software Laboratory III

45

Conclusion:
In this way we have done data analysis using linear regression for Boston Dataset andpredict the
price of houses using the features of the Boston Dataset.
Assignment Question:
1) Compute SST, SSE, SSR, MSE, RMSE, R Square for the below example .
Student Score in X standard (Xi) Score in XII standard (Yi)
1 95 85
2 85 95
3 80 70
4 70 65
5 60 70

2) Comment on whether the model is best fit or not based on the calculated values.
3) Write python code to calculate the RSquare for Boston Dataset. (Consider the linear regression
model created in practical session)

Assignment No: 5

Title:
1. Implement logistic regression using Python/R to perform classification on
Social_Network_Ads.csv dataset.
2. Compute Confusion matrix to find TP, FP, TN, FN, Accuracy, Error rate, Precision, Recall
on the given dataset.
Objective: Students should be able to data analysis using logisticregression using Python for any
open-source dataset
Prerequisite:
1. Basic of Python Programming
2.Concept of Regression.

Contents for Theory:
1. Logistic Regression
2. Differentiate between Linear and Logistic Regression
3. Sigmoid Function
4. Types of Logistic Regression
5. Confusion Matrix Evaluation Metrics

1. Logistic Regression: Classification techniques are an essential part of machine learning
and data mining applications. Approximately 70% of problems in Data Science are
classification problems. There are lots of classification problems that are available, but
logistic regression is common and is a useful regression method for solving the binary
classification problem. Another category of classification is Multinomial classification,
which handles the issues where multiple classes are present in the target variable. For
example, the IRIS dataset is a very famous example of multi-class classification. Other
examples are classifying article/blog/document categories.

Logistic Regression can be used for various classification problems such as spam detection.
Diabetes prediction, if a given customer will purchase a particular product or will they
churn another competitor, whether the user will click on a given advertisement link or not,
and many more examples are in the bucket.
Logistic Regression is one of the most simple and commonly used Machine Learning
algorithms for two-class classification. It is easy to implement and can be used as the
baseline for any binary classification problem. Its basic fundamental concepts are also
constructive in deep learning. Logistic regression describes and estimates the relationship
between one dependent binary variable and independent variables.
Logistic regression is a statistical method for predicting binary classes. The outcome or
target variable is dichotomous in nature. Dichotomous means there are only two possible
classes. For example, it can be used for cancer detection problems. It computes the
probability of an event occurring.
It is a special case of linear regression where the target variable is categorical in nature. It
uses a log of odds as the dependent variable. Logistic Regression predicts the probability
of occurrence of a binary event utilising a logit function.

Linear Regression Equation:

Where, y is a dependent variable and x1, x2 ... and Xn are explanatory variables.
Sigmoid Function:

Apply Sigmoid function on linear regression:

2. Differentiate between Linear and Logistic Regression
Linear regression gives you a continuous output, but logistic regression provides a constant output.
An example of the continuous output is house price and stock price. Example's of the discrete
output is predicting whether a patient has cancer or not, predicting whether the customer will churn.
Linear regression is estimated using Ordinary Least Squares (OLS) while logistic regression is
estimated using Maximum Likelihood Estimation (MLE) approach.

3. Sigmoid Function
The sigmoid function, also called logistic function, gives an ‘S’ shaped curve that can take any
real-valued number and map it into a value between 0 and 1. If the curve goes to positive infinity,
y predicted will become 1, and if the curve goes to negative infinity, y predicted will become 0.
If the output of the sigmoid function is more than 0.5, we can classify the outcome as 1 or YES,
and if it is less than 0.5, we can classify it as 0 or NO. The outputcannotFor example: If the output
is 0.75, we can say in terms of probability as: There is a 75 percent chance that a patient will suffer
from cancer.

4.Types of LogisticRegression
Binary Logistic Regression: The target variable has only two possible outcomes such as
Spam or Not Spam, Cancer or No Cancer.
Multinomial Logistic Regression: The target variable has three or more nominal
categories such as predicting the type of Wine.
Ordinal Logistic Regression: the target variable has three or more ordinal categories
such as restaurant or product rating from 1 to 5.

5. Confusion Matrix Evaluation Metrics
Contingency table or Confusion matrix is often used to measure the performance of classifiers.
A confusion matrix contains information about actual and predicted classifications done by a
classification system. Performance of such systems is commonly evaluated using the data in
the matrix.
The following table shows the confusion matrix for a two class classifier.

Here each row indicates the actual classes recorded in the test data set and the each column
indicates the classes as predicted by the classifier.
Numbers on the descending diagonal indicate correct predictions, while the ascending diagonal
concerns prediction errors.
Some Important measures derived from confusion matrix are:
● Number of positive (Pos) : Total number instances which are labelled as positive in a
given dataset.
● Number of negative (Neg) : Total number instances which are labelled as negative in a
given dataset.
● Number of True Positive (TP) : Number of instances which are actually labelled as
positive and the predicted class by classifier is also positive.
● Number of True Negative (TN) : Number of instances which are actually labelled as
negative and the predicted class by classifier is also negative.
● Number of False Positive (FP) : Number of instances which are actually labelled as
negative and the predicted class by classifier is positive.
● Number of False Negative (FN): Number of instances which are actually labelled as
positiveand the class predicted by the classifier is negative.

TP+FN

● Accuracy: Accuracy is calculated as the number of correctly classified instances divided
by totalnumber of instances.
The ideal value of accuracy is 1, and the worst is 0. It is also calculated as the sum of true
positive and true negative (TP + TN) divided by the total number of instances.

TP+TN
TP+FP+TN+FN

TP+TN
Pos+Neg

● Error Rate: Error Rate is calculated as the number of incorrectly classified instances
divided by total number of instances.
The ideal value of accuracy is 0, and the worst is 1. It is also calculated as the sum of
false positive and false negative (FP + FN) divided by the total number of instances.

FP+FN
TP+FP+TN+FN

FP+FN
Pos+Neg Or

10. err = 1 − acc

● Precision: It is calculated as the number of correctly classified positive instances divided
by the total number of instances which are predicted positive. It is also called confidence
value. The ideal value is 1, whereas the worst is 0.

● Recall: .It is calculated as the number of correctly classified positive instances divided by
the total number of positive instances. It is also called recall or sensitivity. The ideal value
of sensitivity is 1, whereas the worst is 0.
It is calculated as the number of correctly classified positive instances divided by the total
number of positive instances.

recall = TP
acc = =

err = =

Algorithm (Boston Dataset):
Step 1: Import libraries and create alias for Pandas, Numpy and Matplotlib
Step 2: Import the Social_Media_Adv Dataset
Step 3: Initialize the data frame
Step 4: Perform Data
Preprocessing
● ConvertCategoricaltoNumericalValuesif applicable
● Check for Null Value
● Covariance Matrix to select the most promising features
● Divide the dataset into Independent(X) and
Dependent(Y)variables.
● Split the dataset into training and testing datasets
● Scale the Featuresif necessary.
Step 5: Use Logistic regression( Train the Machine ) to Create Model
# import the class
from sklearn.linear_model import LogisticRegression# instantiate themodel
(using the default parameters)
logreg=LogisticRegression()# fit the model ith data
logreg.fit(xtrain,ytrain)
# y_pred=logreg.predict(xtest)
Step 6: Predict the y_pred for all values of train_x and test_x

Step 7: Evaluate the performance of Model for train_y and test_y

Step 8: Calculate the required evaluation parameters
from sklearn.metrics import precision_score,confusion_matrix,accuracy_score,recall_score
cm= confusion_matrix(ytest, y_pred)

Conclusion:
In this way we have done data analysis using logistic regression for Social Media Adv. and
evaluate the performance of model.

: DSBDAL

.

Assignment No: 6

Title :
1. Implement Simple Naïve Bayes classification algorithm using Python/R on iris.csv
dataset.
2. Compute Confusion matrix to find TP, FP, TN, FN, Accuracy, Error rate, Precision,
Recall on the given dataset.

Objective:
 Students should be able to data analysis using Naïve Bayes classification algorithm
using Python for any open-source dataset.
Prerequisite:
1. Basic of Python Programming
2.Concept of Join and Marginal Probability.

Contents for Theory:
1. Concepts used in Naïve Bayes classifier.
2. Naive Bayes Example
3. Confusion Matrix Evaluation Metrics

1. Concepts used in Naïve Bayes classifier.
● Naïve Bayes Classifier can be used for Classification of categorical data.
○ Let there be a ‘j’ number of classes. C={1,2,....j}
○ Let, input observation is specified by ‘P’ features. Therefore input observation x
is given , x = {F1,F2,.....Fp}
○ The Naïve Bayes classifier depends on Bayes' rule from probability theory.
● Prior probabilities: Probabilities which are calculated for some event based on no other
information are called Prior probabilities.

: DSBDAL

For example, P(A), P(B), P(C) are prior probabilities because while calculating P(A),
occurrences of event B or C are not concerned i.e. no information about occurrence of
any other event is used.
Conditional Probabilities:

From equation (1) and (2) ,

Is called the Bayes Rule.
2. Example of Naive Bayes
We have a dataset with some features Outlook, Temp, Humidity, and Windy, and the
target here is to predict whether a person or team will play tennis or not.

Conditional Probability

: DSBDAL

Here, we are predicting the probability of class1 and class2 based on the given condition. If I try
to write the same formula in terms of classes and features, we will get the following equation

Now we have two classes and four features, so if we write this formula for class C1, it will be
something like this.

Here, we replaced Ck with C1 and X with the intersection of X1, X2, X3, X4. You might have a
question, It’s because we are taking the situation when all these features are present at the same
time.
The Naive Bayes algorithm assumes that all the features are independent of each other or in other
words all the features are unrelated. With that assumption, we can further simplify the above
formula and write it in this form

This is the final equation of the Naive Bayes and we have to calculate the probability of both C1
and C2.For this particular example.

: DSBDAL

P (N0 | Today) > P (Yes | Today) So, the prediction that golf would be played is ‘No’.

1) Algorithm (Iris Dataset):
Step 1: Import libraries and create alias for Pandas, Numpy and Matplotlib
Step 2: Import the Iris dataset by calling URL.
Step 3: Initialize the data frame
Step 4: Perform Data Preprocessing
● ConvertCategoricaltoNumericalValuesif applicable
● Check for Null Value
● Divide the dataset into Independent (X) and Dependent (Y) variables.
● Split the dataset into training and testing datasets.
● Scale the Featuresif necessary.
2) Step 5: Use Naive Bayes algorithm(Train the Machine ) to Create Model
# import the class
fromsklearn.naive_bayesimportGaussianNBgaussian =
GaussianNB() gaussian.fit(X_train, y_train)
3) Step 6: Predict the y_pred for all values of train_x and test_x
Y_pred=gaussian.predict(X_test)
4) Step 7:Evaluate the performance of Model for train_y and test_y
accuracy=accuracy_score(y_test,Y_pred)

: DSBDAL
precision =precision_score(y_test, Y_pred,average='micro')recall=
recall_score(y_test, Y_pred,average='micro')
5) Step 8: Calculate the required evaluation parameters
from sklearn.metrics import precision_score,confusion_matrix,accuracy_score,recall_score
cm = confusion_matrix(y_test, Y_pred)
Conclusion:
In this way we have done data analysis using Naive Bayes Algorithm for Iris dataset and
evaluated the performance of the model.

: DSBDAL

Assignment No: 7

Title of the Assignment:
1. Extract Sample document and apply following document preprocessing methods:
Tokenization, POS Tagging, stop words removal, Stemming and Lemmatization.
2. Create representation of document by calculating Term Frequency and Inverse Document
Frequency.

Objective:
Students should be able to perform Text Analysis using TFIDF Algorithm
Prerequisite:
1. Basic of Python Programming
2. Basic of English language.

Contents for Theory:
1. Basic concepts of Text Analytics
2. Text Analysis Operations using natural language toolkit
3. Text Analysis Model using TF-IDF.
4. Bag of Words (BoW)

1. Basic concepts of Text Analytics
One of the most frequent types of day-to-day conversion is text communication. In our everyday
routine, we chat, message, tweet, share status, email, create blogs, and offer opinions and criticism.
All of these actions lead to a substantial amount of unstructured text being produced. It is critical to
examine huge amounts of data in this sector of the online world and social media to determine people's
opinions.

Text mining is also referred to as text analytics. Text mining is a process of exploring sizable textual
data and finding patterns. Text Mining processes the text itself, while NLP processes with the
underlying metadata. Finding frequency counts of words, length of thesentence, presence/absence of
specific words is known as text mining. Natural language processing is one of the components of text
mining. NLP helps identify sentiment,finding entities in the sentence, and category of blog/article.
Text mining is preprocessed data for text analytics. In Text Analytics, statistical and machine learning
algorithms are used to classify information.

: DSBDAL

2. Text Analysis Operations using natural language toolkit
NLTK(natural language toolkit) is a leading platform for building Python programs to work with
human language data. It provides easy-to-use interfaces and lexical resources such as WordNet, along
with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing,
and semantic reasoning and many more.
Analysing movie reviews is one of the classic examples to demonstrate a simple NLP Bag-of-words
model, on movie reviews.
Tokenization:
Tokenization is the first step in text analytics. The process of breaking down a textparagraph into
smaller chunks such as words or sentences is called Tokenization. Token is a single entity that is the
building blocks for a sentence or paragraph.

● Sentence tokenization : split a paragraph into list of sentences using
sent_tokenize() method
● Word tokenization : split a sentence into list of words using word_tokenize() method

Stop words removal
Stopwords considered as noise in the text. Text may contain stop words such as is,am, are, this, a, an,
the, etc. In NLTK for removing stopwords, you need to create a list of stopwords and filter out your
list of tokens from these words.

Stemming and Lemmatization
Stemming is a normalization technique where lists of tokenized words are converted into shortened
root words to remove redundancy. Stemming is the process of reducing inflected (or sometimes
derived) words to their word stem, base or root form.
A computer program that stems word may be called a stemmer.E.g.
A stemmer reduces the words like fishing, fished, and fisher to the stem fish.
The stem need not be a word, for example the Porter algorithm reduces, argue,argued, argues,
arguing, and argus to the stem argu .

: DSBDAL

Lemmatization in NLTK is the algorithmic process of finding the lemma of a word depending on its
meaning and context. Lemmatization usually refers to the morphological analysis of words, which
aims to remove inflectional endings. It helps in returning the base or dictionary form of a word known
as the lemma.
Eg. Lemma for studies is study

Lemmatization Vs Stemming
Stemming algorithm works by cutting the suffix from the word. In a broader sensecuts either the
beginning or end of the word.

On the contrary, Lemmatization is a more powerful operation, and it takes into consideration
morphological analysis of the words. It returns the lemma which is the base form of all its inflectional
forms. In-depth linguistic knowledge is
required to create dictionaries and look for the proper form of the word. Stemming is a general
operation while lemmatization is an intelligent operation where the proper form will be looked in the
dictionary. Hence, lemmatization helps in forming better machine learning features.

2.2. POS Tagging

POS (Parts of Speech) tell us about grammatical information of words of the sentence by assigning
specific token (Determiner, noun, adjective, adverb ,verb, Personal Pronoun etc.) as tag (DT,NN
,JJ,RB,VB,PRP etc) to each words.
Word can have more than one POS depending upon the context where it is used. We can use POS
tags as statistical NLP tasks. It distinguishes a sense of word which is very helpful in text realization
and infer semantic information from text for sentiment analysis.
Part-II

Text Analysis Model using TF-IDF.
Term frequency–inverse document frequency(TFIDF) , is a numerical statistic that is intended to
reflect how important a word is to a document in a collection orcorpus.
● Term Frequency (TF)
It is a measure of the frequency of a word (w) in a document (d). TF is defined as the ratio of a word’s

: DSBDAL

occurrence in a document to the total number of words in a document. The denominator term in the
formula is to normalize since all the corpus documents are of different lengths.

Example:

The initial step is to make a vocabulary of unique words and calculate TF for eachdocument. TF will
be more for words that frequently appear in a document and less for rare words in a document.
● Inverse Document Frequency (IDF)
It is the measure of the importance of a word. Term frequency (TF) does not consider the importance
of words. Some words such as’ of’, ‘and’, etc. can be most frequently present but are of little
significance. IDF provides weightage to each word based on its frequency in the corpus D.

In our example, since we have two documents in the corpus, N=2.

: DSBDAL

Term Frequency — Inverse Document Frequency (TFIDF)

It is the product of TF and IDF.
TFIDF gives more weightage to the word that is rare in the corpus (all the documents).TFIDF
provides more importance to the word that is more frequent in the document.

After applying TFIDF, text in A and B documents can be represented as a TFIDF vector of
dimension equal to the vocabulary words. The value corresponding to each word representsthe
importance of that word in a particular document.
TFIDF is the product of TF with IDF. Since TF values lie between 0 and 1, not using ln can result in
high IDF for some words, thereby dominating the TFIDF. We don’t want that, and therefore, we use

: DSBDAL

ln so that the IDF should not completely dominate the TFIDF.
● Disadvantage of TFIDF
It is unable to capture the semantics. For example, funny and humorous are synonyms, but TFIDF
does not capture that. Moreover, TFIDF can be computationally expensive if the vocabulary is vast.
3. Bag of Words (BoW)
Machine learning algorithms cannot work with raw text directly. Rather, the text must beconverted
into vectors of numbers. In natural language processing, a common technique for extracting features
from text is to place all of the words that occur in the text in abucket. This approach is called a
bag of words model or BoW for short. It’s referred toas a “bag” of words because any information
about the structure of the sentence is lost.

Algorithm for Tokenization, POS Tagging, stop words removal, Stemming and
Lemmatization:
Step 1: Download the required packages
nltk.download('punkt') nltk.download('stopwords')
nltk.download('wordnet') nltk.download('averaged_perceptron_tagger')
Step 2: Initialize the text
text= "Tokenization is the first step in text analytics. The process of breaking down a text paragraph into
smaller chunkssuch as words or sentences is called Tokenization."
Step 3: Perform Tokenization
#SentenceTokenization
from nltk.tokenize import sent_tokenize
tokenized_text= sent_tokenize(text)print(tokenized_text)
#Word Tokenization
from nltk.tokenize import word_tokenizetokenized_word=word_tokenize(text)
print(tokenized_word)

Step 4: Removing Punctuations and Stop Word
# printstop words of English
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))print(stop_words)
text= "How to remove stop words with NLTK library in Python?"text= re.sub('[^a-zA-Z]', ' ',text)
tokens = word_tokenize(text.lower())filtered_text=[]

: DSBDAL

for w in tokens:
if w not in stop_words: filtered_text.append(w)
print("Tokenized Sentence:",tokens) print("Filterd Sentence:",filtered_text)
Step 5 : Perform Stemming
from nltk.stem import PorterStemmer
e_words= ["wait", "waiting", "waited", "waits"]ps =PorterStemmer()
for w in e_words:
rootWord=ps.stem(w)print(rootWord)
Step 6: Perform Lemmatization
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer() text = "studies studying
cries cry" tokenization = nltk.word_tokenize(text)
for w in tokenization:
print("Lemma for {} is {}".format(w,
wordnet_lemmatizer.lemmatize(w)))

Step 7: Apply POS Tagging to text
import nltk
from nltk.tokenize import word_tokenize data="The pink sweater fit her
perfectly"words=word_tokenize(data)
for word in words: print(nltk.pos_tag([word]))

Algorithm for Create representation of document by calculating TFIDF
Step 1: Import the necessary libraries.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
Step 2: Initialize the Documents.
documentA = 'Jupiter is the largest Planet' documentB = 'Mars is the fourth planet from the Sun'
Step 3: Create BagofWords (BoW) for Document A and B.
bagOfWordsA = documentA.split(' ')bagOfWordsB = documentB.split(' ')
Step 4: Create Collection of Unique words from Document A and B.
uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))
Step 5: Create a dictionary of words and their occurrence for each document in the
corpus
numOfWordsA = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsA: numOfWordsA[word] += 1

: DSBDAL
numOfWordsB = dict.fromkeys(uniqueWords, 0)for word in bagOfWordsB:
numOfWordsB[word] += 1
Step 6: Compute the term frequency for each of our documents.
def computeTF(wordDict, bagOfWords):tfDict = {}
bagOfWordsCount = len(bagOfWords) for word, count in
wordDict.items():
tfDict[word] = count / float(bagOfWordsCount)return tfDict
tfA = computeTF(numOfWordsA, bagOfWordsA)tfB =
computeTF(numOfWordsB, bagOfWordsB)
Step 7: Compute the term Inverse Document Frequency.
def computeIDF(documents):
import math
N = len(documents)
idfDict = dict.fromkeys(documents[0].keys(), 0)for document in documents:
for word, val in document.items():if val > 0:
idfDict[word] += 1
for word, val in idfDict.items(): idfDict[word] = math.log(N / float(val))
return idfDict
idfs = computeIDF([numOfWordsA, numOfWordsB])idfs
Step 8: Compute the term TF/IDF for all words.
def computeTFIDF(tfBagOfWords, idfs):tfidf = {}
for word, val in tfBagOfWords.items():tfidf[word] = val * idfs[word]
return tfidf
tfidfA = computeTFIDF(tfA, idfs)tfidfB = computeTFIDF(tfB,
idfs)
df = pd.DataFrame([tfidfA, tfidfB])df
Conclusion:
In this way we have done text data analysis using TF IDF algorithm.

Assignment Question:
1) Perform Stemming for text = "studies studying cries cry". Compare the results generated
with Lemmatization. Comment on your answer how Stemming and Lemmatization
differ from each other.
2) Write Python code for removing stop words from the below documents, conver the

: DSBDAL

documents into lowercase and calculate the TF, IDF and TFIDF score for each document.
documentA = 'Jupiter is the largest Planet' documentB = 'Mars is the fourth planet from the
Sun'

.

: DSBDAL

Assignment No: 8

Title of the Assignment: Data Visualizations I
1. Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and contains information about the
passengers who boarded the unfortunate Titanic ship. Use the Seaborn library to seeif we can
find any patterns in the data.
2. Write a code to check how the price of the ticket (column name: 'fare') for each passengeris
distributed by plotting a histogram.

Objective:

Students should be able to perform the data Visualizationoperation using Python on any open-
source dataset.

Prerequisite:
1. Basic of Python Programming
2. Seaborn Library, Concept of Data Visualization.

Contents for Theory:
1. Seaborn Library Basics
2. Know your Data
3. Finding patterns of data.
4. Checking how the price of the ticket (column name: 'fare') for each passenger is distributed
by plotting a histogram.

Theory:
Data Visualisation plays a very important role in Data mining. Various data scientists spent their time

exploring data through visualisation. To accelerate this process, we need to have a well-
documentation of all the plots.

Even plenty of resources can’t be transformed into valuable goods without planning and architecture.

1. Seaborn Library Basics
Seaborn is a Python data visualisation library based on matplotlib. It provides ahigh-level

: DSBDAL
interface for drawing attractive and informative statistical graphics.
For the installation of Seaborn, you may run any of the following in your command line.
pip install seaborn
conda install seaborn.
To import seaborn you can run the following command.
import seaborn as sns

2. Know your data
The dataset that we are going to use to draw our plots will be the Titanic dataset, which is downloaded
by default with the Seaborn library. All you have to do is use the load_dataset function and pass it
the name of the dataset.
Let's see what the Titanic dataset looks like. Execute the following script:
import pandas as pdimport
numpy as np
import matplotlib.pyplot as pltimport seaborn as
sns

dataset = sns.load_dataset('titanic')dataset.head()

The dataset contains 891 rows and 15 columns and contains information about the passengers who
boarded the unfortunate Titanic ship. The original task is to predict whether or not the passenger
survived depending upon different features such as their age, ticket, cabin they boarded, the class of
the ticket, etc. We will use the Seaborn library to see if we can find any patterns in the data.

3. Finding patterns of data.
Patterns of data can be find out with the help of different types of plots
Types of plots are:

: DSBDAL

A. Distribution Plots
a. Dist-Plot
b. Joint Plot
c. Rug Plot
B. Categorical Plots
a. Bar Plot
b. Count Plot
c. Box Plot
d. Violin Plot

C. Advanced Plots
a. Strip Plot
b. Swarm Plot

D. Matrix Plots

a. Heat Map
b. Cluster Map

Distribution Plots:
These plots help us to visualise the distribution of data. We can use these plots to understand the
mean, median, range, variance, deviation, etc of the data.
Distplot

● Dist plot gives us the histogram of the selected continuous variable.
● It is an example of a univariate analysis.
● We can change the number of bins i.e. number of vertical bars in a histogram

import seaborn as sns

: DSBDAL

sns.distplot(x = dataset['age'], bins = 10)
The line that you see represents the kernel density estimation. You can remove this lineby passing
False as the parameter for the kde attribute as shown below
sns.distplot(dataset['age'], bins = 10,kde=False)

Here the x-axis is the age and the y-axis displays frequency. For example, for bins = 10,there are
around 50 people having age 0 to 10
Joint Plot

● It is the combination of the distplot of two variables.
● It is an example of bivariate analysis.
● We additionally obtain a scatter plot between the variables to reflect their linear
relationship. We can customise the scatter plot into a hexagonal plot, where, the

: DSBDAL
more the colour intensity, the more will be the number of observations.

import seaborn as sns# For Plot 1
sns.jointplot(x = dataset['age'], y = dataset['fare'], kind ='scatter')
# For Plot 2
sns.jointplot(x = dataset['age'], y = dataset['fare'], kind = 'hex')

● From the output, you can see that a joint plot has three parts. A distribution plot at the top
for the column on the x-axis, a distribution plot on the right for the column on the y-axis
and a scatter plot in between that shows the mutual distribution of data for both the
columns. You can see that there is no correlation observed between prices and the fares.
● You can change the type of the joint plot by passing a value for the kind parameter. For
instance, if instead of a scatter plot, you want to display the distribution of data in the form
of a hexagonal plot, you can pass the value hex for the kind parameter.
● In the hexagonal plot, the hexagon with the most number of points gets darker colour. So
if you look at the above plot, you can see that most of the passengers are between the
ages of 20 and 30 and most of them paid between 10-50 for the tickets.

The Rug Plot
The rugplot() is used to draw small bars along the x-axis for each point in the dataset. To plot a
rug plot, you need to pass the name of the column. Let's plot a rug plot for fare.
sns.rugplot(dataset['fare'])

: DSBDAL

From the output, you can see that most of the instances for the fares have values between 0 and 100.
These are some of the most commonly used distribution plots offered by the Python's Seaborn
Library. Let's see some of the categorical plots in the Seaborn library.
Categorical Plots
Categorical plots, as the name suggests, are normally used to plot categorical data. The categorical
plots plot the values in the categorical column against another categorical column ora numeric
column. Let's see some of the most commonly used categorical data.
a. The Bar Plot
The barplot() is used to display the mean value for each value in a categorical column, against a
numeric column. The first parameter is the categorical column, the second parameter is the numeric
column while the third parameter is the dataset. For instance, if you want to know the mean value of
the age of the male and female passengers, you can use the bar plot as follows.
sns.barplot(x='sex', y='age', data=dataset)

: DSBDAL

From the output, you can clearly see that the average age of male passengers is just less than 40 while
the average age of female passengers is around 33.
In addition to finding the average, the bar plot can also be used to calculate other aggregate values
for each category. To do so, you need to pass the aggregate function to the estimator. For instance,
you can calculate the standard deviation for the age of each gender as follows:
import numpy as np
import matplotlib.pyplot as pltimport seaborn as
sns
sns.barplot(x='sex', y='age', data=dataset, estimator=np.std)
Notice, in the above script we use the std aggregate function from the numpy library to calculate the
standard deviation for the ages of male and female passengers. The output looks like this:

: DSBDAL

b. The Count Plot
The count plot is similar to the bar plot, however it displays the count of the categories in a specific
column. For instance, if we want to count the number of males and women passenger wecan do so
using count plot as follows:
sns.countplot(x='sex', data=dataset)

c. The Box Plot
The box plot is used to display the distribution of the categorical data in the form of quartiles.The
centre of the box shows the median value. The value from the lower whisker to the bottomof the
box shows the first quartile. From the bottom of the box to the middle of the box lies the second
quartile. From the middle of the box to the top of the box lies the third quartile and finallyfrom the
top of the box to the top whisker lies the last quartile.
Now let's plot a box plot that displays the distribution for the age with respect to each gender. You
need to pass the categorical column as the first parameter (which is sex in our case) and the numeric
column (age in our case) as the second parameter. Finally, the dataset is passed as the third parameter,
take a look at the following script:
sns.boxplot(x='sex', y='age', data=dataset)

: DSBDAL

Let's try to understand the box plot for females. The first quartile starts at around 1 and ends at 20
which means that 25% of the passengers are aged between 1 and 20. The second quartile starts at
around 20 and ends at around 28 which means that 25% of the passengers are aged between20 and
28. Similarly, the third quartile starts and ends between 28 and 38, hence 25% passengers are aged
within this range and finally the fourth or last quartile starts at 38 and ends around 64.
If there are any outliers or the passengers that do not belong to any of the quartiles, they are called
outliers and are represented by dots on the box plot.
You can make your box plots more fancy by adding another layer of distribution. For instance, ifyou
want to see the box plots of forage of passengers of both genders, along with the informationabout
whether or not they survived, you can pass the survived as value to the hue parameter as shown below:
sns.boxplot(x='sex', y='age', data=dataset, hue="survived")

Now in addition to the information about the age of each gender, you can also see the distribution of
the passengers who survived. For instance, you can see that among the male passengers, on average
more younger people survived as compared to the older ones. Similarly, you can see that the variation

: DSBDAL

among the age of female passengers who did not survive is much greater than the age of the surviving
female passengers.
d. The Violin Plot
The violin plot is similar to the box plot, however, the violin plot allows us to display all the
components that actually correspond to the data point. The violinplot() function is used to plotthe
violin plot. Like the box plot, the first parameter is the categorical column, the second parameter is
the numeric column while the third parameter is the dataset.
Let's plot a violin plot that displays the distribution for the age with respect to each gender.
sns.violinplot(x='sex', y='age', data=dataset)

You can see from the figure above that violin plots provide much more information about the data as
compared to the box plot. Instead of plotting the quartile, the violin plot allows us to see all the
components that actually correspond to the data. The area where the violin plot is thicker has a higher
number of instances for the age. For instance, from the violin plot for males, it is clearly evident that the
number of passengers with age between 20 and 40 is higher than all the rest of the age brackets.
Like box plots, you can also add another categorical variable to the violin plot using the hue parameter
as shown below:
sns.violinplot(x='sex', y='age', data=dataset, hue='survived')

: DSBDAL

Now you can see a lot of information on the violin plot. For instance, if you look at the bottom ofthe
violin plot for the males who survived (left-orange), you can see that it is thicker than the bottom of
the violin plot for the males who didn't survive (left-blue). This means that the numberof young male
passengers who survived is greater than the number of young male passengers who did not survive.
Advanced Plots:
The Strip Plot
The strip plot draws a scatter plot where one of the variables is categorical. We have seen scatter plots
in the joint plot and the pair plot sections where we had two numeric variables. The strip plot is
different in a way that one of the variables is categorical in this case, and for each category in
the categorical variable, you will see a scatter plot with respect to the numeric column.
The stripplot() function is used to plot the violin plot. Like the box plot, the first parameter is the
categorical column, the second parameter is the numeric column while the third parameter is the
dataset. Look at the following script:
sns.stripplot(x='sex', y='age', data=dataset, jitter=False)

: DSBDAL

You can see the scattered plots of age for both males and females. The data points look like strips. It
is difficult to comprehend the distribution of data in this form. To better comprehend thedata, pass
True for the jitter parameter which adds some random noise to the data. Look at the following script:
sns.stripplot(x='sex', y='age', data=dataset, jitter=True)

Now you have a better view for the distribution of age across the genders.
Like violin and box plots, you can add an additional categorical column to strip plot using hue
parameter as shown below:
sns.stripplot(x='sex', y='age', data=dataset, jitter=True, hue='survived')

: DSBDAL

The Swarm Plot
The swarm plot is a combination of the strip and the violin plots. In the swarm plots, the points are
adjusted in such a way that they don't overlap. Let's plot a swarm plot for the distribution of age
against gender. The swarmplot() function is used to plot the violin plot. Like the box plot, thefirst
parameter is the categorical column, the second parameter is the numeric column while the third
parameter is the dataset. Look at the following script:
sns.swarmplot(x='sex', y='age', data=dataset)

: DSBDAL

You can clearly see that the above plot contains scattered data points like the strip plot and the data
points are not overlapping. Rather they are arranged to give a view similar to that of a violinplot.
Let's add another categorical column to the swarm plot using the hue parameter.
sns.swarmplot(x='sex', y='age', data=dataset, hue='survived')

From the output, it is evident that the ratio of surviving males is less than the ratio of surviving
females. Since for the male plot, there are more blue points and less orange points. On the other hand,
for females, there are more orange points (surviving) than the blue points (not surviving). Another
observation is that amongst males of age less than 10, more passengers survived as compared to those
who didn't.
Matrix Plots
Matrix plots are the type of plots that show data in the form of rows and columns. Heat maps are the
prime examples of matrix plots.
Heat Maps
Heat maps are normally used to plot correlation between numeric columns in the form of a matrix. It
is important to mention here that to draw matrix plots, you need to have meaningful information on
rows as well as columns. Let's plot the first five rows of the Titanic dataset to see if both the rows and
column headers have meaningful information. Execute the following script:
import pandas as pd

: DSBDAL

import numpy as np
import matplotlib.pyplot as pltimport seaborn as
sns
dataset = sns.load_dataset('titanic')dataset.head()

From the output, you can see that the column headers contain useful information such aspassengers
surviving, their age, fare etc. However the row headers only contain indexes 0, 1, 2, etc. To plot
matrix plots, we need useful information on both columns and row headers. One wayto do this is to
call the corr() method on the dataset. The corr() function returns the correlation between all the
numeric columns of the dataset. Execute the following script:
dataset.corr()
In the output, you will see that both the columns and the rows have meaningful header information,
as shown below:

: DSBDAL

Now to create a heat map with these correlation values, you need to call the heatmap() function and
pass it your correlation dataframe. Look at the following script:
corr = dataset.corr()
sns.heatmap(corr)

From the output, it can be seen that what heatmap essentially does is that it plots a box for every
combination of rows and column value. The colour of the box depends upon the gradient. For
instance, in the above image if there is a high correlation between two features, the corresponding
cell or the box is white, on the other hand if there is no correlation, the corresponding cell remains
black.
The correlation values can also be plotted on the heatmap by passing True for the annotparameter.
Execute the following script to see this in action:
corr = dataset.corr() sns.heatmap(corr,
annot=True)

Third Year of Artificial Intelligence and Data Science Software Laboratory III

83

You can also change the colour of the heatmap by passing an argument for the cmap
parameter.For now, just look at the following script:
corr = dataset.corr()
sns.heatmap(corr)

a. Cluster Map:
In addition to the heat map, another commonly used matrix plot is the cluster map. The
cluster map basically uses Hierarchical Clustering to cluster the rows and columns of the
matrix.
Let's plot a cluster map for the number of passengers who travelled in a specific month ofa
specific year. Execute the following script:
4. Checking how the price of the ticket (column name: 'fare') for each

Third Year of Artificial Intelligence and Data Science Software Laboratory III

84
passenger isdistributed by plotting a histogram.
import seaborn as sns
dataset = sns.load_dataset('titanic') sns.histplot(dataset['fare'],
kde=False, bins=10)

From the histogram, it is seen that for around 730 passengers the price of the ticket is 50.
For 100 passengers the price of the ticket is 100 and so on.

Conclusion-
Seaborn is an advanced data visualisation library built on top of Matplotlib library. In this

assignment, we looked at how we can draw distributional and categorical plots using the
Seabornlibrary. We have seen how to plot matrix plots in Seaborn. We also saw how to
change plot stylesand use grid functions to manipulate subplots.
Assignment Questions
1. List out different types of plot to find patterns of data
2. Explain when you will use distribution plots and when you will use categorical
plots.
3. Write the conclusion from the following swarm plot (consider titanic dataset)

4. Which parameter is used to add another categorical variable to the violin
plot,Explain with syntax and example.

Third Year of Artificial Intelligence and Data Science Software Laboratory III

85

Assignment No: 9

Title of the Assignment: Data Visualization II
1. Use the inbuilt dataset 'titanic' as used in the above problem. Plot a box plot for distribution
of
age with respect to each gender along with the information about whether they survived or
not. (Column names : 'sex' and 'age')
2. Write observations on the inference from the above statistics.
Objective:
Students should be able to perform the data Visualization operation using Python on any
open-source dataset.

Prerequisite:
1. Basic of Python Programming
2. box plot Library, Concept of Data Visualization.

Contents for Theory:
1. Box plot Basics
2. Know your Data
3. Finding patterns of data.

Theory:
Data Visualisation plays a very important role in Data mining. Various data scientists spent
their time exploring data through visualisation. To accelerate this process, we need to have
a well-documentation of all the plots.
Even plenty of resources can’t be transformed into valuable goods without planning and
architecture.

The Box Plot
The box plot is used to display the distribution of the categorical data in the form of

Third Year of Artificial Intelligence and Data Science Software Laboratory III

86

quartiles.The centre of the box shows the median value. The value from the lower whisker
to the bottomof the box shows the first quartile. From the bottom of the box to the middle
of the box lies the second quartile. From the middle of the box to the top of the box lies the
third quartile and finallyfrom the top of the box to the top whisker lies the last quartile.
Now let's plot a box plot that displays the distribution for the age with respect to each gender.
You need to pass the categorical column as the first parameter (which is sex in our case)
and the numeric column (age in our case) as the second parameter. Finally, the dataset is
passed as the third parameter, take a look at the following script:
sns.boxplot(x='sex', y='age', data=dataset)

Let's try to understand the box plot for females. The first quartile starts at around 1 and ends
at 20 which means that 25% of the passengers are aged between 1 and 20. The second
quartile starts at around 20 and ends at around 28 which means that 25% of the passengers
are aged between20 and 28. Similarly, the third quartile starts and ends between 28 and 38,
hence 25% passengers are aged within this range and finally the fourth or last quartile starts
at 38 and ends around 64.
If there are any outliers or the passengers that do not belong to any of the quartiles, they are
called outliers and are represented by dots on the box plot.
You can make your box plots more fancy by adding another layer of distribution. For
instance, ifyou want to see the box plots of forage of passengers of both genders, along with
the informationabout whether or not they survived, you can pass the survived as value to

Third Year of Artificial Intelligence and Data Science Software Laboratory III

87

the hue parameter as shown below:
sns.boxplot(x='sex', y='age', data=dataset, hue="survived")

Now in addition to the information about the age of each gender, you can also see the
distribution of the passengers who survived. For instance, you can see that among the male
passengers, on average more younger people survived as compared to the older ones.
Similarly, you can see that the variation among the age of female passengers who did not
survive is much greater than the age of the surviving female passengers.
Know your data

The dataset contains 891 rows and 15 columns and contains information about the
passengers who boarded the unfortunate Titanic ship. The original task is to predict whether
or not the passenger survived depending upon different features such as their age, ticket,
cabin they boarded, the class of the ticket, etc.
Theory:
A box plot is a graphical representation of data that shows the distribution of a dataset. It
shows the median, quartiles, and outliers of a dataset. The box represents the interquartile
range (IQR), which is the range between the first quartile (Q1) and the third quartile (Q3).

Third Year of Artificial Intelligence and Data Science Software Laboratory III

88

The line inside the box represents the median. The whiskers represent the range of data,
excluding outliers, and the dots or asterisks represent outliers.
Example:
Here is an example code to plot a box plot for the distribution of age with respect to each
gender in the 'titanic' dataset using Python:
import seaborn as sns
import matplotlib.pyplot as plt
titanic = sns.load_dataset('titanic')
sns.boxplot(x='sex', y='age', hue='survived', data=titanic)
plt.show()
In this code, we first import the required libraries 'seaborn' and 'matplotlib.pyplot'. We then
load the 'titanic' dataset using the 'sns.load_dataset()' function. Next, we use the
'sns.boxplot()' function to plot the box plot for the distribution of age with respect to each
gender, along with information about whether they survived or not. We pass the 'sex' and
'age' columns of the dataset as the x and y parameters, respectively. We also pass the
'survived' column of the dataset as the hue parameter, which colors the box plot based on
whether the passengers survived or not. Finally, we use the 'plt.show()' function to display
the plot.

Conclusion-
We learned how to plot a box plot for the distribution of age with respect to each gender in the

'titanic' dataset, along with information about whether they survived or not. We also drew some
observations from the statistics. Box plots are a useful tool to visualize the distribution of data
and to draw inferences from it.
Assignment Questions
1. List out different types of plot to find patterns of data
2. Explain when you will use distribution plots and when you will use categorical
plots.
3. Write the conclusion from the following swarm plot (consider titanic dataset)

Third Year of Artificial Intelligence and Data Science Software Laboratory III

89
Group B
Assignment 11

PROBLEM STATEMENT:
Create databases and tables, insert small amounts of data, and run simple queries
using Impala
OBJECTIVE:
1. To Learn and understand the concepts of Python Libraries.
2. To learn and understand the Data Science for the analysis of real time problems
3. To understand and practice Data Preprocessing & Data Normalization.

 The objectives of this manual are to provide you with the knowledge to create databases
and tables in Impala, insert data into these tables, and execute simple queries.
 Students will have a good understanding of how to use Impala to analyze data.

Theory:
Impala is an open-source SQL query engine allows you to query data stored in Apache
Hadoop clusters. It's a fast, distributed, and highly scalable system that can run complex
queries on large datasets in near real-time. In this manual, we will guide you through the
process of creating databases and tables, inserting small amounts of data, and running simple
queries using Impala.
Impala uses a distributed architecture, which means that data is stored across multiple nodes in
a cluster. Impala also uses a query engine that allows you to write SQL queries that are executed
in parallel across the nodes in the cluster. This allows for fast query execution, even on large
datasets.
Impala uses a metadata store to keep track of the databases and tables that are created. The
metadata store is used to store information about the location of the data, the schema of the
tables, and other metadata. Impala supports a wide range of data formats, including Parquet,
Avro, and RCFile.
Example:
To create a database in Impala, you can use the following SQL command:

Third Year of Artificial Intelligence and Data Science Software Laboratory III

90

CREATE DATABASE my_database;
This will create a new database called "my_database". To create a table within this database,
you can use the following SQL command:
CREATE TABLE my_table ( id INT, name STRING, age INT );
This will create a new table called "my_table" with three columns: id, name, and age. To insert
data into this table, you can use the following SQL command:

INSERT INTO my_table VALUES (1, 'John', 25), (2, 'Jane', 30), (3, 'Bob', 40);
This will insert three rows into the table, each with a unique id, name, and age. To run a simple
query on this table, you can use the following SQL command:

SELECT * FROM my_table;
This will return all the rows in the table. You can also use more complex queries, such as
aggregations and joins, to analyze the data in the table.

Conclusion:
Impala is a powerful SQL query engine that can be used to analyze large datasets stored in
Apache Hadoop clusters. We have seen how to create databases and tables, insert data, and
execute simple queries in Impala.