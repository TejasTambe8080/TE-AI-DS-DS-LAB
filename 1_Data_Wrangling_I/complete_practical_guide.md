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
