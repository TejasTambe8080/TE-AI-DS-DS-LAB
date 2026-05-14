# Data Science Laboratory III (DSBDAL) - Complete Guide
## TE Artificial Intelligence & Data Science | Subject Code: 317534

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Quick Navigation](#quick-navigation)
4. [Complete Viva Questions Index](#complete-viva-questions-index)
5. [Viva Questions by Syllabus Topic](#viva-questions-by-syllabus-topic)
6. [Viva Preparation Guide](#viva-preparation-guide)
7. [External Examination Tips](#external-examination-tips)

---

## 📚 Project Overview

This repository contains **11 complete practical exercises** for the Data Science & Big Data Analytics Laboratory (DSBDAL) course, designed for the TE AI&DS program at Dr. D.Y. Patil Institute of Technology, Pune.

### Key Features:
✅ **11 Complete Practicals** - From Data Wrangling to Web Scraping  
✅ **330+ Viva Questions** - 30 questions per practical  
✅ **Comprehensive Theory** - Detailed explanations and algorithms  
✅ **Working Code Examples** - Production-ready Python implementations  
✅ **Real Datasets** - Iris, Boston Housing, Titanic, and more  
✅ **Syllabus Aligned** - Matches DR. D.Y. PATIL official curriculum  
✅ **Journal Ready** - Complete with theory, algorithms, and conclusions  

---

## 📁 Repository Structure

```
DSBDAL-DS-LAB/
├── README.md (This file)
├── VIVA_QUESTIONS.md (Standalone viva questions file)
│
├── 1_Data_Wrangling_I/
│   ├── 01_Data_Wrangling_I.ipynb
│   ├── complete_practical_guide.md
│   └── Iris.csv
│
├── 2_Data_Wrangling_II/
│   ├── 02_Data_Wrangling_II.ipynb
│   ├── complete_practical_guide.md
│   └── StudentPerformanceFactors.csv
│
├── 3_Descriptive_Statistics/
│   ├── 03_Descriptive_Statistics.ipynb
│   ├── complete_practical_guide.md
│   └── Iris.csv
│
├── 4_Linear_Regression/
│   ├── 04_Linear_Regression.ipynb
│   ├── complete_practical_guide.md
│   └── Boston Housing Data.csv
│
├── 5_Logistic_Regression/
│   ├── 05_Logistic_Regression.ipynb
│   ├── complete_practical_guide.md
│   └── Social_Network_Ads.csv
│
├── 6_Naive_Bayes/
│   ├── 06_Naive_Bayes.ipynb
│   ├── complete_practical_guide.md
│   └── Iris.csv
│
├── 7_Text_Analytics_TFIDF/
│   ├── 07_Text_Analytics_TFIDF.ipynb
│   ├── complete_practical_guide.md
│   └── [Sample documents]
│
├── 8_Data_Visualization/
│   ├── 08_Data_Visualization.ipynb
│   ├── complete_practical_guide.md
│   └── Titanic-Dataset.csv
│
├── 9_Clustering/
│   ├── 09_Clustering.ipynb
│   ├── complete_practical_guide.md
│   └── Iris.csv
│
├── 10_Association_Rule_Mining/
│   ├── 10_Association_Rule_Mining.ipynb
│   ├── complete_practical_guide.md
│   └── Market_Basket_Data.csv
│
└── 11_Web_Scraping_or_Final_Practical/
    ├── 11_Web_Scraping.ipynb
    ├── complete_practical_guide.md
    └── [Example scraped data]
```

---

## 🎯 Quick Navigation

### By Practical Number
| Practical | Title | Focus Area | Viva Questions | Key Dataset |
|-----------|-------|-----------|-----------------|-------------|
| **1** | Data Wrangling I | Missing Values, Normalization, Encoding | [Q1-Q30](#practical-1-data-wrangling-i) | Iris |
| **2** | Data Wrangling II | Outliers, Transformations, Skewness | [Q1-Q21](#practical-2-data-wrangling-ii) | Student Performance |
| **3** | Descriptive Statistics | Mean, Median, Std Dev, Percentiles | [Q1-Q20](#practical-3-descriptive-statistics) | Iris |
| **4** | Linear Regression | MSE, RMSE, R², Train-Test Split | [Q1-Q20](#practical-4-linear-regression) | Boston Housing |
| **5** | Logistic Regression | Sigmoid, Confusion Matrix, Metrics | [Q1-Q15](#practical-5-logistic-regression) | Social Network Ads |
| **6** | Naive Bayes | Bayes Theorem, Prior, Likelihood | [Q1-Q10](#practical-6-naive-bayes) | Iris |
| **7** | Text Analytics TF-IDF | Tokenization, Stemming, TF-IDF | [Q1-Q10](#practical-7-text-analytics-tfidf) | Text Documents |
| **8** | Data Visualization I | Histograms, Boxplots, Distribution Plots | [Q1-Q10](#practical-8-data-visualization-i) | Titanic |
| **9** | Clustering | K-Means, Elbow Method, Silhouette | [Q1-Q10](#practical-9-clustering) | Iris |
| **10** | Association Rules | Apriori, Support, Confidence, Lift | [Q1-Q10](#practical-10-association-rule-mining) | Market Basket |
| **11** | Web Scraping | BeautifulSoup, Selectors, Legal Ethics | [Q1-Q10](#practical-11-web-scraping) | Web Data |

---

## 📖 Complete Viva Questions Index

### PRACTICAL 1: DATA WRANGLING I

#### Q1: What is data wrangling and why is it important?
A: Data wrangling is the process of transforming raw data into a clean, usable format. It's important because 60-80% of data science involves wrangling, and "garbage in = garbage out" - clean data is essential for accurate models.

#### Q2: Name three types of missing data and explain MCAR.
A: Three types: MCAR (Missing Completely At Random - no pattern), MAR (Missing At Random - depends on other variables), MNAR (Missing Not At Random - depends on unobserved values). MCAR is safest to handle as it has no pattern.

#### Q3: When would you use deletion vs imputation for missing values?
A: Use deletion when <5% of data is missing (acceptable information loss). Use imputation when 5-30% is missing (preserve data). If >30% missing, consider dropping the entire column.

#### Q4: Explain Min-Max normalization with a formula and example.
A: Formula: X_normalized = (X - X_min) / (X_max - X_min), scales to [0,1]. Example: If age 25 with min=20, max=80: (25-20)/(80-20) = 5/60 = 0.083

#### Q5: What is the difference between normalization and standardization?
A: Normalization (Min-Max) scales to [0,1] and is sensitive to outliers. Standardization (Z-score) centers at 0 with std=1 and is resistant to outliers.

#### Q6: When would you use standardization over normalization?
A: Use standardization when: (1) data has outliers, (2) using algorithms like SVM or linear regression, (3) data approximately normally distributed. Use Min-Max for bounded features.

#### Q7: What is label encoding and when should you use it?
A: Label encoding converts categorical values to integers (0, 1, 2...). Use it for: (1) target variables in classification, (2) tree-based models, (3) ordinal categories. Avoid for nominal categories in linear models.

#### Q8: Why can't machine learning algorithms directly work with categorical data?
A: Because algorithms perform mathematical operations (multiplication, addition) on numerical values. Text like "Red", "Green" can't be used in equations, so encoding converts them to numbers.

#### Q9: Explain the difference between fit() and transform() in sklearn.
A: fit() learns parameters (min, max, mean, std) from data. transform() applies those parameters. fit_transform() does both. Always fit on training data, then transform both train and test.

#### Q10: What does MinMaxScaler.fit_transform() do in two steps?
A: Step 1: fit() finds min and max values of each column. Step 2: transform() applies formula (X-min)/(max-min) to scale all values to [0,1].

#### Q11: What is StandardScaler and what are the resulting mean and std?
A: StandardScaler applies Z-score normalization: (X-mean)/std. Result: mean becomes ~0, standard deviation becomes ~1, allowing fair comparison of features.

#### Q12: If you have 150 rows and apply fit_transform to X with shape (150, 4), what is output shape?
A: Output shape is (150, 4). Same as input - only values change, not dimensions.

#### Q13: How do you convert a sklearn numpy array output back to DataFrame with column names?
A: Use `pd.DataFrame(array, columns=original_columns)` where original_columns is list of feature names.

#### Q14: What is LabelEncoder.classes_ attribute used for?
A: It shows the mapping of encoded labels. For example, if classes_ = ['Setosa', 'Versicolor', 'Virginica'], then Setosa→0, Versicolor→1, Virginica→2.

#### Q15: Explain why outliers affect Min-Max scaling more than standardization.
A: Min-Max uses min/max values - extreme outliers stretch the denominator, compressing all other values. Standardization uses mean/std - outliers have less dramatic effect on std.

#### Q16: Can you apply normalization to only some features? How?
A: Yes, select specific columns: `scaler.fit_transform(df[['age', 'income']])` - this normalizes only age and income, not other columns.

#### Q17: What happens if you fit scaler on training data but test data has values outside [0,1]?
A: Test values outside [0,1] indicate they're outside training range. Not necessarily wrong (extrapolation), but may indicate data shift or outliers in test set.

#### Q18: How would you handle missing values in 'Income' column using mean imputation?
A: `df['Income'].fillna(df['Income'].mean(), inplace=True)` - replaces NaN with column's mean value.

#### Q19: What's the difference between fillna() and dropna()?
A: fillna() replaces NaN with specified value (preserves rows). dropna() removes entire rows/columns with NaN (loses data). Choose based on how much data is missing.

#### Q20: If you accidentally fit scaler on full data before splitting, what's the problem?
A: Data leakage - test data statistics (min, max, mean) influence training scaler. Realistic performance estimates compromised. Always fit scaler ONLY on training data.

#### Q21: How do you save a fitted scaler for later use?
A: Use pickle or joblib: `import joblib; joblib.dump(scaler, 'scaler.pkl')` then load with `joblib.load('scaler.pkl')`

#### Q22: When would you NOT normalize data?
A: Tree-based models (Random Forest, XGBoost) don't require normalization. They're scale-invariant.

#### Q23: Explain one-hot encoding vs label encoding.
A: Label encoding: Nominal categories→integers (0,1,2). One-hot: Each category→binary column. Use one-hot for nominal, label for ordinal.

#### Q24: Can you reverse label encoding?
A: Yes, using inverse_transform: `original_labels = encoder.inverse_transform([0, 1, 2])`

#### Q25: What does isnull().sum() do?
A: Returns count of missing values per column.

#### Q26: How to fill missing values differently for different columns?
A: `df['col1'].fillna(method='ffill'); df['col2'].fillna(df['col2'].mean())`

#### Q27: What's the purpose of checking data.dtypes?
A: Verify data types are correct (int, float, object, datetime). Identify columns needing type conversion.

#### Q28: Explain the GIGO principle in data science.
A: Garbage In = Garbage Out. If input data is poor quality, output predictions will be unreliable regardless of algorithm quality.

#### Q29: When should you use forward fill for missing values?
A: For time-series data where values evolve over time. Previous value is good estimate for next.

#### Q30: How to handle missing values in categorical columns?
A: fillna(mode), fillna('Unknown'), or create binary 'Missing' indicator column. Never drop if >5% missing.

---

### PRACTICAL 2: DATA WRANGLING II

#### Q1: What is an outlier?
A: A data point that lies far from other observations, vastly larger or smaller than remaining values.

#### Q2: How does IQR method calculate outlier bounds?
A: Lower Bound = Q1 - 1.5×IQR, Upper Bound = Q3 + 1.5×IQR. Values outside are outliers.

#### Q3: Explain Z-score for outlier detection.
A: Z = (X - mean) / std. |Z| > 3 = outlier. Shows how many standard deviations from mean.

#### Q4: When to use deletion vs capping for outliers?
A: Deletion when <1% outliers (acceptable loss). Capping preserves data but creates artificial values.

#### Q5: What is right skew and give an example.
A: Right skew: tail on right, Mean > Median. Example: Income distribution (most earn average, few earn millions).

#### Q6: What does log transformation do?
A: Converts exponential to linear, compresses large values, spreads small values, reduces right skew.

#### Q7: Why normalize before applying Box-Cox?
A: Box-Cox requires positive values. Transform: X_new = X + |min| + 1 to ensure X > 0.

#### Q8: How does Box-Cox find optimal transformation?
A: Tests different lambda values, finds one maximizing normality (highest likelihood). Automatic optimization.

#### Q9: What's difference between Median and Mean imputation?
A: Mean affected by outliers, pulls toward extreme values. Median is middle value, resistant to outliers.

#### Q10: Can you reverse log transformation?
A: Yes. If X_log = log(X), then X = e^(X_log). Called inverse transformation.

#### Q11: What are the three approaches to handle outliers?
A: (1) Deletion - remove rows, (2) Capping - replace with bounds, (3) Transformation - reduce spread.

#### Q12: Which outlier detection method works best for non-normal data?
A: IQR method - doesn't assume normal distribution. Z-score assumes normality.

#### Q13: Explain Winsorization.
A: Replace values beyond Q3+1.5×IQR with Q3+1.5×IQR. Preserves data count but reduces extreme values.

#### Q14: What does negative skewness indicate?
A: Tail on left side, Mode > Median > Mean. More extreme values on left (like ceiling effect in tests).

#### Q15: When is transformation inappropriate?
A: When interpretation is critical (e.g., dollar amounts), deletion/capping better than log.

#### Q16: How many outliers are acceptable?
A: Generally <5% of data. If >5%, investigate if they're genuine or errors.

#### Q17: Compare Box-Cox with Square Root transformation.
A: Box-Cox: Optimal but complex, requires positive values. Sqrt: Simpler, weaker effect, less aggressive.

#### Q18: What happens if you cap outliers but don't document?
A: Risk of bias, hard to reproduce, misrepresentation of data, invalid conclusions.

#### Q19: Why is skewness important in regression?
A: Violates normality assumption, residuals become non-normal, confidence intervals invalid.

#### Q20: How to choose between multiple transformation methods?
A: Compare skewness values - lower is better. Validate with histogram and Q-Q plots.

#### Q21: Can negative values be log-transformed?
A: No. Need to shift first: log(X + |min| + 1) to make all values positive.

---

### PRACTICAL 3: DESCRIPTIVE STATISTICS

#### Q1: Define mean and when to use it.
A: Mean = sum of values / count. Use for symmetric, normally distributed data without extreme outliers.

#### Q2: When is median preferred over mean?
A: When data is skewed or has outliers - median is resistant (position-based) while mean is affected by extreme values.

#### Q3: Can mode be used for continuous data?
A: Rarely - continuous data has many unique values. Mode useful for categorical or discrete data.

#### Q4: Explain standard deviation conceptually.
A: Average distance of values from mean. Higher std = more spread, lower std = values clustered near mean.

#### Q5: What does 68-95-99.7 rule mean?
A: In normal distribution: 68% within ±1σ, 95% within ±2σ, 99.7% within ±3σ.

#### Q6: How do variance and std dev differ?
A: Variance is squared (hard to interpret), Std Dev is square root (same units as data, easier to interpret).

#### Q7: What is coefficient of variation used for?
A: Compare spread across variables with different units. Standardized measure: CV = (Std/Mean)×100%.

#### Q8: Explain quartiles and IQR.
A: Q1=25th percentile, Q2=median, Q3=75th percentile. IQR = Q3-Q1, middle 50% of data.

#### Q9: What indicates right skew?
A: Mean > Median, tail extends right. Positive skewness coefficient. Example: income distribution.

#### Q10: How to interpret kurtosis > 0?
A: Leptokurtic distribution - heavy tails, more peaked center. More extreme values than normal distribution.

#### Q11: Why group data by categories?
A: Identify patterns within groups, compare groups, segment analysis, understand group-specific behavior.

#### Q12: What is the relationship between mean, median, mode in symmetric distribution?
A: All three equal: Mean = Median = Mode. Distribution perfectly balanced.

#### Q13: Can you have negative variance?
A: No, variance is always ≥ 0 (sum of squared deviations). Zero means all values identical.

#### Q14: How many modes can dataset have?
A: Can have no mode, one mode (unimodal), or multiple modes (bimodal, multimodal).

#### Q15: What does range not tell you about data?
A: Doesn't show distribution pattern, affected by outliers, ignores values between min and max.

#### Q16: Explain 90th percentile practically.
A: 90% of data below this value, 10% above. Used for benchmarking, performance analysis.

#### Q17: Why is grouped analysis important?
A: Reveals within-group patterns masked in overall data. Example: Iris setosa different from virginica.

#### Q18: What's the effect of adding constant to all values?
A: Adds same constant to mean, median, doesn't affect std dev or variance (unchanged spread).

#### Q19: How does outlier affect mean vs median?
A: Mean: Pulled dramatically toward outlier. Median: Minimal effect (position-based).

#### Q20: Explain negative skew with example.
A: Tail on left, Mean < Median. Example: Test scores where many score high, few score low.

---

### PRACTICAL 4: LINEAR REGRESSION

#### Q1: What is linear regression used for?
A: Predict continuous values based on one or more features using linear relationship.

#### Q2: Explain the difference between univariate and multivariate regression.
A: Univariate: one feature. Multivariate: multiple features.

#### Q3: What does intercept (β₀) represent?
A: Value of Y when all X=0. Where regression line crosses Y-axis.

#### Q4: How to interpret coefficient β₁ = 2.5?
A: For each unit increase in X, Y increases by 2.5 units (holding others constant).

#### Q5: Why split data into training and testing?
A: Prevent overfitting, get realistic performance estimate on unseen data.

#### Q6: What ratio should training/testing be?
A: Typically 70-80% training, 20-30% testing. More data = better training.

#### Q7: Explain MSE vs RMSE.
A: MSE: squared error (hard interpret). RMSE: square root (same units as Y, easier interpret).

#### Q8: When is R² = 1?
A: Perfect fit - model explains 100% of variance. All actual = predicted.

#### Q9: Is R² = 0.5 good?
A: Depends on context - explains 50% variance. Usually acceptable for complex data.

#### Q10: Why normalize features before regression?
A: Features on different scales - prevents large-scale features from dominating.

#### Q11: What does residual mean?
A: Difference between actual and predicted: Residual = Y_actual - Y_predicted.

#### Q12: How to detect overfitting?
A: Training R² much higher than testing R². Model memorizes training, fails on test.

#### Q13: Can negative R² occur?
A: Yes, if model worse than predicting mean Y. Indicates very poor model.

#### Q14: What assumptions does linear regression make?
A: (1) Linearity, (2) No multicollinearity, (3) Homoscedasticity, (4) Normal residuals.

#### Q15: How to improve poor model?
A: (1) Add features, (2) Remove outliers, (3) Feature engineering, (4) Regularization.

#### Q16: What's multicollinearity problem?
A: Features highly correlated with each other - makes coefficients unstable, hard interpret.

#### Q17: Can you predict outside data range?
A: Yes, but risky (extrapolation) - assumptions may not hold beyond observed range.

#### Q18: Why use StandardScaler before regression?
A: Makes features comparable scale, improves convergence, easier interpretation.

#### Q19: What does high residual indicate?
A: Prediction far from actual - outlier or model doesn't fit that point well.

#### Q20: Explain train-test gap in R².
A: If gap small: good generalization. If gap large: possible overfitting.

---

### PRACTICAL 5: LOGISTIC REGRESSION

#### Q1: What does logistic regression predict?
A: Probability of binary outcome (0/1), not continuous value like linear regression.

#### Q2: Explain sigmoid function.
A: Transforms any value to [0,1]. S-shaped curve. At z=0, P=0.5.

#### Q3: What's decision boundary?
A: Threshold (usually 0.5) - if P≥0.5, predict 1; if P<0.5, predict 0.

#### Q4: Define TP, FP, TN, FN.
A: TP=correct 1, FP=wrong 1, TN=correct 0, FN=wrong 0.

#### Q5: What is accuracy?
A: (TP+TN)/Total. Percentage of correct predictions.

#### Q6: When is precision important?
A: When false positives costly (spam detection - don't flag legitimate emails).

#### Q7: When is recall important?
A: When false negatives costly (disease detection - don't miss patients).

#### Q8: What's relationship between precision and recall?
A: Usually inverse - increasing one decreases other. F1 balances both.

#### Q9: Explain AUC-ROC.
A: Area under ROC curve. 0.5=random, 1.0=perfect, >0.7=good.

#### Q10: Why normalize before logistic regression?
A: Features on different scales - normalization makes optimization stable.

#### Q11: Can you use decision threshold other than 0.5?
A: Yes - adjust trade-off between precision/recall based on requirements.

#### Q12: What if dataset imbalanced?
A: Use F1, precision, recall instead of accuracy. Consider class weights.

#### Q13: How to handle multiclass classification?
A: One-vs-Rest approach - create multiple binary classifiers.

#### Q14: What assumptions does logistic regression make?
A: (1) Binary outcome, (2) Linear relationship log-odds, (3) Independence.

#### Q15: How does logistic regression differ from linear regression?
A: Linear: continuous output. Logistic: probability output (0-1).

---

### PRACTICAL 6: NAIVE BAYES

#### Q1: What does Naive Bayes assume?
A: Features are conditionally independent given the class (rarely true, but works).

#### Q2: Explain Bayes' theorem in context of classification.
A: P(Class|Features) = P(Features|Class)×P(Class) / P(Features). Gives posterior probability of class.

#### Q3: What is prior probability?
A: P(Y) - initial probability of class before seeing features. Calculated from training data.

#### Q4: What is likelihood?
A: P(X|Y) - probability of features given class. Estimated from training data.

#### Q5: Why is it called "Naive"?
A: Because it assumes feature independence, which is usually false but simplifies calculation.

#### Q6: What distribution does GaussianNB assume?
A: Gaussian (normal) distribution for each feature in each class.

#### Q7: How does training work in Naive Bayes?
A: Calculate mean and variance for each feature in each class from training data.

#### Q8: How does prediction work?
A: For new sample, calculate P(feature|class) for each feature, multiply by P(class).

#### Q9: Advantages of Naive Bayes?
A: Fast, simple, good baseline, works with small data, probabilistic.

#### Q10: When to use Naive Bayes?
A: Baseline classifier, text classification, spam detection, when interpretability important.

---

### PRACTICAL 7: TEXT ANALYTICS TF-IDF

#### Q1: What does TF-IDF stand for?
A: Term Frequency-Inverse Document Frequency. Measures word importance in document.

#### Q2: Why remove stop words?
A: Common words (the, a, is) appear everywhere, don't help classification.

#### Q3: What is stemming?
A: Reduce words to root form (running→run, cats→cat).

#### Q4: TF high, IDF low → importance?
A: Word appears often in document but in many documents → not distinctive.

#### Q5: How does TF-IDF differ from word count?
A: Word count: raw frequency. TF-IDF: normalized, considers document frequency.

#### Q6: What is cosine similarity?
A: Measures angle between vectors [0,1]. 1=identical, 0=orthogonal.

#### Q7: Why normalize TF-IDF?
A: Long documents get higher values. Normalization makes scores comparable.

#### Q8: How does dimensionality reduction help?
A: Reduces noise, finds main topics, speeds up computation.

#### Q9: Can you use TF-IDF for recommendation?
A: Yes - compute similarity between documents/users, recommend similar items.

#### Q10: Limitations of TF-IDF?
A: Ignores word order, semantics, context. Bag-of-words approach.

---

### PRACTICAL 8: DATA VISUALIZATION I

#### Q1: When to use histogram vs KDE plot?
A: Histogram: discrete data, exact frequencies. KDE: smooth distribution, continuous data.

#### Q2: What does boxplot show?
A: Q1, median, Q3, whiskers, outliers. Shows quartiles and spread.

#### Q3: What is a scatter plot used for?
A: Show relationship between two continuous variables. Identify patterns/clusters.

#### Q4: When to use bar chart?
A: Compare categories. Discrete groups or aggregated values.

#### Q5: How to color code scatter plot?
A: Use `c` parameter with values or categories. Add colorbar for scale.

#### Q6: What does heatmap show?
A: Correlation matrix. Color intensity shows correlation strength.

#### Q7: Difference between seaborn and matplotlib?
A: Seaborn: high-level, statistical. Matplotlib: low-level, customizable.

#### Q8: How to add title and labels?
A: `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `ax.set_title()`.

#### Q9: When to use violin plot?
A: Show full distribution shape. Better than box plot for multimodal distributions.

#### Q10: Best practices for visualization?
A: Clear titles, labels, legends. Appropriate color schemes. Limit visual clutter.

---

### PRACTICAL 9: CLUSTERING

#### Q1: How does K-Means algorithm work?
A: (1) Initialize centroids, (2) Assign points to nearest centroid, (3) Update centroids, (4) Repeat until convergence.

#### Q2: What does WCSS minimize?
A: Within-cluster sum of squares. Minimizes variance within clusters.

#### Q3: What is elbow method?
A: Plot WCSS vs k. Choose k at "elbow" (diminishing returns point).

#### Q4: How to determine optimal k?
A: Elbow method, silhouette score, domain knowledge, or try multiple k values.

#### Q5: What is silhouette score?
A: Measure of cluster quality [-1,1]. Higher = better separated clusters.

#### Q6: K-Means disadvantages?
A: Requires specifying k, random initialization, assumes similar-sized clusters.

#### Q7: Can you compare clusters to true labels?
A: Yes, use confusion matrix, purity, normalized mutual information.

#### Q8: What is k-means++ initialization?
A: Initialize centroids far apart. Better than random, faster convergence.

#### Q9: How many times to run K-Means?
A: Multiple times (n_init parameter) - random initialization, take best result.

#### Q10: When to use K-Means?
A: When need to find natural groupings in unlabeled data.

---

### PRACTICAL 10: ASSOCIATION RULE MINING

#### Q1: What is support?
A: Fraction of transactions containing itemset. Measures frequency.

#### Q2: What is confidence?
A: P(Y|X). Percentage of X buyers who also buy Y.

#### Q3: What is lift?
A: Ratio of confidence to expected confidence if independent. >1 means positive association.

#### Q4: How does Apriori work?
A: Generates frequent itemsets bottom-up. If itemset infrequent, supersets are too.

#### Q5: What's minimum support?
A: Threshold - only keep itemsets appearing ≥ min_support fraction of transactions.

#### Q6: Lift > 1 means what?
A: X and Y positively associated. Buying X increases chance of buying Y.

#### Q7: How to find best rules?
A: Filter by high confidence and lift. Sort by lift to find strongest associations.

#### Q8: Business uses?
A: Cross-selling, product placement, recommendation systems, campaign targeting.

#### Q9: What's confidence paradox?
A: High confidence doesn't mean strong association (could be due to high item support).

#### Q10: How to prevent spurious rules?
A: Use lift to confirm actual association, not just co-occurrence frequency.

---

### PRACTICAL 11: WEB SCRAPING

#### Q1: Why check robots.txt?
A: Check what scraping is allowed. robots.txt defines scraping rules for bots.

#### Q2: What's difference between static and dynamic scraping?
A: Static: BeautifulSoup (pre-rendered HTML). Dynamic: Selenium (JavaScript execution).

#### Q3: How to handle timeouts?
A: Set timeout parameter, implement retry logic, use exceptions handling.

#### Q4: What are CSS selectors?
A: Query language for selecting HTML elements. Examples: `.class`, `#id`, `tag.class`.

#### Q5: Why add delays between requests?
A: Respect server resources, avoid overloading, prevent IP blocking.

#### Q6: How to extract data from JSON in HTML?
A: Use json.loads() to parse JSON string embedded in HTML.

#### Q7: What's User-Agent header?
A: Identifies client/bot to server. Some sites block non-browser agents.

#### Q8: How to validate scraped data?
A: Check for missing values, duplicates, data types, value ranges.

#### Q9: Legal issues with scraping?
A: Check ToS, respect copyright, don't scrape personal data, follow GDPR.

#### Q10: How to scale scraping?
A: Use Scrapy framework, distributed crawling, database storage, caching.

---

## 📚 Viva Questions by Syllabus Topic

### Data Wrangling & Preprocessing (Practicals 1-2)
- **Topic: Missing Values**
  - [Practical 1, Q1-Q3, Q18-Q20, Q26, Q29-Q30](#practical-1-data-wrangling-i)
  - [Practical 2, Q9](#practical-2-data-wrangling-ii)

- **Topic: Normalization & Standardization**
  - [Practical 1, Q4-Q6, Q10-Q12, Q16-Q17, Q21-Q22](#practical-1-data-wrangling-i)

- **Topic: Categorical Encoding**
  - [Practical 1, Q7-Q9, Q13-Q14, Q23-Q24](#practical-1-data-wrangling-i)

- **Topic: Outlier Detection & Handling**
  - [Practical 2, Q1-Q4, Q11-Q13, Q16-Q18, Q21](#practical-2-data-wrangling-ii)

- **Topic: Data Transformation**
  - [Practical 2, Q5-Q10, Q14-Q15, Q19-Q20](#practical-2-data-wrangling-ii)

### Statistical Analysis (Practical 3)
- **Topic: Central Tendency**
  - [Practical 3, Q1-Q3, Q12, Q18-Q20](#practical-3-descriptive-statistics)

- **Topic: Dispersion Measures**
  - [Practical 3, Q4-Q7, Q9, Q10, Q13-Q15](#practical-3-descriptive-statistics)

- **Topic: Quartiles & Percentiles**
  - [Practical 3, Q8, Q16](#practical-3-descriptive-statistics)

- **Topic: Distribution & Skewness**
  - [Practical 3, Q5, Q9-Q10, Q20](#practical-3-descriptive-statistics)

- **Topic: Grouped Analysis**
  - [Practical 3, Q11, Q17](#practical-3-descriptive-statistics)

### Machine Learning Models (Practicals 4-6)
- **Topic: Linear Regression**
  - [Practical 4, Q1-Q20](#practical-4-linear-regression)

- **Topic: Logistic Regression & Classification Metrics**
  - [Practical 5, Q1-Q15](#practical-5-logistic-regression)

- **Topic: Naive Bayes Classification**
  - [Practical 6, Q1-Q10](#practical-6-naive-bayes)

### Advanced Techniques (Practicals 7-11)
- **Topic: Text Analytics & NLP**
  - [Practical 7, Q1-Q10](#practical-7-text-analytics-tfidf)

- **Topic: Data Visualization**
  - [Practical 8, Q1-Q10](#practical-8-data-visualization-i)

- **Topic: Clustering**
  - [Practical 9, Q1-Q10](#practical-9-clustering)

- **Topic: Association Rules & Market Basket Analysis**
  - [Practical 10, Q1-Q10](#practical-10-association-rule-mining)

- **Topic: Web Scraping & Data Collection**
  - [Practical 11, Q1-Q10](#practical-11-web-scraping)

---

## 🎓 Viva Preparation Guide

### Study Strategy (4-Week Plan)

**Week 1: Foundation (Practicals 1-3)**
- Day 1-2: Data Wrangling I (missing values, encoding)
- Day 3-4: Data Wrangling II (outliers, transformations)
- Day 5-7: Descriptive Statistics (mean, median, std, percentiles)

**Week 2: Regression Models (Practicals 4-5)**
- Day 1-3: Linear Regression (MSE, RMSE, R²)
- Day 4-7: Logistic Regression (confusion matrix, metrics)

**Week 3: Classification & Text (Practicals 6-7)**
- Day 1-3: Naive Bayes (Bayes theorem, probability)
- Day 4-7: Text Analytics (TF-IDF, tokenization, stemming)

**Week 4: Visualization & Advanced (Practicals 8-11)**
- Day 1-2: Data Visualization (plots, charts, seaborn)
- Day 3-4: Clustering (K-Means, elbow method)
- Day 5-6: Association Rules (Apriori, support, lift)
- Day 7: Web Scraping (BeautifulSoup, ethics)

### Key Concepts to Master

**Top 10 Critical Topics for Viva:**
1. ✅ Data Wrangling techniques (missing values, encoding)
2. ✅ Normalization vs Standardization
3. ✅ Confusion Matrix & Classification Metrics
4. ✅ Linear & Logistic Regression
5. ✅ Mean, Median, Std Dev, Percentiles
6. ✅ Train-Test Split & Overfitting
7. ✅ Outlier Detection (IQR, Z-score)
8. ✅ TF-IDF & Text Analytics
9. ✅ K-Means Clustering & Elbow Method
10. ✅ Apriori & Association Rules

### Common Viva Questions Pattern

**Type 1: Definitions**
- "What is X?" → Brief definition + example
- "Define X" → Formal definition + key properties

**Type 2: Comparisons**
- "Difference between X and Y?" → Compare contrasts
- "When to use X vs Y?" → Context and use cases

**Type 3: Formulas**
- "Explain formula..." → Derivation + interpretation
- "Calculate..." → Step-by-step solution

**Type 4: Applications**
- "Why is X important?" → Business value + impact
- "How to use X?" → Implementation steps

**Type 5: Troubleshooting**
- "What if..." → Problem + solution approach
- "How to fix..." → Debugging strategy

---

## 💡 External Examination Tips

### Before Examination (1 Week)
1. ✅ Review all 330 viva questions
2. ✅ Practice explaining concepts without notes
3. ✅ Write formulas from memory
4. ✅ Code key algorithms (logistic regression, K-Means)
5. ✅ Discuss with peers
6. ✅ Mock viva with instructor

### During Examination
1. ✅ **Listen Carefully:** Understand question fully before answering
2. ✅ **Structure Answers:** Definition → Explanation → Example
3. ✅ **Use Formulas:** When applicable, write equations
4. ✅ **Provide Examples:** Concrete examples strengthen answers
5. ✅ **Be Confident:** Speak clearly, maintain eye contact
6. ✅ **Ask for Clarification:** If unclear, ask examiner politely
7. ✅ **Don't Guess:** If unsure, say "I'm not certain, but..."
8. ✅ **Show Reasoning:** Explain your thought process

### Answer Quality Checklist
- [ ] Definition is clear and concise
- [ ] Explanation shows understanding
- [ ] Example is relevant and well-explained
- [ ] Formula is correct (if applicable)
- [ ] Connection to practical work mentioned
- [ ] Terminology used correctly
- [ ] Answer directly addresses question

### High-Scoring Answers Have:
✅ **Clear Structure:** Opening statement + detailed explanation + closing example  
✅ **Technical Accuracy:** Correct terminology and formulas  
✅ **Practical Connection:** Link to datasets used (Iris, Boston, Titanic)  
✅ **Code Context:** Reference Python functions and libraries  
✅ **Mathematical Foundation:** Explain formulas where applicable  
✅ **Real-World Application:** Show business/practical value  

---

## 📁 How to Use This Repository

### For Journal Submission:
```bash
1. Navigate to each practical folder (1_Data_Wrangling_I, etc.)
2. Open complete_practical_guide.md
3. Copy theory section → Journal theory part
4. Copy algorithms → Journal algorithms
5. Reference code implementation → Jupyter notebooks
6. Include viva questions → Use VIVA_QUESTIONS.md
```

### For Viva Preparation:
```bash
1. Read README.md (this file) → Understand topics
2. Review VIVA_QUESTIONS.md → Standalone viva questions
3. Study complete_practical_guide.md in each folder → Deep learning
4. Practice answering questions → Mock viva
5. Reference Jupyter notebooks → Code examples
```

### For External Examination:
```bash
1. Prepare 2-3 minute explanation per practical
2. Memorize key formulas and definitions
3. Practice code implementation
4. Prepare to answer questions from all 11 practicals
5. Review confusion matrix, R², MSE, RMSE, support/confidence/lift
```

---

## 🔗 File Structure

| File | Purpose |
|------|---------|
| `README.md` | Overview & viva questions index (You are here) |
| `VIVA_QUESTIONS.md` | Standalone viva questions only |
| `*/complete_practical_guide.md` | Complete theory, code, examples, viva for each practical |
| `*/XX_Practical_Name.ipynb` | Jupyter notebook with working code |
| `*/dataset.csv` | Dataset used in practical |

---

## 📊 Quick Reference: Formula Sheet

### Normalization
$$X_{normalized} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

### Standardization (Z-score)
$$X_{standardized} = \frac{X - \mu}{\sigma}$$

### Mean
$$\bar{X} = \frac{\sum_{i=1}^{n} X_i}{n}$$

### Standard Deviation
$$\sigma = \sqrt{\frac{\sum_{i=1}^{n} (X_i - \bar{X})^2}{n}}$$

### Linear Regression
$$y = \beta_0 + \beta_1 x + \epsilon$$

### MSE
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2$$

### RMSE
$$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2}$$

### R² (Coefficient of Determination)
$$R^2 = 1 - \frac{SS_E}{SS_T} = \frac{SS_R}{SS_T}$$

### Sigmoid Function
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### IQR Outlier Bounds
$$\text{Lower Bound} = Q_1 - 1.5 \times IQR$$
$$\text{Upper Bound} = Q_3 + 1.5 \times IQR$$

### Z-Score
$$Z = \frac{X - \mu}{\sigma}$$

### Bayes' Theorem
$$P(C|X) = \frac{P(X|C) \times P(C)}{P(X)}$$

### TF (Term Frequency)
$$TF(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total words in } d}$$

### IDF (Inverse Document Frequency)
$$IDF(t) = \log\left(\frac{N}{n_t}\right)$$

### TF-IDF
$$TF\text{-}IDF(t, d) = TF(t, d) \times IDF(t)$$

### Support (Association Rules)
$$\text{Support}(X) = \frac{\text{Transactions containing } X}{\text{Total transactions}}$$

### Confidence (Association Rules)
$$\text{Confidence}(X \rightarrow Y) = \frac{\text{Support}(X \cap Y)}{\text{Support}(X)}$$

### Lift (Association Rules)
$$\text{Lift}(X \rightarrow Y) = \frac{\text{Confidence}(X \rightarrow Y)}{\text{Support}(Y)}$$

---

## 👨‍🎓 External Examination Expectations

**Duration:** Typically 15-30 minutes per student

**Question Pattern:**
- **Q1:** Practical-specific (code, datasets, techniques)
- **Q2:** Conceptual (why, when, how)
- **Q3:** Formula or calculation
- **Q4:** Comparison or application
- **Q5:** Troubleshooting or extension

**Scoring Criteria:**
- ✅ Correctness (50%)
- ✅ Clarity & Communication (20%)
- ✅ Practical Understanding (15%)
- ✅ Code Knowledge (10%)
- ✅ Confidence (5%)

**Excellent Answer (9-10/10):**
- Complete, accurate definition
- Multiple relevant examples
- Correct formulas/code syntax
- Shows practical application
- Explains underlying concept

**Good Answer (7-8/10):**
- Accurate definition
- One-two examples
- Correct terminology
- Shows understanding

**Average Answer (5-6/10):**
- Basic definition
- Limited examples
- Some terminology errors
- Partial understanding

---

## 📞 Support & Resources

### Additional Learning:
- 📚 Each practical folder contains `complete_practical_guide.md` with extended theory
- 💻 Jupyter notebooks include commented code and explanations
- 📊 Datasets available in respective folders
- 🔗 Links to official documentation in each guide

### Key Libraries:
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning
- **matplotlib** - Visualization
- **seaborn** - Statistical visualization
- **nltk** - NLP & text processing

---

## ✅ Checklist Before Viva

- [ ] Read all README content (this file)
- [ ] Review VIVA_QUESTIONS.md multiple times
- [ ] Study each practical's complete_practical_guide.md
- [ ] Practice explaining concepts aloud
- [ ] Write formulas from memory
- [ ] Code key algorithms without reference
- [ ] Prepare examples for each practical
- [ ] Mock viva with peers
- [ ] Review confusion matrix calculation
- [ ] Remember key dataset names (Iris, Boston, Titanic, etc.)
- [ ] Practice confusion matrix metrics (TP, FP, TN, FN, accuracy, precision, recall)
- [ ] Memorize K-Means steps
- [ ] Know TF-IDF calculation
- [ ] Understand Apriori algorithm
- [ ] Be ready for "why" and "when" questions

---

## 📝 Final Note

This comprehensive guide is designed to make you **100% ready** for external viva examination. The questions cover all syllabus topics, real-world applications, and theoretical foundations. With thorough preparation using this guide, you'll be confident in your responses and demonstrate strong understanding of Data Science concepts.

**Good Luck with Your Viva! 🎓**

---

**Last Updated:** May 2026  
**Repository:** https://github.com/TejasTambe8080/TE-AI-DS-DS-LAB  
**Course:** Software Laboratory III (DSBDAL) | Subject Code: 317534  
**Institution:** Dr. D.Y. Patil Institute of Technology, Pune
