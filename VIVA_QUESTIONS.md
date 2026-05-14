# VIVA QUESTIONS - ALL PRACTICALS
## External Examination Question Bank
### DSBDAL - Software Laboratory III (Subject Code: 317534)

---

## PRACTICAL 1: DATA WRANGLING I (30 Questions)

**1. What is data wrangling and why is it important?**
Data wrangling is the process of transforming raw data into a clean, usable format. It's important because 60-80% of data science involves wrangling, and "garbage in = garbage out" - clean data is essential for accurate models.

**2. Name three types of missing data and explain MCAR.**
Three types: MCAR (Missing Completely At Random - no pattern), MAR (Missing At Random - depends on other variables), MNAR (Missing Not At Random - depends on unobserved values). MCAR is safest to handle as it has no pattern.

**3. When would you use deletion vs imputation for missing values?**
Use deletion when <5% of data is missing (acceptable information loss). Use imputation when 5-30% is missing (preserve data). If >30% missing, consider dropping the entire column.

**4. Explain Min-Max normalization with a formula and example.**
Formula: X_normalized = (X - X_min) / (X_max - X_min), scales to [0,1]. Example: If age 25 with min=20, max=80: (25-20)/(80-20) = 5/60 = 0.083

**5. What is the difference between normalization and standardization?**
Normalization (Min-Max) scales to [0,1] and is sensitive to outliers. Standardization (Z-score) centers at 0 with std=1 and is resistant to outliers.

**6. When would you use standardization over normalization?**
Use standardization when: (1) data has outliers, (2) using algorithms like SVM or linear regression, (3) data approximately normally distributed. Use Min-Max for bounded features.

**7. What is label encoding and when should you use it?**
Label encoding converts categorical values to integers (0, 1, 2...). Use it for: (1) target variables in classification, (2) tree-based models, (3) ordinal categories. Avoid for nominal categories in linear models.

**8. Why can't machine learning algorithms directly work with categorical data?**
Because algorithms perform mathematical operations (multiplication, addition) on numerical values. Text like "Red", "Green" can't be used in equations, so encoding converts them to numbers.

**9. Explain the difference between fit() and transform() in sklearn.**
fit() learns parameters (min, max, mean, std) from data. transform() applies those parameters. fit_transform() does both. Always fit on training data, then transform both train and test.

**10. What does MinMaxScaler.fit_transform() do in two steps?**
Step 1: fit() finds min and max values of each column. Step 2: transform() applies formula (X-min)/(max-min) to scale all values to [0,1].

**11. What is StandardScaler and what are the resulting mean and std?**
StandardScaler applies Z-score normalization: (X-mean)/std. Result: mean becomes ~0, standard deviation becomes ~1, allowing fair comparison of features.

**12. If you have 150 rows and apply fit_transform to X with shape (150, 4), what is output shape?**
Output shape is (150, 4). Same as input - only values change, not dimensions.

**13. How do you convert a sklearn numpy array output back to DataFrame with column names?**
Use `pd.DataFrame(array, columns=original_columns)` where original_columns is list of feature names.

**14. What is LabelEncoder.classes_ attribute used for?**
It shows the mapping of encoded labels. For example, if classes_ = ['Setosa', 'Versicolor', 'Virginica'], then Setosa→0, Versicolor→1, Virginica→2.

**15. Explain why outliers affect Min-Max scaling more than standardization.**
Min-Max uses min/max values - extreme outliers stretch the denominator, compressing all other values. Standardization uses mean/std - outliers have less dramatic effect on std.

**16. Can you apply normalization to only some features? How?**
Yes, select specific columns: `scaler.fit_transform(df[['age', 'income']])` - this normalizes only age and income, not other columns.

**17. What happens if you fit scaler on training data but test data has values outside [0,1]?**
Test values outside [0,1] indicate they're outside training range. Not necessarily wrong (extrapolation), but may indicate data shift or outliers in test set.

**18. How would you handle missing values in 'Income' column using mean imputation?**
`df['Income'].fillna(df['Income'].mean(), inplace=True)` - replaces NaN with column's mean value.

**19. What's the difference between fillna() and dropna()?**
fillna() replaces NaN with specified value (preserves rows). dropna() removes entire rows/columns with NaN (loses data). Choose based on how much data is missing.

**20. If you accidentally fit scaler on full data before splitting, what's the problem?**
Data leakage - test data statistics (min, max, mean) influence training scaler. Realistic performance estimates compromised. Always fit scaler ONLY on training data.

**21. How do you save a fitted scaler for later use?**
Use pickle or joblib: `import joblib; joblib.dump(scaler, 'scaler.pkl')` then load with `joblib.load('scaler.pkl')`

**22. When would you NOT normalize data?**
Tree-based models (Random Forest, XGBoost) don't require normalization. They're scale-invariant.

**23. Explain one-hot encoding vs label encoding.**
Label encoding: Nominal categories→integers (0,1,2). One-hot: Each category→binary column. Use one-hot for nominal, label for ordinal.

**24. Can you reverse label encoding?**
Yes, using inverse_transform: `original_labels = encoder.inverse_transform([0, 1, 2])`

**25. What does isnull().sum() do?**
Returns count of missing values per column.

**26. How to fill missing values differently for different columns?**
`df['col1'].fillna(method='ffill'); df['col2'].fillna(df['col2'].mean())`

**27. What's the purpose of checking data.dtypes?**
Verify data types are correct (int, float, object, datetime). Identify columns needing type conversion.

**28. Explain the GIGO principle in data science.**
Garbage In = Garbage Out. If input data is poor quality, output predictions will be unreliable regardless of algorithm quality.

**29. When should you use forward fill for missing values?**
For time-series data where values evolve over time. Previous value is good estimate for next.

**30. How to handle missing values in categorical columns?**
fillna(mode), fillna('Unknown'), or create binary 'Missing' indicator column. Never drop if >5% missing.

---

## PRACTICAL 2: DATA WRANGLING II (21 Questions)

**1. What is an outlier?**
A data point that lies far from other observations, vastly larger or smaller than remaining values.

**2. How does IQR method calculate outlier bounds?**
Lower Bound = Q1 - 1.5×IQR, Upper Bound = Q3 + 1.5×IQR. Values outside are outliers.

**3. Explain Z-score for outlier detection.**
Z = (X - mean) / std. |Z| > 3 = outlier. Shows how many standard deviations from mean.

**4. When to use deletion vs capping for outliers?**
Deletion when <1% outliers (acceptable loss). Capping preserves data but creates artificial values.

**5. What is right skew and give an example.**
Right skew: tail on right, Mean > Median. Example: Income distribution (most earn average, few earn millions).

**6. What does log transformation do?**
Converts exponential to linear, compresses large values, spreads small values, reduces right skew.

**7. Why normalize before applying Box-Cox?**
Box-Cox requires positive values. Transform: X_new = X + |min| + 1 to ensure X > 0.

**8. How does Box-Cox find optimal transformation?**
Tests different lambda values, finds one maximizing normality (highest likelihood). Automatic optimization.

**9. What's difference between Median and Mean imputation?**
Mean affected by outliers, pulls toward extreme values. Median is middle value, resistant to outliers.

**10. Can you reverse log transformation?**
Yes. If X_log = log(X), then X = e^(X_log). Called inverse transformation.

**11. What are the three approaches to handle outliers?**
(1) Deletion - remove rows, (2) Capping - replace with bounds, (3) Transformation - reduce spread.

**12. Which outlier detection method works best for non-normal data?**
IQR method - doesn't assume normal distribution. Z-score assumes normality.

**13. Explain Winsorization.**
Replace values beyond Q3+1.5×IQR with Q3+1.5×IQR. Preserves data count but reduces extreme values.

**14. What does negative skewness indicate?**
Tail on left side, Mode > Median > Mean. More extreme values on left (like ceiling effect in tests).

**15. When is transformation inappropriate?**
When interpretation is critical (e.g., dollar amounts), deletion/capping better than log.

**16. How many outliers are acceptable?**
Generally <5% of data. If >5%, investigate if they're genuine or errors.

**17. Compare Box-Cox with Square Root transformation.**
Box-Cox: Optimal but complex, requires positive values. Sqrt: Simpler, weaker effect, less aggressive.

**18. What happens if you cap outliers but don't document?**
Risk of bias, hard to reproduce, misrepresentation of data, invalid conclusions.

**19. Why is skewness important in regression?**
Violates normality assumption, residuals become non-normal, confidence intervals invalid.

**20. How to choose between multiple transformation methods?**
Compare skewness values - lower is better. Validate with histogram and Q-Q plots.

**21. Can negative values be log-transformed?**
No. Need to shift first: log(X + |min| + 1) to make all values positive.

---

## PRACTICAL 3: DESCRIPTIVE STATISTICS (20 Questions)

**1. Define mean and when to use it.**
Mean = sum of values / count. Use for symmetric, normally distributed data without extreme outliers.

**2. When is median preferred over mean?**
When data is skewed or has outliers - median is resistant (position-based) while mean is affected by extreme values.

**3. Can mode be used for continuous data?**
Rarely - continuous data has many unique values. Mode useful for categorical or discrete data.

**4. Explain standard deviation conceptually.**
Average distance of values from mean. Higher std = more spread, lower std = values clustered near mean.

**5. What does 68-95-99.7 rule mean?**
In normal distribution: 68% within ±1σ, 95% within ±2σ, 99.7% within ±3σ.

**6. How do variance and std dev differ?**
Variance is squared (hard to interpret), Std Dev is square root (same units as data, easier to interpret).

**7. What is coefficient of variation used for?**
Compare spread across variables with different units. Standardized measure: CV = (Std/Mean)×100%.

**8. Explain quartiles and IQR.**
Q1=25th percentile, Q2=median, Q3=75th percentile. IQR = Q3-Q1, middle 50% of data.

**9. What indicates right skew?**
Mean > Median, tail extends right. Positive skewness coefficient. Example: income distribution.

**10. How to interpret kurtosis > 0?**
Leptokurtic distribution - heavy tails, more peaked center. More extreme values than normal distribution.

**11. Why group data by categories?**
Identify patterns within groups, compare groups, segment analysis, understand group-specific behavior.

**12. What is the relationship between mean, median, mode in symmetric distribution?**
All three equal: Mean = Median = Mode. Distribution perfectly balanced.

**13. Can you have negative variance?**
No, variance is always ≥ 0 (sum of squared deviations). Zero means all values identical.

**14. How many modes can dataset have?**
Can have no mode, one mode (unimodal), or multiple modes (bimodal, multimodal).

**15. What does range not tell you about data?**
Doesn't show distribution pattern, affected by outliers, ignores values between min and max.

**16. Explain 90th percentile practically.**
90% of data below this value, 10% above. Used for benchmarking, performance analysis.

**17. Why is grouped analysis important?**
Reveals within-group patterns masked in overall data. Example: Iris setosa different from virginica.

**18. What's the effect of adding constant to all values?**
Adds same constant to mean, median, doesn't affect std dev or variance (unchanged spread).

**19. How does outlier affect mean vs median?**
Mean: Pulled dramatically toward outlier. Median: Minimal effect (position-based).

**20. Explain negative skew with example.**
Tail on left, Mean < Median. Example: Test scores where many score high, few score low.

---

## PRACTICAL 4: LINEAR REGRESSION (20 Questions)

**1. What is linear regression used for?**
Predict continuous values based on one or more features using linear relationship.

**2. Explain the difference between univariate and multivariate regression.**
Univariate: one feature. Multivariate: multiple features.

**3. What does intercept (β₀) represent?**
Value of Y when all X=0. Where regression line crosses Y-axis.

**4. How to interpret coefficient β₁ = 2.5?**
For each unit increase in X, Y increases by 2.5 units (holding others constant).

**5. Why split data into training and testing?**
Prevent overfitting, get realistic performance estimate on unseen data.

**6. What ratio should training/testing be?**
Typically 70-80% training, 20-30% testing. More data = better training.

**7. Explain MSE vs RMSE.**
MSE: squared error (hard interpret). RMSE: square root (same units as Y, easier interpret).

**8. When is R² = 1?**
Perfect fit - model explains 100% of variance. All actual = predicted.

**9. Is R² = 0.5 good?**
Depends on context - explains 50% variance. Usually acceptable for complex data.

**10. Why normalize features before regression?**
Features on different scales - prevents large-scale features from dominating.

**11. What does residual mean?**
Difference between actual and predicted: Residual = Y_actual - Y_predicted.

**12. How to detect overfitting?**
Training R² much higher than testing R². Model memorizes training, fails on test.

**13. Can negative R² occur?**
Yes, if model worse than predicting mean Y. Indicates very poor model.

**14. What assumptions does linear regression make?**
(1) Linearity, (2) No multicollinearity, (3) Homoscedasticity, (4) Normal residuals.

**15. How to improve poor model?**
(1) Add features, (2) Remove outliers, (3) Feature engineering, (4) Regularization.

**16. What's multicollinearity problem?**
Features highly correlated with each other - makes coefficients unstable, hard interpret.

**17. Can you predict outside data range?**
Yes, but risky (extrapolation) - assumptions may not hold beyond observed range.

**18. Why use StandardScaler before regression?**
Makes features comparable scale, improves convergence, easier interpretation.

**19. What does high residual indicate?**
Prediction far from actual - outlier or model doesn't fit that point well.

**20. Explain train-test gap in R².**
If gap small: good generalization. If gap large: possible overfitting.

---

## PRACTICAL 5: LOGISTIC REGRESSION (15 Questions)

**1. What does logistic regression predict?**
Probability of binary outcome (0/1), not continuous value like linear regression.

**2. Explain sigmoid function.**
Transforms any value to [0,1]. S-shaped curve. At z=0, P=0.5.

**3. What's decision boundary?**
Threshold (usually 0.5) - if P≥0.5, predict 1; if P<0.5, predict 0.

**4. Define TP, FP, TN, FN.**
TP=correct 1, FP=wrong 1, TN=correct 0, FN=wrong 0.

**5. What is accuracy?**
(TP+TN)/Total. Percentage of correct predictions.

**6. When is precision important?**
When false positives costly (spam detection - don't flag legitimate emails).

**7. When is recall important?**
When false negatives costly (disease detection - don't miss patients).

**8. What's relationship between precision and recall?**
Usually inverse - increasing one decreases other. F1 balances both.

**9. Explain AUC-ROC.**
Area under ROC curve. 0.5=random, 1.0=perfect, >0.7=good.

**10. Why normalize before logistic regression?**
Features on different scales - normalization makes optimization stable.

**11. Can you use decision threshold other than 0.5?**
Yes - adjust trade-off between precision/recall based on requirements.

**12. What if dataset imbalanced?**
Use F1, precision, recall instead of accuracy. Consider class weights.

**13. How to handle multiclass classification?**
One-vs-Rest approach - create multiple binary classifiers.

**14. What assumptions does logistic regression make?**
(1) Binary outcome, (2) Linear relationship log-odds, (3) Independence.

**15. How does logistic regression differ from linear regression?**
Linear: continuous output. Logistic: probability output (0-1).

---

## PRACTICAL 6: NAIVE BAYES (10 Questions)

**1. What does Naive Bayes assume?**
Features are conditionally independent given the class (rarely true, but works).

**2. Explain Bayes' theorem in context of classification.**
P(Class|Features) = P(Features|Class)×P(Class) / P(Features). Gives posterior probability of class.

**3. What is prior probability?**
P(Y) - initial probability of class before seeing features. Calculated from training data.

**4. What is likelihood?**
P(X|Y) - probability of features given class. Estimated from training data.

**5. Why is it called "Naive"?**
Because it assumes feature independence, which is usually false but simplifies calculation.

**6. What distribution does GaussianNB assume?**
Gaussian (normal) distribution for each feature in each class.

**7. How does training work in Naive Bayes?**
Calculate mean and variance for each feature in each class from training data.

**8. How does prediction work?**
For new sample, calculate P(feature|class) for each feature, multiply by P(class).

**9. Advantages of Naive Bayes?**
Fast, simple, good baseline, works with small data, probabilistic.

**10. When to use Naive Bayes?**
Baseline classifier, text classification, spam detection, when interpretability important.

---

## PRACTICAL 7: TEXT ANALYTICS TF-IDF (10 Questions)

**1. What does TF-IDF stand for?**
Term Frequency-Inverse Document Frequency. Measures word importance in document.

**2. Why remove stop words?**
Common words (the, a, is) appear everywhere, don't help classification.

**3. What is stemming?**
Reduce words to root form (running→run, cats→cat).

**4. TF high, IDF low → importance?**
Word appears often in document but in many documents → not distinctive.

**5. How does TF-IDF differ from word count?**
Word count: raw frequency. TF-IDF: normalized, considers document frequency.

**6. What is cosine similarity?**
Measures angle between vectors [0,1]. 1=identical, 0=orthogonal.

**7. Why normalize TF-IDF?**
Long documents get higher values. Normalization makes scores comparable.

**8. How does dimensionality reduction help?**
Reduces noise, finds main topics, speeds up computation.

**9. Can you use TF-IDF for recommendation?**
Yes - compute similarity between documents/users, recommend similar items.

**10. Limitations of TF-IDF?**
Ignores word order, semantics, context. Bag-of-words approach.

---

## PRACTICAL 8: DATA VISUALIZATION I (10 Questions)

**1. When to use histogram vs KDE plot?**
Histogram: discrete data, exact frequencies. KDE: smooth distribution, continuous data.

**2. What does boxplot show?**
Q1, median, Q3, whiskers, outliers. Shows quartiles and spread.

**3. What is a scatter plot used for?**
Show relationship between two continuous variables. Identify patterns/clusters.

**4. When to use bar chart?**
Compare categories. Discrete groups or aggregated values.

**5. How to color code scatter plot?**
Use `c` parameter with values or categories. Add colorbar for scale.

**6. What does heatmap show?**
Correlation matrix. Color intensity shows correlation strength.

**7. Difference between seaborn and matplotlib?**
Seaborn: high-level, statistical. Matplotlib: low-level, customizable.

**8. How to add title and labels?**
`plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `ax.set_title()`.

**9. When to use violin plot?**
Show full distribution shape. Better than box plot for multimodal distributions.

**10. Best practices for visualization?**
Clear titles, labels, legends. Appropriate color schemes. Limit visual clutter.

---

## PRACTICAL 9: CLUSTERING (10 Questions)

**1. How does K-Means algorithm work?**
(1) Initialize centroids, (2) Assign points to nearest centroid, (3) Update centroids, (4) Repeat until convergence.

**2. What does WCSS minimize?**
Within-cluster sum of squares. Minimizes variance within clusters.

**3. What is elbow method?**
Plot WCSS vs k. Choose k at "elbow" (diminishing returns point).

**4. How to determine optimal k?**
Elbow method, silhouette score, domain knowledge, or try multiple k values.

**5. What is silhouette score?**
Measure of cluster quality [-1,1]. Higher = better separated clusters.

**6. K-Means disadvantages?**
Requires specifying k, random initialization, assumes similar-sized clusters.

**7. Can you compare clusters to true labels?**
Yes, use confusion matrix, purity, normalized mutual information.

**8. What is k-means++ initialization?**
Initialize centroids far apart. Better than random, faster convergence.

**9. How many times to run K-Means?**
Multiple times (n_init parameter) - random initialization, take best result.

**10. When to use K-Means?**
When need to find natural groupings in unlabeled data.

---

## PRACTICAL 10: ASSOCIATION RULE MINING (10 Questions)

**1. What is support?**
Fraction of transactions containing itemset. Measures frequency.

**2. What is confidence?**
P(Y|X). Percentage of X buyers who also buy Y.

**3. What is lift?**
Ratio of confidence to expected confidence if independent. >1 means positive association.

**4. How does Apriori work?**
Generates frequent itemsets bottom-up. If itemset infrequent, supersets are too.

**5. What's minimum support?**
Threshold - only keep itemsets appearing ≥ min_support fraction of transactions.

**6. Lift > 1 means what?**
X and Y positively associated. Buying X increases chance of buying Y.

**7. How to find best rules?**
Filter by high confidence and lift. Sort by lift to find strongest associations.

**8. Business uses?**
Cross-selling, product placement, recommendation systems, campaign targeting.

**9. What's confidence paradox?**
High confidence doesn't mean strong association (could be due to high item support).

**10. How to prevent spurious rules?**
Use lift to confirm actual association, not just co-occurrence frequency.

---

## PRACTICAL 11: WEB SCRAPING (10 Questions)

**1. Why check robots.txt?**
Check what scraping is allowed. robots.txt defines scraping rules for bots.

**2. What's difference between static and dynamic scraping?**
Static: BeautifulSoup (pre-rendered HTML). Dynamic: Selenium (JavaScript execution).

**3. How to handle timeouts?**
Set timeout parameter, implement retry logic, use exceptions handling.

**4. What are CSS selectors?**
Query language for selecting HTML elements. Examples: `.class`, `#id`, `tag.class`.

**5. Why add delays between requests?**
Respect server resources, avoid overloading, prevent IP blocking.

**6. How to extract data from JSON in HTML?**
Use json.loads() to parse JSON string embedded in HTML.

**7. What's User-Agent header?**
Identifies client/bot to server. Some sites block non-browser agents.

**8. How to validate scraped data?**
Check for missing values, duplicates, data types, value ranges.

**9. Legal issues with scraping?**
Check ToS, respect copyright, don't scrape personal data, follow GDPR.

**10. How to scale scraping?**
Use Scrapy framework, distributed crawling, database storage, caching.

---

## 📊 QUESTION SUMMARY

| Practical | Title | Questions | Total |
|-----------|-------|-----------|-------|
| 1 | Data Wrangling I | 30 | 30 |
| 2 | Data Wrangling II | 21 | 51 |
| 3 | Descriptive Statistics | 20 | 71 |
| 4 | Linear Regression | 20 | 91 |
| 5 | Logistic Regression | 15 | 106 |
| 6 | Naive Bayes | 10 | 116 |
| 7 | Text Analytics TF-IDF | 10 | 126 |
| 8 | Data Visualization I | 10 | 136 |
| 9 | Clustering | 10 | 146 |
| 10 | Association Rules | 10 | 156 |
| 11 | Web Scraping | 10 | 166 |

**Total Viva Questions: 166 Questions**

---

## 🎯 HOW TO USE THIS DOCUMENT

### For External Viva Preparation:
1. Read each question carefully
2. Try to answer from memory
3. Check your answer against the provided response
4. Note areas of weakness
5. Review weak topics using complete_practical_guide.md in each practical folder

### For Mock Viva:
1. Have someone ask questions randomly from this document
2. Time your responses (aim for 1-2 minutes per question)
3. Practice explaining clearly and concisely
4. Build confidence by repetition

### For Journal/Theory Part:
1. Use this as reference for viva questions section
2. Combine with theory from complete_practical_guide.md
3. Include key formulas and examples

---

## ✅ Viva Preparation Checklist

- [ ] Read each question
- [ ] Answer mentally without looking at responses
- [ ] Check your answer
- [ ] Repeat 2-3 times
- [ ] Take mock vivas
- [ ] Review areas you struggled with
- [ ] Practice explaining to others
- [ ] Time yourself on answers

**Good Luck! 🎓**

---

**Repository:** https://github.com/TejasTambe8080/TE-AI-DS-DS-LAB  
**Course:** Software Laboratory III (DSBDAL)  
**Institution:** Dr. D.Y. Patil Institute of Technology, Pune  
**Last Updated:** May 2026
