# PRACTICAL 8: DATA VISUALIZATION

## Practical Title
Data Visualization and Exploratory Data Analysis with Titanic Dataset

## Aim
To create effective visualizations for exploratory data analysis and communicate insights using Matplotlib and Seaborn.

## Objective
1. Create histograms, boxplots, scatter plots
2. Visualize relationships between variables
3. Compare distributions across groups
4. Create heatmaps for correlation analysis
5. Communicate findings through visualizations

## Problem Statement
Analyze Titanic dataset (891 passengers) to visualize survival patterns, demographics, and correlations using professional visualizations.

---

## THEORY

### 1. Types of Visualizations

**Distribution (Single Variable):**
- Histogram: Frequency distribution
- KDE Plot: Smooth distribution
- Box Plot: Quartiles, outliers

**Relationships (Two Variables):**
- Scatter Plot: X vs Y relationship
- Line Plot: Trends over time
- Heatmap: Correlation matrix

**Comparisons (Multiple Groups):**
- Bar Chart: Category comparison
- Violin Plot: Distribution + median
- Count Plot: Category frequencies

### 2. Best Practices
- Choose appropriate plot type
- Use clear labels and titles
- Limit colors (max 5-6)
- Include legend and grid
- Make title descriptive

### 3. Seaborn vs Matplotlib
- **Matplotlib:** Low-level, customizable
- **Seaborn:** High-level, statistical plots

---

## COMPLETE CODE

```python
# ===== PRACTICAL 8: DATA VISUALIZATION =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

print("=" * 80)
print("PRACTICAL 8: DATA VISUALIZATION - TITANIC DATASET ANALYSIS")
print("=" * 80)

# ===== STEP 1: LOAD DATA =====
print("\nSTEP 1: LOAD TITANIC DATASET")
print("-" * 80)

df = pd.read_csv('Titanic-Dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"Features: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nTarget variable (Survived):")
print(df['Survived'].value_counts())

# ===== STEP 2: DATA PREPROCESSING =====
print("\n" + "=" * 80)
print("STEP 2: DATA PREPROCESSING FOR VISUALIZATION")
print("-" * 80)

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Create categories
df['Survived_Label'] = df['Survived'].map({0: 'Did not Survive', 1: 'Survived'})
df['Sex_Label'] = df['Sex'].map({'male': 'Male', 'female': 'Female'})
df['Pclass_Label'] = df['Pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})

print(f"Data cleaned and preprocessed")
print(f"Missing values after preprocessing: {df.isnull().sum().sum()}")

# ===== STEP 3: DISTRIBUTION PLOTS =====
print("\n" + "=" * 80)
print("STEP 3: DISTRIBUTION VISUALIZATION")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram: Age
axes[0, 0].hist(df['Age'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Age (years)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Passenger Age', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Histogram: Fare
axes[0, 1].hist(df['Fare'], bins=30, color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Fare ($)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Passenger Fare', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Box Plot: Age by Survived
df.boxplot(column='Age', by='Survived_Label', ax=axes[1, 0])
axes[1, 0].set_xlabel('Survival Status')
axes[1, 0].set_ylabel('Age (years)')
axes[1, 0].set_title('Age Distribution by Survival Status', fontweight='bold')
axes[1, 0].get_figure().suptitle('')

# Box Plot: Fare by Class
df.boxplot(column='Fare', by='Pclass_Label', ax=axes[1, 1])
axes[1, 1].set_xlabel('Passenger Class')
axes[1, 1].set_ylabel('Fare ($)')
axes[1, 1].set_title('Fare Distribution by Passenger Class', fontweight='bold')
axes[1, 1].get_figure().suptitle('')

plt.tight_layout()
plt.savefig('titanic_distributions.png', dpi=150, bbox_inches='tight')
print("Distribution plots saved as 'titanic_distributions.png'")
plt.show()

# ===== STEP 4: CATEGORICAL ANALYSIS =====
print("\n" + "=" * 80)
print("STEP 4: CATEGORICAL ANALYSIS")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count Plot: Survival
survive_counts = df['Survived_Label'].value_counts()
axes[0].bar(survive_counts.index, survive_counts.values, color=['red', 'green'], alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Count')
axes[0].set_title('Survival Count', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(survive_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

# Count Plot: Gender
gender_counts = df['Sex_Label'].value_counts()
axes[1].bar(gender_counts.index, gender_counts.values, color=['blue', 'pink'], alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Count')
axes[1].set_title('Passenger Gender Distribution', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(gender_counts.values):
    axes[1].text(i, v + 10, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('titanic_categorical.png', dpi=150, bbox_inches='tight')
print("Categorical plots saved as 'titanic_categorical.png'")
plt.show()

# ===== STEP 5: GROUP COMPARISONS =====
print("\n" + "=" * 80)
print("STEP 5: GROUP COMPARISONS")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Survival by Gender
survival_gender = pd.crosstab(df['Sex_Label'], df['Survived_Label'])
survival_gender.plot(kind='bar', ax=axes[0, 0], color=['red', 'green'], alpha=0.7)
axes[0, 0].set_xlabel('Gender')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Survival by Gender', fontweight='bold')
axes[0, 0].legend(title='Status')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Survival by Class
survival_class = pd.crosstab(df['Pclass_Label'], df['Survived_Label'])
survival_class.plot(kind='bar', ax=axes[0, 1], color=['red', 'green'], alpha=0.7)
axes[0, 1].set_xlabel('Passenger Class')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Survival by Passenger Class', fontweight='bold')
axes[0, 1].legend(title='Status')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Scatter: Age vs Fare (colored by Survival)
scatter = axes[1, 0].scatter(df['Age'], df['Fare'], 
                            c=df['Survived'], cmap='RdYlGn', 
                            s=50, alpha=0.6, edgecolor='black')
axes[1, 0].set_xlabel('Age (years)')
axes[1, 0].set_ylabel('Fare ($)')
axes[1, 0].set_title('Age vs Fare (Colored by Survival)', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=axes[1, 0])
cbar.set_label('Survived')

# Violin Plot: Age by Survival and Gender
parts = axes[1, 1].violinplot([df[df['Survived']==0]['Age'].dropna(),
                               df[df['Survived']==1]['Age'].dropna()],
                             positions=[1, 2], showmeans=True, showmedians=True)
axes[1, 1].set_xticks([1, 2])
axes[1, 1].set_xticklabels(['Did not Survive', 'Survived'])
axes[1, 1].set_ylabel('Age (years)')
axes[1, 1].set_title('Age Distribution by Survival (Violin Plot)', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('titanic_comparisons.png', dpi=150, bbox_inches='tight')
print("Comparison plots saved as 'titanic_comparisons.png'")
plt.show()

# ===== STEP 6: CORRELATION ANALYSIS =====
print("\n" + "=" * 80)
print("STEP 6: CORRELATION ANALYSIS")
print("-" * 80)

# Select numeric columns
numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
corr_matrix = df[numeric_cols].corr()

print(f"\nCorrelation Matrix:")
print(corr_matrix.round(3))

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Correlation Matrix - Titanic Dataset', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('titanic_correlation.png', dpi=150, bbox_inches='tight')
print("Correlation heatmap saved as 'titanic_correlation.png'")
plt.show()

# ===== STEP 7: ADVANCED VISUALIZATIONS =====
print("\n" + "=" * 80)
print("STEP 7: ADVANCED VISUALIZATIONS")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# KDE Plot: Age by Survival
for survive in [0, 1]:
    data = df[df['Survived'] == survive]['Age'].dropna()
    data.plot.kde(ax=axes[0], label=['Did not Survive', 'Survived'][survive], linewidth=2)
axes[0].set_xlabel('Age (years)')
axes[0].set_ylabel('Density')
axes[0].set_title('Age Distribution by Survival (KDE)', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Hexbin Plot: Age vs Fare
hexbin = axes[1].hexbin(df['Age'], df['Fare'], gridsize=20, cmap='YlOrRd')
axes[1].set_xlabel('Age (years)')
axes[1].set_ylabel('Fare ($)')
axes[1].set_title('Age vs Fare (Hexbin Density)', fontweight='bold')
cbar = plt.colorbar(hexbin, ax=axes[1])
cbar.set_label('Count')

plt.tight_layout()
plt.savefig('titanic_advanced.png', dpi=150, bbox_inches='tight')
print("Advanced plots saved as 'titanic_advanced.png'")
plt.show()

# ===== STEP 8: SUMMARY STATISTICS =====
print("\n" + "=" * 80)
print("STEP 8: SUMMARY STATISTICS")
print("=" * 80)

print(f"\nSurvival Statistics:")
print(f"  Total passengers: {len(df)}")
print(f"  Survived: {(df['Survived']==1).sum()} ({(df['Survived']==1).sum()/len(df)*100:.1f}%)")
print(f"  Did not survive: {(df['Survived']==0).sum()} ({(df['Survived']==0).sum()/len(df)*100:.1f}%)")

print(f"\nAge Statistics:")
print(f"  Mean: {df['Age'].mean():.2f} years")
print(f"  Median: {df['Age'].median():.2f} years")
print(f"  Range: {df['Age'].min():.0f} - {df['Age'].max():.0f} years")

print(f"\nFare Statistics:")
print(f"  Mean: ${df['Fare'].mean():.2f}")
print(f"  Median: ${df['Fare'].median():.2f}")
print(f"  Range: ${df['Fare'].min():.0f} - ${df['Fare'].max():.0f}")

print(f"\nSurvival by Gender:")
print(df.groupby('Sex_Label')['Survived'].apply(lambda x: (x.sum()/len(x)*100)).round(1))

print(f"\n✓ DATA VISUALIZATION PRACTICAL COMPLETED!")
print("=" * 80)
```

---

## VISUALIZATION TYPES

| Type | Use Case | Function |
|------|----------|----------|
| Histogram | Distribution | `plt.hist()` |
| Box Plot | Quartiles, outliers | `plt.boxplot()` |
| Scatter | 2D relationship | `plt.scatter()` |
| Line | Trends | `plt.plot()` |
| Bar | Category comparison | `plt.bar()` |
| Heatmap | Correlation | `sns.heatmap()` |
| KDE | Smooth distribution | `data.plot.kde()` |
| Violin | Distribution shape | `plt.violinplot()` |

---

## EXTERNAL VIVA QUESTIONS (10 Key)

### Q1: When to use histogram vs KDE plot?
**A:** Histogram: discrete data, exact frequencies. KDE: smooth distribution, continuous data.

### Q2: What does boxplot show?
**A:** Q1, median, Q3, whiskers, outliers. Shows quartiles and spread.

### Q3: What is a scatter plot used for?
**A:** Show relationship between two continuous variables. Identify patterns/clusters.

### Q4: When to use bar chart?
**A:** Compare categories. Discrete groups or aggregated values.

### Q5: How to color code scatter plot?
**A:** Use `c` parameter with values or categories. Add colorbar for scale.

### Q6: What does heatmap show?
**A:** Correlation matrix. Color intensity shows correlation strength.

### Q7: Difference between seaborn and matplotlib?
**A:** Seaborn: high-level, statistical. Matplotlib: low-level, customizable.

### Q8: How to add title and labels?
**A:** `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `ax.set_title()`.

### Q9: When to use violin plot?
**A:** Show full distribution shape. Better than box plot for multimodal distributions.

### Q10: Best practices for visualization?
**A:** Clear titles, labels, legends. Appropriate color schemes. Limit visual clutter.

---

**Ready for External Examination and Journal Submission**
