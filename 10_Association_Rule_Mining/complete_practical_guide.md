# PRACTICAL 10: ASSOCIATION RULE MINING

## Practical Title
Market Basket Analysis using Apriori Algorithm and Association Rules

## Aim
To discover association rules and patterns in transactional data using Apriori algorithm and metrics like support, confidence, and lift.

## Objective
1. Understand support, confidence, and lift metrics
2. Implement Apriori algorithm
3. Generate association rules
4. Interpret rule strength and business impact
5. Visualize frequent itemsets

## Problem Statement
Analyze transaction data to find patterns: "If customer buys X, they likely buy Y". Useful for product recommendations and cross-selling.

---

## THEORY

### 1. Key Metrics

**Support:** Probability item appears in transaction
$$support(X) = \frac{\text{Transactions with X}}{\text{Total Transactions}}$$

**Confidence:** Conditional probability of Y given X
$$confidence(X \to Y) = \frac{support(X, Y)}{support(X)}$$

**Lift:** Strength of association
$$lift(X \to Y) = \frac{confidence(X \to Y)}{support(Y)}$$

Interpretation:
- Lift > 1: Positive association (buying X increases Y probability)
- Lift = 1: Independent
- Lift < 1: Negative association

### 2. Apriori Principle
If itemset is infrequent, all supersets are infrequent.

### 3. Association Rules
X → Y where:
- X (antecedent): Items we're checking
- Y (consequent): Item we predict
- confidence: How likely Y when X bought
- lift: How much more likely Y due to X

---

## COMPLETE CODE

```python
# ===== PRACTICAL 10: ASSOCIATION RULE MINING =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import defaultdict, Counter

print("=" * 80)
print("PRACTICAL 10: ASSOCIATION RULE MINING - MARKET BASKET ANALYSIS")
print("=" * 80)

# ===== STEP 1: CREATE TRANSACTION DATA =====
print("\nSTEP 1: PREPARE TRANSACTION DATA")
print("-" * 80)

# Sample transaction data (what items bought together)
transactions = [
    ['Milk', 'Bread', 'Butter', 'Cheese'],
    ['Bread', 'Butter', 'Coffee'],
    ['Milk', 'Bread', 'Cheese'],
    ['Milk', 'Coffee', 'Sugar'],
    ['Bread', 'Butter', 'Cheese', 'Milk'],
    ['Coffee', 'Sugar', 'Cream'],
    ['Milk', 'Cheese', 'Butter'],
    ['Bread', 'Cheese', 'Coffee'],
    ['Milk', 'Bread', 'Coffee', 'Sugar'],
    ['Butter', 'Cheese', 'Cream'],
    ['Milk', 'Butter', 'Sugar'],
    ['Bread', 'Milk', 'Cheese'],
    ['Coffee', 'Cream', 'Sugar'],
    ['Milk', 'Bread', 'Butter'],
    ['Cheese', 'Bread', 'Coffee']
]

print(f"Total transactions: {len(transactions)}")
print(f"\nSample transactions:")
for i, trans in enumerate(transactions[:5]):
    print(f"  T{i+1}: {trans}")

# ===== STEP 2: CALCULATE SUPPORT =====
print("\n" + "=" * 80)
print("STEP 2: CALCULATE SUPPORT FOR ITEMSETS")
print("-" * 80)

# Support for 1-itemsets
item_counts = defaultdict(int)
for transaction in transactions:
    for item in transaction:
        item_counts[item] += 1

total_transactions = len(transactions)
support_1itemset = {item: count / total_transactions 
                    for item, count in item_counts.items()}

print(f"\n1-Itemset Support (min support = 0.2):")
for item in sorted(support_1itemset, key=support_1itemset.get, reverse=True):
    if support_1itemset[item] >= 0.2:
        print(f"  {item}: {support_1itemset[item]:.2%}")

# Support for 2-itemsets
pairs = []
for transaction in transactions:
    if len(transaction) >= 2:
        for pair in combinations(sorted(transaction), 2):
            pairs.append(pair)

pair_counts = Counter(pairs)
support_2itemset = {pair: count / total_transactions 
                   for pair, count in pair_counts.items()
                   if count / total_transactions >= 0.2}

print(f"\n2-Itemset Support (min support = 0.2):")
for pair in sorted(support_2itemset, key=support_2itemset.get, reverse=True):
    print(f"  {pair[0]}, {pair[1]}: {support_2itemset[pair]:.2%}")

# ===== STEP 3: GENERATE ASSOCIATION RULES =====
print("\n" + "=" * 80)
print("STEP 3: GENERATE ASSOCIATION RULES")
print("-" * 80)

rules = []

for (item1, item2), support_xy in support_2itemset.items():
    support_x = support_1itemset[item1]
    support_y = support_1itemset[item2]
    
    # Confidence: P(Y|X)
    confidence_xy = support_xy / support_x
    
    # Lift: How much more likely Y due to X
    lift_xy = support_xy / (support_x * support_y)
    
    rules.append({
        'Antecedent': item1,
        'Consequent': item2,
        'Support': support_xy,
        'Confidence': confidence_xy,
        'Lift': lift_xy
    })
    
    # Reverse rule
    confidence_yx = support_xy / support_y
    lift_yx = support_xy / (support_x * support_y)
    
    rules.append({
        'Antecedent': item2,
        'Consequent': item1,
        'Support': support_xy,
        'Confidence': confidence_yx,
        'Lift': lift_yx
    })

rules_df = pd.DataFrame(rules)

print(f"\nTotal rules generated: {len(rules_df)}")
print(f"\nTop 10 Rules by Lift:")
top_rules = rules_df.nlargest(10, 'Lift')[['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift']]
for idx, row in top_rules.iterrows():
    print(f"  {row['Antecedent']} → {row['Consequent']}")
    print(f"    Support: {row['Support']:.2%}, Confidence: {row['Confidence']:.2%}, Lift: {row['Lift']:.2f}")

# ===== STEP 4: FILTER RULES =====
print("\n" + "=" * 80)
print("STEP 4: FILTER STRONG RULES")
print("-" * 80)

min_confidence = 0.4
min_lift = 1.2

strong_rules = rules_df[(rules_df['Confidence'] >= min_confidence) & 
                        (rules_df['Lift'] >= min_lift)]

print(f"Minimum Confidence: {min_confidence}")
print(f"Minimum Lift: {min_lift}")
print(f"Strong rules found: {len(strong_rules)}")

print(f"\nStrong Rules:")
for idx, row in strong_rules.iterrows():
    print(f"  {row['Antecedent']} → {row['Consequent']}")
    print(f"    Confidence: {row['Confidence']:.2%}, Lift: {row['Lift']:.2f}")

# ===== STEP 5: RULE INTERPRETATION =====
print("\n" + "=" * 80)
print("STEP 5: BUSINESS INTERPRETATION")
print("-" * 80)

print(f"\nInterpreting Rules:")
for idx, row in strong_rules.head(3).iterrows():
    antecedent = row['Antecedent']
    consequent = row['Consequent']
    confidence = row['Confidence']
    lift = row['Lift']
    
    print(f"\nRule: If buy {antecedent}, then buy {consequent}")
    print(f"  Confidence: {confidence:.1%} - {confidence*100:.0f}% of {antecedent} buyers also buy {consequent}")
    print(f"  Lift: {lift:.2f}x - {antecedent} buyers are {(lift-1)*100:.0f}% more likely to buy {consequent}")
    if lift > 1.5:
        print(f"  Action: Strong recommendation for cross-selling")
    elif lift > 1.2:
        print(f"  Action: Moderate recommendation for bundling")
    else:
        print(f"  Action: Consider for promotions")

# ===== STEP 6: VISUALIZATION =====
print("\n" + "=" * 80)
print("STEP 6: VISUALIZATION")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Support Distribution
axes[0, 0].hist(rules_df['Support'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(x=0.2, color='r', linestyle='--', linewidth=2, label='Min Support')
axes[0, 0].set_xlabel('Support')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Support', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Plot 2: Confidence vs Lift
scatter = axes[0, 1].scatter(rules_df['Confidence'], rules_df['Lift'], 
                            s=100, alpha=0.6, c=rules_df['Support'], 
                            cmap='viridis', edgecolor='black')
axes[0, 1].axhline(y=1, color='r', linestyle='--', linewidth=2, label='Lift=1 (Independent)')
axes[0, 1].axvline(x=min_confidence, color='g', linestyle='--', linewidth=2, label=f'Min Confidence={min_confidence}')
axes[0, 1].set_xlabel('Confidence')
axes[0, 1].set_ylabel('Lift')
axes[0, 1].set_title('Confidence vs Lift', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=axes[0, 1])
cbar.set_label('Support')

# Plot 3: Top Rules by Lift
top_10_lift = rules_df.nlargest(10, 'Lift')
rule_names = [f"{row['Antecedent']}→{row['Consequent']}" 
              for _, row in top_10_lift.iterrows()]
axes[1, 0].barh(rule_names, top_10_lift['Lift'], color='coral', edgecolor='black')
axes[1, 0].axvline(x=1, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Lift')
axes[1, 0].set_title('Top 10 Rules by Lift', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Plot 4: Support vs Confidence (sized by Lift)
top_rules_viz = rules_df.nlargest(15, 'Lift')
scatter2 = axes[1, 1].scatter(top_rules_viz['Support'], top_rules_viz['Confidence'],
                             s=top_rules_viz['Lift']*100, alpha=0.6, 
                             c=top_rules_viz['Lift'], cmap='RdYlGn', edgecolor='black')
axes[1, 1].set_xlabel('Support')
axes[1, 1].set_ylabel('Confidence')
axes[1, 1].set_title('Support vs Confidence (Size=Lift)', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
cbar2 = plt.colorbar(scatter2, ax=axes[1, 1])
cbar2.set_label('Lift')

plt.tight_layout()
plt.savefig('association_rules.png', dpi=150, bbox_inches='tight')
print("Visualizations saved as 'association_rules.png'")
plt.show()

# ===== STEP 7: SUMMARY =====
print("\n" + "=" * 80)
print("STEP 7: SUMMARY")
print("=" * 80)

print(f"\nASSOCIATION RULE MINING SUMMARY:")
print(f"  Total transactions analyzed: {len(transactions)}")
print(f"  Unique items: {len(support_1itemset)}")
print(f"  Frequent 2-itemsets: {len(support_2itemset)}")
print(f"  Total rules generated: {len(rules_df)}")
print(f"  Strong rules (conf≥{min_confidence}, lift≥{min_lift}): {len(strong_rules)}")

print(f"\nTop 3 Items by Support:")
for item, sup in sorted(support_1itemset.items(), 
                        key=lambda x: x[1], reverse=True)[:3]:
    print(f"  {item}: {sup:.2%}")

print(f"\n✓ ASSOCIATION RULE MINING PRACTICAL COMPLETED!")
print("=" * 80)
```

---

## KEY FORMULAS

### Support
$$support(X) = \frac{|T(X)|}{|T|}$$

### Confidence
$$confidence(X \to Y) = \frac{support(X, Y)}{support(X)}$$

### Lift
$$lift(X \to Y) = \frac{support(X, Y)}{support(X) \times support(Y)}$$

---

## EXTERNAL VIVA QUESTIONS (10 Key)

### Q1: What is support?
**A:** Fraction of transactions containing itemset. Measures frequency.

### Q2: What is confidence?
**A:** P(Y|X). Percentage of X buyers who also buy Y.

### Q3: What is lift?
**A:** Ratio of confidence to expected confidence if independent. >1 means positive association.

### Q4: How does Apriori work?
**A:** Generates frequent itemsets bottom-up. If itemset infrequent, supersets are too.

### Q5: What's minimum support?
**A:** Threshold - only keep itemsets appearing ≥ min_support fraction of transactions.

### Q6: Lift > 1 means what?
**A:** X and Y positively associated. Buying X increases chance of buying Y.

### Q7: How to find best rules?
**A:** Filter by high confidence and lift. Sort by lift to find strongest associations.

### Q8: Business uses?
**A:** Cross-selling, product placement, recommendation systems, campaign targeting.

### Q9: What's confidence paradox?
**A:** High confidence doesn't mean strong association (could be due to high item support).

### Q10: How to prevent spurious rules?
**A:** Use lift to confirm actual association, not just co-occurrence frequency.

---

**Ready for External Examination and Journal Submission**
