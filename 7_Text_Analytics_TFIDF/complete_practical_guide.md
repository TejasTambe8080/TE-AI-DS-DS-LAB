# PRACTICAL 7: TEXT ANALYTICS - TF-IDF

## Practical Title
Text Analytics and Natural Language Processing with TF-IDF Vectorization

## Aim
To preprocess text data and extract features using TF-IDF vectorization for text classification and analysis.

## Objective
1. Perform text preprocessing (tokenization, stop word removal, stemming)
2. Apply POS tagging for grammatical analysis
3. Calculate TF-IDF vectors
4. Reduce dimensionality with SVD
5. Classify documents using TF-IDF features

---

## THEORY

### 1. Text Preprocessing Steps
- **Tokenization:** Split text into words
- **Lowercase:** Normalize case
- **Remove Punctuation:** Clean special characters
- **Remove Stop Words:** Filter common words (the, a, is)
- **Stemming:** Reduce to root (running→run)
- **Lemmatization:** Normalize to base form

### 2. TF (Term Frequency)
$$TF(t,d) = \frac{\text{Count of term } t \text{ in document } d}{\text{Total words in document } d}$$

Measures how often word appears in document.

### 3. IDF (Inverse Document Frequency)
$$IDF(t) = \log\left(\frac{\text{Total documents}}{\text{Documents containing term } t}\right)$$

Measures how unique/important word is across all documents.

### 4. TF-IDF
$$TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)$$

High value: Important word in specific document
Low value: Common word across documents

### 5. POS Tagging
Grammatical role:
- NN: Noun
- VB: Verb
- JJ: Adjective
- RB: Adverb

---

## COMPLETE CODE

```python
# ===== PRACTICAL 7: TEXT ANALYTICS - TF-IDF =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, download

print("=" * 80)
print("PRACTICAL 7: TEXT ANALYTICS - TF-IDF VECTORIZATION")
print("=" * 80)

# Download NLTK resources
download('punkt', quiet=True)
download('averaged_perceptron_tagger', quiet=True)
download('wordnet', quiet=True)
download('stopwords', quiet=True)

# ===== STEP 1: SAMPLE TEXT DATA =====
print("\nSTEP 1: PREPARE TEXT DATA")
print("-" * 80)

documents = [
    "Machine learning is a subset of artificial intelligence that focuses on data",
    "Deep learning uses neural networks with multiple layers for pattern recognition",
    "Natural language processing enables computers to understand human language",
    "Data science combines statistics mathematics and programming for insights",
    "Artificial intelligence automates tasks and improves business processes",
    "Neural networks are inspired by biological neurons in the human brain",
    "Python is a popular programming language for data science and machine learning",
    "Feature engineering is crucial for building accurate predictive models",
    "Cross validation helps evaluate model generalization on unseen data",
    "Supervised learning requires labeled training data for model training"
]

labels = [0, 1, 2, 2, 0, 1, 0, 0, 0, 0]  # 0=ML, 1=DL, 2=NLP

df = pd.DataFrame({'text': documents, 'label': labels})
print(f"Total documents: {len(documents)}")
print(f"\nSample documents:")
for i, doc in enumerate(documents[:3]):
    print(f"  {i+1}. {doc[:60]}...")

# ===== STEP 2: TEXT PREPROCESSING =====
print("\n" + "=" * 80)
print("STEP 2: TEXT PREPROCESSING")
print("-" * 80)

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    
    return tokens

print("\nOriginal text:")
sample_text = documents[0]
print(f"  {sample_text}")

print(f"\nPreprocessed tokens:")
tokens = preprocess_text(sample_text)
print(f"  {tokens}")

# ===== STEP 3: POS TAGGING =====
print("\n" + "=" * 80)
print("STEP 3: PART-OF-SPEECH (POS) TAGGING")
print("-" * 80)

print(f"\nPOS Tags for: '{documents[0]}'")
tokens_pos = word_tokenize(documents[0].lower())
pos_tags = pos_tag(tokens_pos)
for token, tag in pos_tags[:10]:
    print(f"  {token:15} → {tag}")

# ===== STEP 4: TF-IDF VECTORIZATION =====
print("\n" + "=" * 80)
print("STEP 4: TF-IDF VECTORIZATION")
print("-" * 80)

vectorizer = TfidfVectorizer(max_features=20, lowercase=True, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")
print(f"  Samples: {tfidf_matrix.shape[0]}")
print(f"  Features (words): {tfidf_matrix.shape[1]}")

print(f"\nFeature Names (Top 20 terms):")
feature_names = vectorizer.get_feature_names_out()
print(f"  {', '.join(feature_names)}")

# Display TF-IDF scores for first document
print(f"\nTF-IDF Scores for Document 1:")
tfidf_dense = tfidf_matrix.toarray()
scores = tfidf_dense[0]
nonzero_idx = np.where(scores > 0)[0]

for idx in sorted(nonzero_idx, key=lambda i: scores[i], reverse=True):
    print(f"  {feature_names[idx]:15} → {scores[idx]:.4f}")

# ===== STEP 5: WORD FREQUENCY ANALYSIS =====
print("\n" + "=" * 80)
print("STEP 5: WORD FREQUENCY ANALYSIS")
print("-" * 80)

# Calculate average TF-IDF per term
avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
top_indices = np.argsort(avg_tfidf)[-10:][::-1]

print(f"\nTop 10 Terms by Average TF-IDF:")
for rank, idx in enumerate(top_indices, 1):
    print(f"  {rank}. {feature_names[idx]:15} → {avg_tfidf[idx]:.4f}")

# ===== STEP 6: DIMENSIONALITY REDUCTION =====
print("\n" + "=" * 80)
print("STEP 6: DIMENSIONALITY REDUCTION (SVD)")
print("-" * 80)

svd = TruncatedSVD(n_components=3, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

print(f"Original shape: {tfidf_matrix.shape}")
print(f"Reduced shape: {tfidf_reduced.shape}")
print(f"Explained variance: {svd.explained_variance_ratio_.sum():.4f}")

# ===== STEP 7: DOCUMENT SIMILARITY =====
print("\n" + "=" * 80)
print("STEP 7: DOCUMENT SIMILARITY")
print("-" * 80)

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(tfidf_matrix)
print(f"Similarity Matrix shape: {similarity.shape}")

print(f"\nMost Similar Documents:")
for i in range(1):
    idx = np.argsort(similarity[i])[-3:][::-1]
    print(f"  Document {i}: {documents[i][:40]}...")
    for j, sim_idx in enumerate(idx):
        if sim_idx != i:
            print(f"    Similar: {documents[sim_idx][:40]}... (Similarity: {similarity[i, sim_idx]:.4f})")

# ===== STEP 8: TEXT CLASSIFICATION =====
print("\n" + "=" * 80)
print("STEP 8: TEXT CLASSIFICATION WITH TF-IDF")
print("-" * 80)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    documents, labels, test_size=0.3, random_state=42
)

# Vectorize
vectorizer_clf = TfidfVectorizer(max_features=20, stop_words='english')
X_train_tfidf = vectorizer_clf.fit_transform(X_train)
X_test_tfidf = vectorizer_clf.transform(X_test)

# Train classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Predict
y_pred = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Classifier: Naive Bayes + TF-IDF")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ===== STEP 9: VISUALIZATION =====
print("\n" + "=" * 80)
print("STEP 9: VISUALIZATION")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Word Frequency
axes[0, 0].barh(feature_names[top_indices], avg_tfidf[top_indices], color='steelblue')
axes[0, 0].set_xlabel('Average TF-IDF')
axes[0, 0].set_title('Top 10 Terms by TF-IDF', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Plot 2: Document Similarity Heatmap
sns.heatmap(similarity, cmap='YlOrRd', ax=axes[0, 1], cbar_kws={'label': 'Similarity'})
axes[0, 1].set_title('Document Similarity Matrix', fontweight='bold')
axes[0, 1].set_xlabel('Document')
axes[0, 1].set_ylabel('Document')

# Plot 3: Cumulative Explained Variance
cumsum = np.cumsum(svd.explained_variance_ratio_)
axes[1, 0].plot(range(1, len(cumsum) + 1), cumsum, 'bo-', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Number of Components')
axes[1, 0].set_ylabel('Cumulative Explained Variance')
axes[1, 0].set_title('SVD Explained Variance', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Classification Performance
categories = ['ML', 'DL', 'NLP']
class_acc = []
for label_val in range(3):
    mask = np.array(y_test) == label_val
    if mask.sum() > 0:
        acc = (y_pred[mask] == label_val).sum() / mask.sum()
        class_acc.append(acc)

axes[1, 1].bar(categories, class_acc, color=['blue', 'green', 'red'], alpha=0.7)
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Class-wise Classification Accuracy', fontweight='bold')
axes[1, 1].set_ylim([0, 1.1])
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('tfidf_analysis.png', dpi=150, bbox_inches='tight')
print("Visualization saved as 'tfidf_analysis.png'")
plt.show()

# ===== STEP 10: SUMMARY =====
print("\n" + "=" * 80)
print("STEP 10: TEXT ANALYTICS SUMMARY")
print("=" * 80)

print(f"\nTEXT PREPROCESSING PIPELINE:")
print(f"  1. Tokenization: Split into words")
print(f"  2. Lowercase: Normalize case")
print(f"  3. Remove stopwords: Filter common words")
print(f"  4. Stemming: Root form (running→run)")

print(f"\nTF-IDF VECTORIZATION:")
print(f"  Documents: {len(documents)}")
print(f"  Vocabulary size: {len(feature_names)}")
print(f"  Sparsity: High (most entries zero)")

print(f"\nCLASSIFICATION PERFORMANCE:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Model: Naive Bayes")
print(f"  Features: TF-IDF vectors")

print(f"\n✓ TEXT ANALYTICS PRACTICAL COMPLETED!")
print("=" * 80)
```

---

## KEY FORMULAS

### TF-IDF
$$TF\text{-}IDF = TF \times IDF = \frac{f(t,d)}{|d|} \times \log\left(\frac{N}{n_t}\right)$$

### Cosine Similarity
$$\cos(\theta) = \frac{A \cdot B}{||A|| ||B||} = \frac{\sum A_i B_i}{\sqrt{\sum A_i^2}\sqrt{\sum B_i^2}}$$

---

## IMPORTANT FUNCTIONS

| Function | Purpose |
|----------|---------|
| `TfidfVectorizer()` | Convert text to TF-IDF |
| `word_tokenize()` | Split into words |
| `stopwords.words()` | Get stop words |
| `PorterStemmer()` | Stemming |
| `pos_tag()` | POS tagging |
| `cosine_similarity()` | Document similarity |
| `TruncatedSVD()` | Reduce dimensions |

---

## EXTERNAL VIVA QUESTIONS (10 Key)

### Q1: What does TF-IDF stand for?
**A:** Term Frequency-Inverse Document Frequency. Measures word importance in document.

### Q2: Why remove stop words?
**A:** Common words (the, a, is) appear everywhere, don't help classification.

### Q3: What is stemming?
**A:** Reduce words to root form (running→run, cats→cat).

### Q4: TF high, IDF low → importance?
**A:** Word appears often in document but in many documents → not distinctive.

### Q5: How does TF-IDF differ from word count?
**A:** Word count: raw frequency. TF-IDF: normalized, considers document frequency.

### Q6: What is cosine similarity?
**A:** Measures angle between vectors [0,1]. 1=identical, 0=orthogonal.

### Q7: Why normalize TF-IDF?
**A:** Long documents get higher values. Normalization makes scores comparable.

### Q8: How does dimensionality reduction help?
**A:** Reduces noise, finds main topics, speeds up computation.

### Q9: Can you use TF-IDF for recommendation?
**A:** Yes - compute similarity between documents/users, recommend similar items.

### Q10: Limitations of TF-IDF?
**A:** Ignores word order, semantics, context. Bag-of-words approach.

---

**Ready for External Examination and Journal Submission**
