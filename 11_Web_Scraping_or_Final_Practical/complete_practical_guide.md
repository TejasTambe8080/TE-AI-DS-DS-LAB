# PRACTICAL 11: WEB SCRAPING & DATA COLLECTION

## Practical Title
Web Scraping and Data Collection from Online Sources

## Aim
To collect, parse, and extract data from websites programmatically for data science applications.

## Objective
1. Understand web scraping principles and ethics
2. Use BeautifulSoup for HTML parsing
3. Extract structured data from web pages
4. Handle common challenges (dynamic content, errors)
5. Store and process scraped data

## Problem Statement
Scrape product data or news articles from websites to create a dataset for analysis. Learn best practices and legal considerations.

---

## THEORY

### 1. Web Scraping Basics
- **HTTP Request:** Fetch HTML content
- **HTML Parsing:** Extract structured data
- **Data Extraction:** Find specific elements
- **Storage:** Save to CSV/database

### 2. Technologies
- **BeautifulSoup:** Parse HTML/XML
- **Requests:** Make HTTP requests
- **Selenium:** Handle JavaScript-heavy sites
- **Scrapy:** Full framework for large-scale scraping

### 3. HTML Structure
```html
<html>
  <head><title>Page Title</title></head>
  <body>
    <div class="container">
      <p class="text">Content</p>
    </div>
  </body>
</html>
```

### 4. CSS Selectors
- `tag`: Select by tag name
- `.class`: Select by class
- `#id`: Select by ID
- `tag.class`: Combined

### 5. Best Practices & Ethics
- Check robots.txt
- Read Terms of Service
- Rate limiting (delay between requests)
- Identify as bot in User-Agent
- Don't overload servers
- Respect copyright

---

## COMPLETE CODE

```python
# ===== PRACTICAL 11: WEB SCRAPING & DATA COLLECTION =====

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt

print("=" * 80)
print("PRACTICAL 11: WEB SCRAPING AND DATA COLLECTION")
print("=" * 80)

# ===== STEP 1: BASIC WEB SCRAPING =====
print("\nSTEP 1: BASIC WEB SCRAPING - FETCH HTML CONTENT")
print("-" * 80)

# Example: Scrape quotes from a public website
url = "http://quotes.toscrape.com"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

try:
    response = requests.get(url, headers=headers, timeout=5)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print(f"Successfully fetched: {url}")
        print(f"Content length: {len(response.text)} characters")
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        print(f"\nPage Title: {soup.title.string if soup.title else 'Not found'}")
except Exception as e:
    print(f"Error fetching page: {e}")

# ===== STEP 2: EXTRACT DATA STRUCTURES =====
print("\n" + "=" * 80)
print("STEP 2: EXTRACT STRUCTURED DATA")
print("-" * 80)

# Create sample HTML for demonstration
sample_html = """
<html>
<body>
<div class="container">
    <div class="quote">
        <span class="text">"Life is what happens to you while you're busy making other plans."</span>
        <small class="author">— John Lennon</small>
        <div class="tags">
            <a class="tag">life</a>
            <a class="tag">plans</a>
        </div>
    </div>
    <div class="quote">
        <span class="text">"The way to get started is to quit talking and begin doing."</span>
        <small class="author">— Walt Disney</small>
        <div class="tags">
            <a class="tag">action</a>
            <a class="tag">start</a>
        </div>
    </div>
</div>
</body>
</html>
"""

soup = BeautifulSoup(sample_html, 'html.parser')

# Extract quotes
quotes = []
for quote_div in soup.find_all('div', class_='quote'):
    text = quote_div.find('span', class_='text').string
    author = quote_div.find('small', class_='author').string
    
    tags = []
    for tag in quote_div.find_all('a', class_='tag'):
        tags.append(tag.string)
    
    quotes.append({
        'text': text,
        'author': author,
        'tags': ', '.join(tags)
    })

quotes_df = pd.DataFrame(quotes)
print(f"Extracted {len(quotes_df)} quotes")
print(f"\nQuotes DataFrame:")
print(quotes_df)

# ===== STEP 3: CSS SELECTORS =====
print("\n" + "=" * 80)
print("STEP 3: CSS SELECTORS FOR DATA EXTRACTION")
print("-" * 80)

# Example CSS selectors
print(f"\nCSS Selector Examples:")
print(f"  'div.quote' → All div tags with class 'quote'")
print(f"  'span.text' → All span tags with class 'text'")
print(f"  'div#main' → Div with id 'main'")
print(f"  'a[href]' → All links with href attribute")

# Using select() with CSS selectors
texts = soup.select('span.text')
print(f"\nUsing 'select' with CSS selector:")
print(f"  Found {len(texts)} text elements")
for text in texts:
    print(f"  - {text.string[:50]}...")

# ===== STEP 4: PARSE JSON DATA =====
print("\n" + "=" * 80)
print("STEP 4: PARSE AND EXTRACT JSON DATA")
print("-" * 80)

# Sample JSON (often embedded in HTML)
json_data = {
    "products": [
        {"id": 1, "name": "Laptop", "price": 1000, "rating": 4.5},
        {"id": 2, "name": "Phone", "price": 500, "rating": 4.2},
        {"id": 3, "name": "Tablet", "price": 300, "rating": 4.0}
    ]
}

products_df = pd.DataFrame(json_data['products'])
print(f"Extracted {len(products_df)} products from JSON")
print(products_df)

# ===== STEP 5: HANDLE COMMON CHALLENGES =====
print("\n" + "=" * 80)
print("STEP 5: HANDLE WEB SCRAPING CHALLENGES")
print("-" * 80)

print(f"\nCommon Challenges and Solutions:")
print(f"1. TIMEOUT ERRORS")
print(f"   Solution: Use timeout parameter, retry logic")
print(f"   Example: requests.get(url, timeout=10)")

print(f"\n2. DYNAMIC CONTENT (JavaScript)")
print(f"   Solution: Use Selenium instead of BeautifulSoup")
print(f"   Example: from selenium import webdriver")

print(f"\n3. BLOCKED BY RATE LIMITING")
print(f"   Solution: Add delays between requests")
print(f"   Example: time.sleep(1) between requests")

print(f"\n4. INCORRECT DATA EXTRACTION")
print(f"   Solution: Inspect HTML, test selectors")
print(f"   Example: Use browser developer tools")

print(f"\n5. HANDLING MISSING DATA")
print(f"   Solution: Check if element exists before extracting")
print(f"   Example: if element: value = element.string")

# Example: Error handling
print(f"\nError Handling Example:")

def safe_scrape(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise exception for bad status
        return BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.Timeout:
        print(f"Timeout: Page took too long to load")
        return None
    except requests.exceptions.ConnectionError:
        print(f"Connection error: Could not reach page")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e.response.status_code}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

print(f"Defined safe_scrape() function with error handling")

# ===== STEP 6: DATA STORAGE =====
print("\n" + "=" * 80)
print("STEP 6: STORE SCRAPED DATA")
print("-" * 80)

# Save to CSV
quotes_df.to_csv('scraped_quotes.csv', index=False)
print(f"Saved quotes to 'scraped_quotes.csv'")

# Save to JSON
quotes_dict = quotes_df.to_dict(orient='records')
with open('scraped_quotes.json', 'w') as f:
    json.dump(quotes_dict, f, indent=2)
print(f"Saved quotes to 'scraped_quotes.json'")

print(f"\nData stored successfully!")

# ===== STEP 7: DATA QUALITY CHECK =====
print("\n" + "=" * 80)
print("STEP 7: DATA QUALITY ASSESSMENT")
print("-" * 80)

print(f"\nScraped Data Quality:")
print(f"  Total records: {len(quotes_df)}")
print(f"  Missing values: {quotes_df.isnull().sum().sum()}")
print(f"  Duplicates: {quotes_df.duplicated().sum()}")
print(f"  Data types:")
for col in quotes_df.columns:
    print(f"    {col}: {quotes_df[col].dtype}")

# ===== STEP 8: DATA ANALYSIS =====
print("\n" + "=" * 80)
print("STEP 8: ANALYZE SCRAPED DATA")
print("-" * 80)

print(f"\nText Length Statistics:")
quotes_df['text_length'] = quotes_df['text'].str.len()
print(quotes_df[['text', 'text_length']].describe())

# ===== STEP 9: VISUALIZATION =====
print("\n" + "=" * 80)
print("STEP 9: VISUALIZE SCRAPED DATA")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Text length distribution
axes[0].hist(quotes_df['text_length'], bins=10, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Quote Length (characters)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Quote Lengths', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Plot 2: Sample data table visualization
products_df_plot = products_df.sort_values('price', ascending=False)
axes[1].barh(products_df_plot['name'], products_df_plot['price'], color='coral', edgecolor='black')
axes[1].set_xlabel('Price ($)')
axes[1].set_title('Product Prices from JSON', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('scraping_analysis.png', dpi=150, bbox_inches='tight')
print("Visualizations saved as 'scraping_analysis.png'")
plt.show()

# ===== STEP 10: BEST PRACTICES SUMMARY =====
print("\n" + "=" * 80)
print("STEP 10: WEB SCRAPING BEST PRACTICES")
print("=" * 80)

best_practices = {
    'Ethics': [
        '✓ Check robots.txt',
        '✓ Read Terms of Service',
        '✓ Don\'t scrape copyrighted content',
        '✓ Respect rate limits'
    ],
    'Technical': [
        '✓ Identify as bot in User-Agent',
        '✓ Add delays between requests',
        '✓ Handle errors gracefully',
        '✓ Validate extracted data'
    ],
    'Performance': [
        '✓ Cache requests',
        '✓ Use connection pooling',
        '✓ Implement retry logic',
        '✓ Monitor resource usage'
    ]
}

for category, practices in best_practices.items():
    print(f"\n{category}:")
    for practice in practices:
        print(f"  {practice}")

# ===== STEP 11: SUMMARY =====
print("\n" + "=" * 80)
print("STEP 11: WEB SCRAPING SUMMARY")
print("=" * 80)

print(f"\nWEB SCRAPING WORKFLOW:")
print(f"  1. Identify target website")
print(f"  2. Fetch HTML/JSON content")
print(f"  3. Parse structure")
print(f"  4. Extract relevant data")
print(f"  5. Handle errors")
print(f"  6. Store data (CSV/JSON/DB)")
print(f"  7. Validate quality")
print(f"  8. Analyze results")

print(f"\nLIBRARIES USED:")
print(f"  - requests: HTTP requests")
print(f"  - BeautifulSoup: HTML parsing")
print(f"  - json: JSON parsing")
print(f"  - pandas: Data storage/manipulation")

print(f"\nADVANCED TECHNIQUES:")
print(f"  - Selenium: JavaScript rendering")
print(f"  - Scrapy: Full framework")
print(f"  - Playwright: Modern automation")
print(f"  - Proxies: Hide identity")

print(f"\n✓ WEB SCRAPING PRACTICAL COMPLETED!")
print(f"✓ ALL 11 PRACTICALS COMPLETED SUCCESSFULLY!")
print("=" * 80)

print(f"\n" + "=" * 80)
print(f"SUMMARY: DATA SCIENCE PRACTICALS 1-11")
print(f"=" * 80)
print(f"""
PRACTICAL PROGRESSION:
  1. Data Wrangling I (Missing Values, Normalization, Encoding)
  2. Data Wrangling II (Outliers, Transformations, Skewness)
  3. Descriptive Statistics (Central Tendency, Dispersion, Groups)
  4. Linear Regression (Prediction, MSE, RMSE, R²)
  5. Logistic Regression (Classification, Confusion Matrix, Metrics)
  6. Naive Bayes (Probabilistic Classification, Bayes Theorem)
  7. Text Analytics (TF-IDF, NLP, Tokenization)
  8. Data Visualization (EDA, Plots, Analysis)
  9. Clustering (K-Means, Unsupervised, Elbow Method)
  10. Association Rules (Market Basket, Apriori, Rules)
  11. Web Scraping (Data Collection, HTML Parsing)

KEY LEARNING OUTCOMES:
  ✓ Data preprocessing and cleaning
  ✓ Statistical analysis and hypothesis testing
  ✓ Supervised learning (regression, classification)
  ✓ Unsupervised learning (clustering)
  ✓ Feature engineering and transformation
  ✓ Model evaluation and interpretation
  ✓ Data visualization and communication
  ✓ Real-world data collection
  ✓ Business application of ML

TOOLS AND LIBRARIES:
  ✓ Python, pandas, numpy, scikit-learn
  ✓ Matplotlib, seaborn for visualization
  ✓ NLTK for NLP
  ✓ BeautifulSoup for web scraping
  ✓ Various ML algorithms and techniques

READY FOR:
  ✓ External examination and viva
  ✓ Journal submission
  ✓ Project presentations
  ✓ Real-world data science applications
""")
print("=" * 80)
```

---

## KEY CONCEPTS

### HTTP Request Cycle
1. Create request with URL, headers
2. Send to server
3. Receive response (status code + HTML)
4. Parse HTML structure
5. Extract data

### HTML Structure
- Tags: `<div>`, `<p>`, `<a>`, etc.
- Attributes: `class`, `id`, `href`
- Text: Content between tags

### Common Issues
- Dynamic content (needs Selenium)
- Rate limiting (add delays)
- Authentication (handle cookies)
- Missing data (validate extraction)

---

## EXTERNAL VIVA QUESTIONS (10 Key)

### Q1: Why check robots.txt?
**A:** Check what scraping is allowed. robots.txt defines scraping rules for bots.

### Q2: What's difference between static and dynamic scraping?
**A:** Static: BeautifulSoup (pre-rendered HTML). Dynamic: Selenium (JavaScript execution).

### Q3: How to handle timeouts?
**A:** Set timeout parameter, implement retry logic, use exceptions handling.

### Q4: What are CSS selectors?
**A:** Query language for selecting HTML elements. Examples: `.class`, `#id`, `tag.class`.

### Q5: Why add delays between requests?
**A:** Respect server resources, avoid overloading, prevent IP blocking.

### Q6: How to extract data from JSON in HTML?
**A:** Use json.loads() to parse JSON string embedded in HTML.

### Q7: What's User-Agent header?
**A:** Identifies client/bot to server. Some sites block non-browser agents.

### Q8: How to validate scraped data?
**A:** Check for missing values, duplicates, data types, value ranges.

### Q9: Legal issues with scraping?
**A:** Check ToS, respect copyright, don't scrape personal data, follow GDPR.

### Q10: How to scale scraping?
**A:** Use Scrapy framework, distributed crawling, database storage, caching.

---

## CONCLUSION

Web Scraping is powerful for:
- **Data Collection:** Build datasets from web
- **Market Research:** Monitor competitor data
- **News Analysis:** Track trends
- **Real Estate:** Property data aggregation
- **E-commerce:** Price monitoring

**Important:** Always scrape ethically and legally!

---

# COMPREHENSIVE SUMMARY: ALL 11 PRACTICALS

## Practical 1: Data Wrangling I ✓
- **Focus:** Missing values, normalization, encoding
- **Dataset:** Iris (150 samples, 3 species)
- **Techniques:** Min-Max scaling, Standardization, Label Encoding
- **Output:** Cleaned, normalized dataset ready for modeling

## Practical 2: Data Wrangling II ✓
- **Focus:** Outlier detection, skewness reduction, transformations
- **Dataset:** Student Performance (30 students)
- **Techniques:** IQR, Z-score, Log/Sqrt/Box-Cox transformations
- **Output:** Data ready for advanced analysis

## Practical 3: Descriptive Statistics ✓
- **Focus:** Central tendency, dispersion, grouped analysis
- **Dataset:** Iris (by species)
- **Metrics:** Mean, median, std, variance, percentiles, skewness
- **Output:** Statistical insights by species

## Practical 4: Linear Regression ✓
- **Focus:** Prediction, model evaluation, metrics
- **Dataset:** Boston Housing (506 samples, 13 features)
- **Metrics:** MSE, RMSE, R², train-test performance
- **Output:** Predictive model for house prices

## Practical 5: Logistic Regression ✓
- **Focus:** Binary classification, confusion matrix, metrics
- **Dataset:** Social Network Ads
- **Metrics:** Accuracy, precision, recall, F1, ROC-AUC
- **Output:** Classification model for purchase prediction

## Practical 6: Naive Bayes ✓
- **Focus:** Probabilistic classification, Bayes theorem
- **Dataset:** Iris
- **Metrics:** Confusion matrix, accuracy, per-class performance
- **Output:** Probabilistic classifier with 95%+ accuracy

## Practical 7: Text Analytics ✓
- **Focus:** NLP, TF-IDF, text preprocessing
- **Data:** Text documents
- **Techniques:** Tokenization, stemming, POS tagging, vectorization
- **Output:** TF-IDF matrix, document similarity, text classification

## Practical 8: Data Visualization ✓
- **Focus:** EDA, plotting, communication
- **Dataset:** Titanic (891 passengers)
- **Visualizations:** Histograms, boxplots, scatter, heatmaps, violin
- **Output:** Business insights through visual analysis

## Practical 9: Clustering ✓
- **Focus:** Unsupervised learning, K-Means
- **Dataset:** Iris
- **Methods:** Elbow, silhouette analysis, cluster quality
- **Output:** 3 clusters with 90%+ accuracy vs true species

## Practical 10: Association Rules ✓
- **Focus:** Market basket analysis, Apriori algorithm
- **Data:** Transaction data
- **Metrics:** Support, confidence, lift
- **Output:** Business rules for cross-selling and bundling

## Practical 11: Web Scraping ✓
- **Focus:** Data collection, HTML parsing, ethics
- **Techniques:** BeautifulSoup, CSS selectors, error handling
- **Output:** Scraped dataset, CSV/JSON storage

---

**All 11 Practicals: COMPLETE AND JOURNAL-READY! ✓**

**Ready for External Examination and Journal Submission**
