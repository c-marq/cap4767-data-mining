# Lab 04 — Market Basket Analysis
**CAP4767 Data Mining with Python | Individual Assignment | 20 Points (+3 bonus)**

---

## Overview
In this lab you run a complete Market Basket Analysis pipeline on a dataset of your choice. You identify the top association rules by lift, produce at least two visualizations, and write a 1-page business memo to a non-technical store owner explaining three actionable rules. The lab is completed individually.

---

## Your Dataset
Choose **one** of the following. Uncomment ONE option in the dataset selection cell and run it.

| Option | Dataset | Difficulty | Notes |
|--------|---------|------------|-------|
| A | Instacart | Expert | Download from Kaggle. Multiple CSVs requiring joins. 3.4M+ orders. |
| B | Brazilian E-Commerce (Olist) | Intermediate | One merge required. Use `product_category_name` as the item. |
| C | Restaurant Orders | Beginner | Single CSV. Fewer products = higher `min_support`. |

Where each option comes from:
- Option A: download from `https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset`, upload the CSV files to Colab, and read `order_products__prior.csv`, `products.csv`, and `aisles.csv`. Use `product_name` or `aisle` as the item, and consider subsetting to the first 100K orders for performance.
- Option B: files under `https://raw.githubusercontent.com/c-marq/cap4767-data-mining/refs/heads/main/data`, namely `olist_order_items_dataset.csv` and `olist_products_dataset.csv`, merged on `product_id`. The transaction ID is `order_id`.
- Option C: `https://raw.githubusercontent.com/c-marq/cap4767-data-mining/refs/heads/main/data/restaurant-1-orders.csv`

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/labs/lab04_market_basket_analysis.ipynb`

The starter notebook has the full instructions for every task.

---

## Tasks and Points

| Part | Tasks | Points |
|------|-------|--------|
| 1: Data Loading and Cleaning | Task 1.1 Clean the data · Task 1.2 Summary statistics | 3 |
| 2: Basket Transformation | Task 2.1 One-hot basket matrix, shape and density | 2 |
| 3: Apriori and Rule Generation | Task 3.1 Run Apriori and justify `min_support` · Task 3.2 Generate and filter rules (lift, `min_threshold=1.5`, a confidence filter of your choice) | 4 |
| 4: Visualizations (2 of 3) | Task 4.1 Visualization 1 · Task 4.2 Visualization 2 (network graph, support vs confidence scatter, or lift heatmap) | 4 |
| 5: Business Memo (at least 250 words) | 1-page memo to a non-technical store owner: what you did, three actionable rules, one recommendation to skip | 5 |
| 6: Reflection | 3–4 sentence reflection on the rules your data revealed | 2 |
| **Total** | | **20** |
| Bonus: Temporal Comparison | Run the full pipeline on two time periods and compare the rules (up to 3 extra points, no scaffolding) | +3 |

---

## What to Document
Every threshold decision should be explained. Show your experimentation. A student who documents "I tried 0.05, got 3 rules, so I lowered to 0.02 and got 47 rules" demonstrates real understanding.

---

## Submission
Submit via Canvas.

**File naming:** `LastName_FirstName_Lab04.ipynb`

**What to submit:** Your completed Colab notebook downloaded as .ipynb. All cells must be run with outputs visible before downloading.
