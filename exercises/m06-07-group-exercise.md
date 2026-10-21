# Modules 6–7 Group Exercise — RFM + K-Means on a New Dataset
**CAP4767 Data Mining with Python | Modules 6–7 | 10 Points**

---

## Overview
Your group runs the complete RFM + K-Means pipeline on a different dataset from the demo, applying what you learned in both sessions: RFM scoring, manual segmentation, K-Means clustering, and segment comparison. Before coding, the group discusses what Recency, Frequency, and Monetary mean for the new dataset and whether you expect revenue concentration to be more or less extreme than in the Online Retail demo dataset. The deliverable is the completed notebook plus a 3-minute presentation. Groups self-assign four roles: Lead Coder, Data Interpreter, Presenter, and QA Reviewer.

---

## Your Dataset
Choose **one** of two options. Run the setup cell without modifying it (it provides the `segment_customer()` function from the demo), then uncomment ONE dataset option in the data loading cell and run it.

**Option A: Brazilian E-Commerce (Olist)**, about 100K orders. More challenging: it requires merging 3 tables (orders, customers, payments). Use `customer_unique_id` (not `customer_id`) because one person can have multiple customer IDs; revenue is `payment_value`. The notebook loads three files from this base URL:

`base = "https://raw.githubusercontent.com/c-marq/cap4767-data-mining/refs/heads/main/data"`

- `{base}/olist_customers_dataset.csv`
- `{base}/olist_orders_dataset.csv`
- `{base}/olist_order_payments_dataset.csv`

**Option B: Restaurant Orders**, about 1K orders. Simpler and beginner-friendly. There is no true customer ID, so each order number is treated as a unique customer. This limits RFM interpretation (Frequency will always be 1), and you discuss that limitation in your share-out.

- `https://raw.githubusercontent.com/c-marq/cap4767-data-mining/refs/heads/main/data/restaurant-1-orders.csv`

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/exercises/m06-07_group_exercise.ipynb`

The starter notebook has the full instructions for every task.

---

## Tasks and Points

| Task | What you do |
|------|-------------|
| 1: Calculate RFM Metrics | Set a snapshot date (1 day after the last transaction) · Recency, Frequency, and Monetary per customer · `rfm.describe()` |
| 2: Assign RFM Scores and Manual Segments | Quintile scores (1–5) for R, F, and M · composite score · apply `segment_customer()` · segment counts and interpretation |
| 3: Scale Features and Run Elbow Method | `StandardScaler` on the R, F, M scores · K-Means for k = 2 through 8 · inertia plot · chosen k with justification |
| 4: Fit K-Means and Profile Clusters | Fit with your chosen k · cluster profiling heatmap (mean R, F, M per cluster) · cluster sizes · name each cluster |
| 5: Compare Manual Segments vs K-Means | Crosstab of manual segments against clusters · interpretation |
| Six Success Criteria Questions | Answer six short questions about your cleaned dataset, the top segments, revenue concentration, the elbow result, the highest-value cluster, and what K-Means revealed |
| Share-Out | 3-minute presentation: dataset and R/F/M mapping, one finding from each method, and one business recommendation |

The exercise is worth **10 points** in total. The notebook does not assign points to individual tasks.

---

## Submission
Each student submits their own copy individually via Canvas.

**File naming:** `LastName_FirstName_Exercise05.ipynb`

**What to submit:** Completed notebook with all cells run and all markdown discussion cells filled in.
