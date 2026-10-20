# Module 8 Group Exercise — Market Basket Analysis
**CAP4767 Data Mining with Python | Module 8 | 10 Points**

---

## Overview
Your group runs a complete Market Basket Analysis pipeline on the Online Retail II dataset, identifies the strongest association rules by lift, produces one visualization, and presents one product placement and one cross-sell recommendation. Before coding, the group discusses what new question Market Basket Analysis answers that RFM and K-Means could not, and names one business decision a marketing director would want it to inform. The deliverable is the completed notebook plus a 3-minute presentation. Groups self-assign four roles: Lead Coder, Data Interpreter, Presenter, and QA Reviewer.

---

## Your Dataset
**Online Retail II**: online retail transactions, loaded from an Excel file. The setup cell loads it with `pd.read_excel` and cleans it: it removes cancelled invoices, rows with a missing `CustomerID`, rows with non-positive `Quantity` or `UnitPrice`, non-product stock codes, and rows with a missing `Description`. Do not modify the setup cell.

Load URL: `https://github.com/fenago/datasets/raw/refs/heads/main/Online%20Retail%20(1).xlsx`

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/exercises/m08_group_exercise.ipynb`

The starter notebook has the full instructions for every task.

---

## Tasks and Points

| Task | What you do | Points |
|------|-------------|--------|
| 1: Transform to Basket Format | One-hot encoded basket matrix: one row per invoice, one column per `Description`, Boolean values · print the shape | 1 |
| 2: Run Apriori | `apriori()` with `min_support=0.02` and `use_colnames=True` · count of frequent itemsets · top 10 by support | 1 |
| 3: Generate and Filter Rules | Rules by lift with `min_threshold=2.0` · second filter `confidence >= 0.3` · sort by lift · display the top 15 rules with all 5 metrics | 2 |
| 4: Interpret the Top 3 Rules | For each of the top 3 rules by lift: which products are associated, how strong the association is, and why they might be bought together | 2 |
| 5: Produce One Visualization | Choose one: network graph, support vs confidence scatter, or lift heatmap | 2 |
| 6: Business Recommendations | One product placement recommendation (cite the rule and its lift) and one cross-sell recommendation (cite the rule and its confidence) | 2 |
| **Total** | | **10** |

---

## Submission
Each student submits their own copy individually via Canvas.

**File naming:** `LastName_FirstName_Exercise06.ipynb`

**What to submit:** Completed notebook with all cells run, rules table visible, and all markdown discussion cells filled in.
