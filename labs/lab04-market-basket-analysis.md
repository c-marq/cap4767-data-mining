# Lab 4 — Market Basket Analysis
**CAP4767 Data Mining with Python | Individual Assignment | 20 Points**

---

## Overview
In class you ran market basket analysis on the Online Retail II dataset as a group. This lab asks you to apply the same technique to a different context — and push further into interpreting and acting on the rules you find.

This lab is completed individually. Your notebook must reflect your own work and your own interpretation.

---

## Your Dataset
Choose **one** of the following:

| Dataset | Source | Description |
|---------|--------|-------------|
| Groceries | Kaggle / mlxtend examples | 9,835 transactions from a grocery store |
| Instacart Orders | Kaggle | Online grocery order data with prior purchase history |
| Bread Basket | Kaggle | Bakery transactions by time of day |

You may also use any transaction dataset of your choice — message via Canvas Inbox before starting.

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/labs/lab04-market-basket-analysis.ipynb`

---

## Tasks

**Task 1 — Load and prepare transaction data**
Load your dataset. Clean it: remove nulls, cancelled orders, and any rows that would corrupt the basket structure. Document every cleaning step and why you made it.

**Task 2 — Build the basket matrix**
Create a one-hot encoded basket matrix. Print the shape and the first 5 rows. In a markdown cell, explain what one row in this matrix represents.

**Task 3 — Find frequent itemsets**
Run `apriori`. Experiment with at least two different `min_support` thresholds. Document what happens to the number of itemsets as you change the threshold. Choose a final threshold and justify your choice.

> **Reflection checkpoint:** If your frequent itemsets dataframe is empty, your support threshold is too high. If it has tens of thousands of rows, it is too low. Finding the right threshold is part of the skill — document your process.

**Task 4 — Generate and filter association rules**
Generate rules using lift as your metric. Filter to rules where confidence > 0.4 AND lift > 1.5. How many rules remain? If you have more than 50, tighten your filter. If you have fewer than 5, loosen it. Document your final thresholds and reasoning.

**Task 5 — Rank and interpret the top 10 rules**
Sort by lift descending. Print the top 10 rules. For each of the top 3 rules, write one sentence in plain English that a store manager could understand and act on.

**Task 6 — Visualize the rules**
Create at least two visualizations:
1. Scatter plot: Support vs. Confidence, colored by Lift
2. A second visualization of your choice (network graph, heatmap, bar chart of top antecedents)

In a markdown cell, describe what each visualization reveals that the table alone does not.

**Task 7 — Write your strategic recommendation**
In a final markdown cell (minimum 150 words), write a recommendation for the business whose data you used. Include:
- Your top 3 actionable rules translated into business language
- One specific promotion or product placement recommendation with a rationale
- One rule that looks strong statistically but you would not act on — and why

---

## What to Document
Every threshold decision should be explained. Show your experimentation. A student who documents "I tried 0.05, got 3 rules, so I lowered to 0.02 and got 47 rules" demonstrates real understanding.

---

## Submission
Submit via Canvas.

**File naming:** `LastName_FirstName_Lab04.ipynb`

**What to submit:** Your completed Colab notebook downloaded as .ipynb. All cells must be run with outputs visible before downloading.

---

## Rubric

| Criterion | Excellent (20) | Good (14–16) | Needs Work (6–8) |
|-----------|---------------|-------------|-----------------|
| **Technical completeness** (8 pts) | Basket matrix built, itemsets found, rules generated and filtered, two visualizations present | Most steps complete, one visualization or filtering step missing | Rules not generated or basket matrix incorrect |
| **Threshold experimentation and reasoning** (6 pts) | Multiple thresholds tested, final choice documented with reasoning | Some experimentation shown, reasoning present but brief | No experimentation, thresholds used without justification |
| **Business recommendation** (6 pts) | 150+ words, top 3 rules translated clearly, one rule critically evaluated | Recommendation present, translation somewhat technical, no critical evaluation | Missing recommendation or rules not translated to business language |
