# Lab 2 — Regression Pipeline
**CAP4767 Data Mining with Python | Individual Assignment | 20 Points**
**Due: Sunday, March 22 by 11:59 PM EST**

---

## Overview
In class you built linear and logistic regression models on the Telco dataset as a group. Now you will apply the complete regression pipeline to a different dataset independently — from raw data to model evaluation to business interpretation.

This lab is completed individually. Your notebook must reflect your own work and your own interpretation.

---

## Your Dataset
Choose **one** of the following:

| Dataset | Source | Description |
|---------|--------|-------------|
| Medical Insurance Costs | Kaggle | Predict individual insurance charges based on demographics |
| Housing Prices | Kaggle | Predict home sale prices from property features |
| Student Performance | Kaggle | Predict final exam score from study habits and demographics |

You may also use any regression-appropriate dataset of your choice — message via Canvas Inbox before starting if you want to use a different dataset.

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/labs/lab02-regression.ipynb`

---

## Tasks

**Task 1 — Load and explore**
Load your dataset. Check shape, dtypes, missing values, and class distribution (if applicable). In a markdown cell, identify your target variable and explain why it is appropriate for regression.

**Task 2 — Prepare features**
Handle missing values. Encode categorical columns. Split into train and test sets (80/20). Scale numeric features. Document every decision you make.

**Task 3 — Build a linear regression model**
Fit LinearRegression. Calculate RMSE and R² on the test set. Print the top 5 features by coefficient magnitude.

> **Reflection checkpoint:** Is your R² above 0.5? If not, what might explain the low explanatory power? Write 2–3 sentences in a markdown cell.

**Task 4 — Interpret the coefficients**
In a markdown cell, explain in plain English what the largest positive and largest negative coefficient mean for your specific dataset. Avoid generic explanations — reference the actual feature names and values.

**Task 5 — Build a logistic regression model**
If your dataset has a binary target (or if you create one by binning your continuous target), fit LogisticRegression. Print the classification report and confusion matrix.

If your dataset does not have a natural binary target, create one: bin your continuous target at the median (above median = 1, below = 0) and classify that.

**Task 6 — Evaluate and compare**
Build a summary table showing both models: target variable, metric used, and score. In a markdown cell, explain which model is more useful for a business decision and why.

**Task 7 — Write your analysis**
In a final markdown cell (minimum 150 words), answer:
- What did you learn about this dataset from the regression coefficients?
- What would you recommend to a business stakeholder based on your findings?
- What is one limitation of your model that a decision-maker should know about?

---

## What to Document
After every task that produces output, add a markdown cell explaining what you see. Document what you tried, what did not work, and what you adjusted — that process is worth points.

---

## Submission
Submit via Canvas.

**File naming:** `LastName_FirstName_Lab02.ipynb`

**What to submit:** Your completed Colab notebook downloaded as .ipynb. All cells must be run with outputs visible before downloading.

---

## Rubric

| Criterion | Excellent (20) | Good (14–16) | Needs Work (6–8) |
|-----------|---------------|-------------|-----------------|
| **Technical completeness** (8 pts) | Both models built, correct metrics calculated, feature interpretation present | Most steps complete, one model or interpretation has minor gaps | Missing model or no evaluation metrics |
| **Interpretation quality** (8 pts) | Coefficients explained with dataset-specific context, final analysis 150+ words | Interpretation present but generic, analysis present but brief | No interpretation or no final analysis |
| **Code quality and documentation** (4 pts) | Runs cleanly, markdown throughout, logical structure | Mostly runs, some markdown | Does not run or no markdown |
