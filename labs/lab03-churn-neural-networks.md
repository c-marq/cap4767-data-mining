# Lab 3 — Churn Analysis: Logistic Regression + Neural Network
**CAP4767 Data Mining with Python | Individual Assignment | 20 Points**
**Due: Sunday, April 5 by 11:59 PM EST**

---

## Overview
You have now built churn models in class. This lab asks you to go further: apply the full classification pipeline to a new dataset, make deliberate architectural decisions on your neural network, and write a business recommendation a real manager could act on.

This lab is completed individually. Your notebook must reflect your own work and your own interpretation.

---

## Your Dataset
Choose **one** of the following:

| Dataset | Source | Description |
|---------|--------|-------------|
| Bank Customer Churn | Kaggle | 10,000 bank customers, binary churn target |
| E-Commerce Churn | Kaggle | Online retail customer churn with behavioral features |
| Credit Card Attrition | Kaggle | Credit card customers who closed their accounts |

You may also use any binary classification dataset of your choice — message via Canvas Inbox before starting.

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/labs/lab03-churn-neural-networks.ipynb`

---

## Tasks

**Task 1 — Load, explore, and document class imbalance**
Load your dataset. Check the churn rate. If churners are less than 30% of the dataset, document that you have class imbalance and note how it might affect your results.

**Task 2 — Feature engineering and preparation**
Handle missing values. Encode categoricals. Drop any columns that would cause data leakage (e.g., account closed date). Split 80/20. Scale.

**Task 3 — Build a logistic regression baseline**
Fit LogisticRegression. Print the full classification report and confusion matrix. Calculate AUC-ROC. This is your baseline — document it clearly.

> **Reflection checkpoint:** Look at the recall for the churn class specifically. Is the model catching most churners? Write 2–3 sentences explaining what the confusion matrix tells you about the model's practical usefulness.

**Task 4 — Build and train a neural network**
Build a Keras Sequential model. You must justify your architecture choices in a markdown cell before the model code:
- How many hidden layers and why?
- How many neurons per layer and why?
- Why sigmoid on the output layer?

Train for at least 100 epochs. Plot training vs. validation loss.

**Task 5 — Evaluate and diagnose**
Print the classification report for the neural network. Is the model overfitting? How do you know? What would you do to address it?

**Task 6 — Compare the models**
Build a comparison table: logistic regression vs. neural network across accuracy, precision, recall, F1 (churn class), and AUC-ROC. Plot both ROC curves on the same chart.

**Task 7 — Write your business recommendation**
In a final markdown cell (minimum 200 words), write a recommendation addressed to the VP of Customer Retention at the company whose data you used. Include:
- Which model you recommend deploying and why
- What the model's recall rate means in practical terms (how many churners will it catch per 1,000 customers?)
- What action the company should take when a customer is flagged as high-risk
- One important limitation your model has that the VP needs to know

---

## What to Document
Justify your architecture decisions before writing the model code. Document what did not work. A thoughtful reflection on failure earns more points than silent broken code.

---

## Submission
Submit via Canvas.

**File naming:** `LastName_FirstName_Lab03.ipynb`

**What to submit:** Your completed Colab notebook downloaded as .ipynb. All cells must be run with outputs visible before downloading.

---

## Rubric

| Criterion | Excellent (20) | Good (14–16) | Needs Work (6–8) |
|-----------|---------------|-------------|-----------------|
| **Technical completeness** (8 pts) | Both models complete, AUC-ROC calculated, ROC curves plotted | Both models present, one metric or plot missing | One model missing or no evaluation |
| **Architecture justification and reflection** (6 pts) | Architecture choices explained before code, overfitting diagnosed, limitations addressed | Most justifications present, some gaps | No architecture justification or no reflection |
| **Business recommendation** (6 pts) | 200+ words, specific to the dataset, includes practical impact numbers | Recommendation present, somewhat general, may lack numbers | Missing or generic recommendation not tied to the data |
