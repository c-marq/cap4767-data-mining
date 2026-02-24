# Lab 1 — Forecasting with SARIMAX and Prophet
**CAP4767 Data Mining with Python | Individual Assignment | 20 points**

---

## Overview
In class you built forecasting models on the Australian Tourism dataset as a group. Now you will apply the same workflow independently to a different dataset — demonstrating that you can transfer the skill, not just follow along.

This lab is completed individually. Your notebook must reflect your own work and your own interpretation.

---

## Your Dataset
Choose **one** of the following:

| Dataset | Source | Description |
|---------|--------|-------------|
| Air Passengers | Available in statsmodels | Monthly international airline passengers, 1949–1960 |
| Monthly Retail Sales | Kaggle | US monthly retail sales by category |
| Energy Consumption | Kaggle | Hourly or daily power consumption data |

You may also use any time series dataset of your choice with instructor approval — message via Canvas Inbox before starting if you want to use a different dataset.

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/labs/lab01-forecasting.ipynb`

---

## Tasks

**Task 1 — Load and explore your dataset**
Load your chosen dataset. Set the datetime index. Plot the raw series. In a markdown cell, describe what you observe: trend, seasonality, any anomalies.

**Task 2 — Decompose the series**
Run `seasonal_decompose`. Plot all four components. In a markdown cell, describe each component in 1–2 sentences.

**Task 3 — Test for stationarity**
Run the ADF test. Is the series stationary? If not, difference it and test again. Document your steps.

**Task 4 — Build a SARIMAX model**
Use `auto_arima` to find parameters. Fit SARIMAX on the training set (80%). Generate forecasts for the test period. Plot actual vs. predicted. Calculate RMSE and MAPE.

**Task 5 — Build a Prophet model**
Reformat the data for Prophet. Fit and forecast the same test period. Plot the forecast with components. Calculate RMSE and MAPE.

**Task 6 — Compare the models**
Build a comparison table: SARIMAX vs. Prophet with RMSE and MAPE. Plot both forecasts on the same chart alongside actuals.

**Task 7 — Write your analysis**
In a final markdown cell (minimum 150 words), answer:
- Which model performed better on your dataset and why?
- What characteristics of your specific dataset might explain the difference?
- If a business needed a 12-month forecast from this data, which model would you recommend and why?

---

## What to Document
After every task that produces output, add a markdown cell explaining what you see. A notebook full of code with no explanation earns fewer points than a notebook with thoughtful commentary even if some code has minor errors.

If something does not work, document what you tried, what error you got, and what you think caused it. That process documentation earns partial credit.

---

## Submission
Submit via Canvas.

**File naming:** `LastName_FirstName_Lab01.ipynb`

**What to submit:** Your completed Colab notebook downloaded as .ipynb. All cells must be run with outputs visible before downloading.

---

## Rubric

| Criterion | Excellent (20) | Good (14–16) | Needs Work (6–8) |
|-----------|---------------|-------------|-----------------|
| **Technical completeness** (8 pts) | All 7 tasks complete, both models built and evaluated with correct metrics | Most tasks complete, one model has minor issues | Missing tasks or metrics not calculated |
| **Interpretation quality** (8 pts) | Every output explained, final analysis is specific and evidence-based, 150+ words | Most outputs explained, analysis present but general | Little to no explanation, or analysis missing |
| **Code quality and documentation** (4 pts) | Runs cleanly, markdown cells between every task, logical structure | Mostly runs, some markdown present | Does not run or no markdown |
