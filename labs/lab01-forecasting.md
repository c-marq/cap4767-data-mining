# Lab 1 — Time Series Exploration & Forecasting
**CAP4767 Data Mining with Python | Individual Assignment | 20 points**

---

## Overview
In class your group forecast hotel occupancy (`occupancy_rate_pct`) on the Florida Hotel Occupancy dataset. Now you will apply the same workflow independently to a different column of that dataset, demonstrating that you can transfer the skill, not just follow along.

This lab is completed individually. Your notebook must reflect your own work and your own interpretation.

---

## Your Dataset
**Florida Hotel Occupancy**: quarterly data from 2005 to 2024 (80 observations, 10 columns). The source of this dataset is not documented, and it appears to be simulated teaching data rather than official statistics. The starter notebook loads it for you. Choose **one numeric column** as your forecasting target. Do not use `occupancy_rate_pct`; that was the group exercise target.

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/labs/lab01_forecasting.ipynb`

The starter notebook has the full instructions for every task.

---

## Tasks and Points

| Part | Tasks | Points |
|------|-------|--------|
| A: Exploration (Module 1 skills) | A1 Time series plot · A2 Rolling windows · A3 Resampling · A4 ADF stationarity test · A5 Seasonal decomposition (1.6 points each) | 8 |
| B: Forecasting (Module 2 skills) | B1 Train/test split · B2 SARIMAX forecast · B3 Prophet forecast · B4 Model comparison · B5 Future forecast (1.6 points each) | 8 |
| C: Reflection | C1 Model analysis · C2 Real-world application (2 points each) | 4 |
| **Total** | | **20** |

This lab rewards **process over perfection**. If your code doesn't work but you explain what you tried and what went wrong, you earn most of the points.

---

## What to Document
After every task that produces output, add a markdown cell explaining what you see. A notebook full of code with no explanation earns fewer points than a notebook with thoughtful commentary even if some code has minor errors.

If something does not work, document what you tried, what error you got, and what you think caused it. That process documentation earns partial credit.

---

## Submission
Submit via Canvas.

**File naming:** `LastName_FirstName_Lab01.ipynb`

**What to submit:** Your completed Colab notebook downloaded as .ipynb. All cells must be run with outputs visible before downloading.
