# Module 2 Group Exercise — Forecasting Florida Hotel Occupancy
**CAP4767 Data Mining with Python | Module 2 | 10 Points**

---

## Overview
Your group applies the SARIMAX and Prophet forecasting pipeline to quarterly Florida hotel occupancy data, compares model performance using RMSE and R², and presents a recommendation to a simulated hotel revenue team. Along the way you practice exploratory analysis of a seasonal time series, stationarity testing (ADF), SARIMAX parameter selection with `auto_arima`, Prophet forecasting with direct parameters, and communicating results to a non-technical audience. Before coding, the group discusses when a Miami Beach hotel would see its highest and lowest occupancy and what events could cause unexpected spikes or dips. Groups self-assign four roles: Lead Coder, Data Interpreter, Presenter, and QA Reviewer.

---

## Your Dataset
**Florida Hotel Occupancy**: quarterly data from 2005 to 2024 (80 observations, 10 columns). The source of this dataset is not documented, and it appears to be simulated teaching data rather than official statistics. The forecasting target is `occupancy_rate_pct`, the percentage of hotel rooms occupied each quarter. The setup cell loads it as `hotel_df`; do not modify it.

Load URL: `https://raw.githubusercontent.com/c-marq/cap4767-data-mining/refs/heads/main/data/florida_hotel_occupancy.csv`

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/exercises/m02_group_exercise.ipynb`

The starter notebook has the full instructions for every task.

---

## Tasks and Points

| Step | Tasks |
|------|-------|
| 1: Explore the Data | Summary statistics with `.describe()` · extract `occupancy_rate_pct` as a series with quarterly frequency |
| 2: Exploratory Analysis | Time series plot · ADF stationarity test · seasonal decomposition (period = 4) |
| 3: Train/Test Split | First 64 quarters for training, remaining 16 for testing · plot both with a red dashed line at the split |
| 4: SARIMAX Forecasting | `auto_arima` with `seasonal=True`, `m=4` · fit SARIMAX and forecast the test period · calculate RMSE and R² |
| 5: Prophet Forecasting | Prophet-formatted DataFrame · fit with multiplicative seasonality · forecast the test period · calculate RMSE and R² |
| 6: Compare Models & Build Recommendation | Comparison table · combined overlay plot of both forecasts · group recommendation |
| 7: Group Recommendation | 3–5 sentences for a Florida hotel revenue team: which model performed better and by how much, which quarters each model struggled with most, and which model you would recommend for production forecasting and why |

The exercise is worth **10 points** in total. The notebook does not assign points to individual steps.

---

## Submission
Submit via Canvas.

**Deliverable:** The completed notebook (each student submits their own copy) plus a 3-minute presentation.

**File naming:** `LastName_FirstName_Exercise02.ipynb`

**What to submit:** Completed notebook with all cells run, comparison table visible, and markdown discussion cells filled in.
