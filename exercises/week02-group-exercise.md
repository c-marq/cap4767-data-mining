# Group Exercise 2 — Forecasting with SARIMAX and Prophet
**CAP4767 Data Mining with Python | Week 2 | 10 Points**

---

## Objective
Build two forecasting models on the same dataset and compare their predictions. By the end of this exercise your group will have a clear, evidence-based opinion on which model performed better — and more importantly, why.

---

## Your Dataset
**Australian Domestic Tourism** — same dataset from Week 1. This time we forecast it.

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/exercises/week02-group-exercise.ipynb`

---

## Group Roles
Rotate from last week — no one should hold the same role twice in a row.

| Role | Responsibility |
|------|---------------|
| **Lead Coder** | Drives the notebook, shares screen, types the code |
| **Data Interpreter** | Narrates what each output means in plain English |
| **QA Reviewer** | Catches errors, checks outputs match checkpoints |
| **Presenter** | Owns the share-out at the end of class |

---

## Discussion First (5 minutes before opening Colab)
As a group, answer this before writing a single line of code:

> *"What is the difference between explaining the past and predicting the future? Why does a good description of historical data not automatically mean a good forecast?"*

Have the Presenter write a 2–3 sentence answer in the notebook's discussion cell.

---

## Tasks

**Step 1 — Load, prepare, and split the data**
Run the setup cell. Set the datetime index and split into train (80%) and test (20%) sets. Print the size of each split.

> 🛑 **Checkpoint:** Your train set should end before your test set begins with no overlap. Print the last date of train and first date of test to confirm.

**Step 2 — Build the SARIMAX model**
Use `auto_arima` to find the best order parameters. Fit a SARIMAX model on the training set. Print the model summary.

**Step 3 — Forecast with SARIMAX and evaluate**
Generate forecasts for the test period. Plot actual vs. predicted. Calculate RMSE and MAPE.

> 🛑 **Checkpoint:** Your forecast line should cover exactly the same date range as your test set. If it does not, check the `steps` parameter in your `forecast()` call.

**Step 4 — Build the Prophet model**
Reformat the data for Prophet (`ds` and `y` columns). Fit the model and generate a forecast for the same test period. Plot the Prophet forecast with components.

**Step 5 — Evaluate Prophet**
Calculate RMSE and MAPE for Prophet on the same test period. Build a comparison table showing both models side by side.

**Step 6 — Declare a winner**
In a markdown cell, your group must take a position:
- Which model performed better on this dataset?
- Is lower RMSE the only thing that matters, or are there other considerations?
- Would your answer change if this were a 5-year forecast instead of a short-term one?

**Step 7 — Visualize both forecasts together**
Plot both model forecasts on the same chart alongside the actual values. Label clearly.

---

## Share-Out (Last 10 minutes of class)
The **Presenter** walks the class through:
1. The comparison table — RMSE and MAPE side by side
2. Your group's verdict: which model wins and why
3. One scenario where the losing model might actually be the better choice

---

## Submission
Each student submits their own copy individually via Canvas.

**File naming:** `LastName_FirstName_Exercise02.ipynb`

**What to submit:** Completed notebook with all cells run, comparison table visible, and markdown discussion cells filled in.

---

## Rubric

| Criterion | Excellent (10) | Good (7–8) | Needs Work (3–4) |
|-----------|---------------|-----------|-----------------|
| Both models built and evaluated | SARIMAX and Prophet complete with RMSE/MAPE for both | One model complete, other has minor issues | Only one model or no evaluation metrics |
| Comparison and interpretation | Thoughtful verdict with specific evidence | Comparison present, reasoning general | No comparison or no interpretation |
| Code runs without errors | Runs top to bottom cleanly | Minor errors, mostly runs | Does not run |
