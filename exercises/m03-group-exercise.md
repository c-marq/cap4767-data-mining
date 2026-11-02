# Module 3 Group Exercise: Multiple Regression (Predict and Interpret)
**CAP4767 Data Mining with Python | Module 3 | 10 Points**

---

## Overview
Your group builds a multiple regression model on new data (not the WA housing data from the demo), evaluates it, interprets the coefficients, and presents the findings to the class in a 2-minute share-out. The steps are the same as the Module 3 demo.

Each group picks ONE dataset. Default: odd-numbered breakout rooms take **gym members**, even-numbered rooms take **credit card customers**. A group may pick the other one. Before coding, the group imagines it is the analytics team for the decision maker in the table below and discusses which columns will be the strongest predictors, which may be redundant, and what column missing from the dataset would improve predictions. Groups self-assign four roles: Lead Coder, Data Interpreter, Presenter, and QA Reviewer.

**Companion page:** the Module 3 practice worksheet walks through these same steps on the same two datasets, with hints and solutions: `https://c-marq.github.io/cap4767-data-mining/practice/m03-regression/`

---

## Your Dataset
You change one line in the starter notebook's setup cell, `dataset = "gym"` or `dataset = "card"`. The cell then loads the data and sets the target, the four number columns for Task 4, and the name of the yes/no column for Task 5. Do not change anything else in that cell.

| | Gym members | Credit card customers |
|---|---|---|
| Data | 973 rows, 15 columns, no missing values | 10,127 rows, 20 columns (the setup cell drops `CLIENTNUM` and the two `Naive_Bayes` columns) |
| Target | `Calories_Burned` (calories) | `Total_Trans_Amt` (dollars spent in the last 12 months) |
| Four number columns (Task 4) | `Session_Duration (hours)`, `Avg_BPM`, `Age`, `Fat_Percentage` | `Total_Trans_Ct`, `Total_Relationship_Count`, `Credit_Limit`, `Customer_Age` |
| Yes/no column (Task 5) | `is_male` = 1 if `Gender == "Male"`, else 0 | `premium_card` = 1 if `Card_Category != "Blue"` (Silver, Gold, Platinum), else 0 |
| Who wants the model | A fitness club chain. Its app would show members a calorie estimate after each workout. | A bank's credit card business, setting next year's rewards budget. |
| Load URL | `https://raw.githubusercontent.com/c-marq/cap4767-data-mining/refs/heads/main/data/gym_members_exercise_tracking.csv` | `https://raw.githubusercontent.com/c-marq/cap4767-data-mining/refs/heads/main/data/BankChurners.csv` |
| Source | Kaggle, Gym Members Exercise Dataset (valakhorasani), Apache 2.0. Generated data, cleaner than real club data. | Kaggle, Credit Card customers (sakshigoyal7), CC0 |

Every model uses the same split: `train_test_split(..., test_size=0.25, random_state=42)`. This is the same split as the practice worksheet.

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/exercises/m03_group_exercise.ipynb`

The starter notebook has the full instructions for every task, three checkpoints with the expected numbers for both datasets, and a troubleshooting table.

---

## Tasks and Points

| Task | What you do | Points |
|------|-------------|--------|
| 1: Explore the Data | `.info()` and `.describe()` | 1.25 |
| 2: Correlation Analysis | Correlations with your target, sorted bar chart, name the top 3 columns by absolute size and say which are negative | 1.25 |
| 3: Simple Regression Baseline | Simple linear regression on your strongest predictor (`test_size=0.25`, `random_state=42`) · report test R² | 1.25 |
| 4: Multiple Regression | The four number columns from the setup cell, same split · report test R² and compare to Task 3 | 1.25 |
| 5: Add a Yes/No Column | Create `is_male` (gym) or `premium_card` (card) · add it to the Task 4 model · report test R² | 1.25 |
| 6: Calculate RMSE | RMSE of your best model on the test rows · interpret it in calories or dollars | 1.25 |
| 7: Residual Plot | Residuals (actual minus predicted) vs predicted values with a red zero line · written interpretation (minimum 2 sentences) | 1.25 |
| 8: Coefficient Interpretation | Coefficients with feature names in a sorted DataFrame · which feature has the largest positive effect, and does the yes/no column make business sense (minimum 3 sentences) | 1.25 |
| **Total** | | **10** |

The group discussion and share-out have no point value listed in the notebook.

---

## Submission
Each student submits their own copy individually via Canvas.

**File naming:** `LastName_FirstName_Exercise03.ipynb`

**What to submit:** Completed notebook with all cells run and all markdown discussion cells filled in. Leave the `dataset` line set to the dataset your group used.
