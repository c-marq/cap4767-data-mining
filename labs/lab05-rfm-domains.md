# Lab 05 — RFM Applied to a Non-Retail Domain
**CAP4767 Data Mining with Python | Individual Assignment | 20 Points (+2 bonus)**

---

## Overview
In this lab you apply the universal RFM + K-Means framework to a non-retail domain of your choice, then write a 1-page domain brief that translates your findings for a non-technical decision-maker. The lab is completed individually. The domain and dataset you choose may be carried forward as the foundation for your final project.

---

## Your Dataset
Choose **one** of the following. Uncomment ONE option in the dataset selection cell.

| Option | Dataset | Entity | Suggested R / F / M |
|--------|---------|--------|---------------------|
| A | Gym Members | Member | Experience_Level (proxy) / Workout_Frequency / Calories_Burned |
| B | Blood Donation | Donor | Recency (months) / Frequency (count) / Monetary (cc) |
| C | Your Own | Your choice | Must be non-retail, 100+ entities, clear R/F/M mapping |

Load URLs:
- Option A: `https://raw.githubusercontent.com/c-marq/cap4767-data-mining/refs/heads/main/data/gym_members_exercise_tracking.csv`
- Option B: `https://raw.githubusercontent.com/c-marq/cap4767-data-mining/refs/heads/main/data/Blood%20Transfusion%20Service%20data.csv`
- Option C: the notebook has a `pd.read_csv("YOUR_URL_HERE")` placeholder for your own dataset's URL.

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/labs/lab05_rfm_domains.ipynb`

The starter notebook has the full instructions for every task.

---

## Tasks and Points

| Task | What you do | Points |
|------|-------------|--------|
| 1: Select and Justify Your Domain | Define your entity, your R, F, and M columns (what "high" and "low" mean, and whether lower Recency is better or worse), and why the mapping is meaningful | 2 |
| 2: Load and Explore | Shape, `.dtypes`, first 5 rows, and `.describe()` for R, F, M · note data quality concerns | 2 |
| 3: Calculate and Score RFM | Calculate the three metrics (aggregate if needed) · score each with `pd.qcut()` (or `pd.cut()` with fewer bins) · handle Recency reversal · display the scored table for at least 10 entities | 3 |
| 4: Check Metric Independence | Correlation matrix for R, F, M · are any pairs above 0.85? | 1 |
| 5: Cluster with K-Means | `StandardScaler` · elbow method for k = 2 to 8 · fit K-Means with your chosen k and print profiles | 3 |
| 6: Visualize (2 charts) | Scatter of Frequency vs Magnitude colored by cluster · cluster profile chart (heatmap or grouped bar) | 2 |
| 7: Name and Interpret Clusters | Domain name, size, profile summary, and recommended action for each cluster | 3 |
| 8: Domain Brief (at least 300 words) | 1-page memo to a non-technical decision-maker. Sub-rubric: problem statement (1) · method explained (1) · findings with visualization (1) · actionable recommendations (1) · professional tone (1) | 5 |
| 9: Reflection | 4–5 sentences on applying the framework to a non-retail domain | 2 |
| **Total** | | **23 (capped at 20 + 2 bonus)** |
| Bonus Challenge | Run the full pipeline twice, once with your original Magnitude metric and once with a different one, and compare cluster profiles side by side (+2 extra credit, no scaffolding) | +2 |

---

## What to Document
Explain your reasoning in the markdown cells, especially your R/F/M mapping (Task 1), your data quality notes (Task 2), and the names you give your clusters (Task 7).

---

## Submission
Submit via Canvas.

**File naming:** `LastName_FirstName_Lab05.ipynb`

**What to submit:** Your completed Colab notebook downloaded as .ipynb. All cells must be run with outputs visible before downloading.
