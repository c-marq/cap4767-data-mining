# Module 9 Group Exercise — RFM + K-Means on a Non-Retail Domain
**CAP4767 Data Mining with Python | Module 9 | 10 Points**

---

## Overview
Your group applies the universal RFM + K-Means pipeline to a non-retail domain dataset. You define your own metric mapping, run the clustering pipeline, name your clusters, and present one intervention recommendation per cluster. Before coding, the group discusses what Recency and Magnitude mean in your chosen domain and what an outlier cluster would most likely represent. The deliverable is the completed notebook plus a 3-minute presentation, and groups self-assign four roles: Lead Coder, Data Interpreter, Presenter, and QA Reviewer.

---

## Your Dataset
Each group selects **one** dataset. No two groups should use the same dataset. Run the setup cell without modifying it, then uncomment ONE option in the dataset selection cell.

| Option | Dataset | Entity | Suggested R / F / M |
|--------|---------|--------|---------------------|
| A | Gym Members | Member | Experience_Level (proxy) / Workout_Frequency / Calories_Burned |
| B | Blood Donation | Donor | Recency (months) / Frequency (donations) / Monetary (cc blood) |
| C | UFO Sightings (by city) | City | Days since last report / Report count / Median duration |

Load URLs:
- Option A: `https://raw.githubusercontent.com/c-marq/cap4767-data-mining/refs/heads/main/data/gym_members_exercise_tracking.csv`
- Option B: `https://raw.githubusercontent.com/c-marq/cap4767-data-mining/refs/heads/main/data/Blood%20Transfusion%20Service%20data.csv`
- Option C: `https://raw.githubusercontent.com/c-marq/cap4767-data-mining/main/data/ufo-sightings.csv` (read with the column names listed in the notebook's `ufo_cols`)

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/exercises/m09_group_exercise.ipynb`

The starter notebook has the full instructions for every task.

---

## Tasks and Points

| Task | What you do | Points |
|------|-------------|--------|
| 1: Load and Explore | Shape, column names, first 5 rows, and `.describe()` · confirm your planned R/F/M columns are present and numeric | 1 |
| 2: Define and Calculate RFM Metrics | State your RFM mapping (column and meaning for each of R, F, M) · calculate the metrics, aggregating first if needed | 2 |
| 3: Check Correlations | Correlation matrix for R, F, M · are any pairs correlated above 0.85? | 1 |
| 4: Standardize and Elbow Method | `StandardScaler` · elbow method for k = 2 to 8 · chosen k with justification | 2 |
| 5: Fit K-Means and Profile Clusters | Fit K-Means, assign labels, print profiles · best cluster, most concerning cluster, any outlier cluster | 2 |
| 6: Visualize | Scatter plot of Frequency vs Magnitude, colored by cluster | 1 |
| 7: Name and Recommend | Table with a domain name, size, and recommendation for each cluster | 1 |
| **Total** | | **10** |

---

## Submission
Each student submits their own copy individually via Canvas.

**File naming:** `LastName_FirstName_Exercise07.ipynb`

**What to submit:** Completed notebook with all cells run, cluster profiles visible, and all markdown cells filled in.
