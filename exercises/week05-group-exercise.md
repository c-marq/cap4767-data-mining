# Group Exercise 5 — RFM + K-Means Customer Segmentation
**CAP4767 Data Mining with Python | Week 5 | 10 Points**

---

## Objective
Calculate RFM scores, compute Customer Lifetime Value, and use K-Means clustering to discover natural customer segments in a retail transaction dataset. Then translate those segments into a business strategy your group can defend.

---

## Your Dataset
**Online Retail II** — 500,000+ transactions from a UK-based retailer, 2009–2011.

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/exercises/week05-group-exercise.ipynb`

---

## Group Roles
Rotate from last week.

| Role | Responsibility |
|------|---------------|
| **Lead Coder** | Drives the notebook, shares screen, types the code |
| **Data Interpreter** | Narrates what each output means in plain English |
| **QA Reviewer** | Catches errors, checks outputs match checkpoints |
| **Presenter** | Owns the share-out at the end of class |

---

## Discussion First (5 minutes before opening Colab)
As a group, answer this before writing a single line of code:

> *"Your company has 50,000 customers and a marketing budget for 5,000 personalized emails. How would you decide which 5,000 customers to contact — and what would you say to each group?"*

Have the Presenter write a 2–3 sentence answer in the notebook's discussion cell.

---

## Tasks

**Step 1 — Load and clean the data**
Run the setup cell. Remove cancelled orders (InvoiceNo starting with 'C'), negative quantities, and missing CustomerIDs. Print how many rows remain after cleaning.

> 🛑 **Checkpoint:** You should have significantly fewer rows than the raw dataset. If the row count did not change, your filter did not run.

**Step 2 — Calculate RFM metrics**
Set a snapshot date of one day after the last transaction. Calculate for each customer:
- **Recency:** Days since last purchase
- **Frequency:** Number of unique invoices
- **Monetary:** Total spend

**Step 3 — Calculate Customer Lifetime Value**
Add a CLTV column: `Frequency × Monetary / Recency`. Print the top 10 customers by CLTV.

**Step 4 — Find the optimal number of clusters**
Scale the RFM features. Run the Elbow Method for k = 1 to 10. Plot the WCSS curve. As a group, agree on the optimal k and note your reasoning in a markdown cell.

> 🛑 **Checkpoint:** The elbow should be visible between k=3 and k=6. If the curve is perfectly smooth with no bend, check that your data is scaled.

**Step 5 — Fit K-Means and assign segments**
Fit K-Means with your chosen k. Add a `Cluster` column to the RFM dataframe.

**Step 6 — Profile each cluster**
Calculate the mean Recency, Frequency, Monetary, and CLTV per cluster. Name each cluster based on its profile (e.g., Champions, At Risk, Lost, New Customers).

**Step 7 — Revisit your discussion answer**
Return to the marketing budget question. Using your cluster profiles, which segment gets the 5,000 emails — and what is the message? Use specific numbers from your cluster profiles to justify your answer.

---

## Share-Out (Last 10 minutes of class)
The **Presenter** walks the class through:
1. The cluster profile table — mean RFM values per segment
2. Your segment names and the reasoning behind each name
3. Your answer to the marketing budget question with supporting numbers

---

## Submission
Each student submits their own copy individually via Canvas.

**File naming:** `LastName_FirstName_Exercise05.ipynb`

**What to submit:** Completed notebook with all cells run, cluster profile table visible, and all markdown discussion cells filled in.

---

## Rubric

| Criterion | Excellent (10) | Good (7–8) | Needs Work (3–4) |
|-----------|---------------|-----------|-----------------|
| RFM, CLTV, and K-Means complete | All three calculated correctly, clusters profiled and named | Most steps complete, minor calculation issues | Missing major components |
| Business interpretation | Segment names justified, marketing answer uses specific numbers | Interpretation present, somewhat general | No business interpretation |
| Code runs without errors | Runs top to bottom cleanly | Minor errors, mostly runs | Does not run |
