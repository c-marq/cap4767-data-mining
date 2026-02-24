# Group Exercise 6 — Market Basket Analysis
**CAP4767 Data Mining with Python | Week 6 | 10 Points**

---

## Objective
Discover which products customers buy together, evaluate association rules using support, confidence, and lift, and translate the strongest rules into concrete merchandising or recommendation strategies.

---

## Your Dataset
**Online Retail II** — same transaction dataset, now used for basket analysis.

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/exercises/week06-group-exercise.ipynb`

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

> *"Amazon shows you 'Customers who bought this also bought...' How do you think they know that? And why does it matter that it is based on actual purchase behavior rather than just product categories?"*

Have the Presenter write a 2–3 sentence answer in the notebook's discussion cell.

---

## Tasks

**Step 1 — Load and prepare transaction data**
Run the setup cell. Clean the data: remove cancellations, nulls, and negative quantities. Filter to UK transactions only to reduce noise.

**Step 2 — Build the basket matrix**
Group by InvoiceNo and Description. Create a one-hot encoded basket matrix where each row is a transaction and each column is a product. Print the shape.

> 🛑 **Checkpoint:** The basket matrix should have far more columns than rows. If rows > columns, your grouping logic is likely inverted.

**Step 3 — Find frequent itemsets**
Run `apriori` with `min_support=0.02`. How many frequent itemsets did you find? Try lowering the threshold to `0.01` — how does the count change? Choose one threshold and note your reasoning.

**Step 4 — Generate association rules**
Generate rules using `metric='lift'` and `min_threshold=1.0`. Print the top 10 rules sorted by lift.

> 🛑 **Checkpoint:** Lift > 1 means the items appear together more than chance would predict. If all your lift values are exactly 1.0, check that your support threshold is not too high.

**Step 5 — Filter for actionable rules**
Filter to rules where confidence > 0.5 AND lift > 2. How many rules remain? Print them sorted by confidence descending.

**Step 6 — Interpret the top 3 rules**
In a markdown cell, translate the top 3 rules into plain English:
- What does each rule say in a sentence a store manager could understand?
- What specific action would you recommend based on each rule?

**Step 7 — Visualize the rules**
Create a scatter plot with Support on the x-axis, Confidence on the y-axis, and Lift as the color. What pattern do you observe?

---

## Share-Out (Last 10 minutes of class)
The **Presenter** walks the class through:
1. Your top 3 rules in plain English
2. One concrete recommendation for the retailer based on your strongest rule
3. One limitation of this analysis — what does it not tell you?

---

## Submission
Each student submits their own copy individually via Canvas.

**File naming:** `LastName_FirstName_Exercise06.ipynb`

**What to submit:** Completed notebook with all cells run, rules table visible, and all markdown discussion cells filled in.

---

## Rubric

| Criterion | Excellent (10) | Good (7–8) | Needs Work (3–4) |
|-----------|---------------|-----------|-----------------|
| Frequent itemsets and rules generated | Both complete, filtering applied, visualization present | Rules generated, filtering or visualization missing | Rules not generated or metrics not calculated |
| Plain English interpretation | Top 3 rules explained clearly with specific recommendations | Interpretation present, recommendations vague | No interpretation or no recommendations |
| Code runs without errors | Runs top to bottom cleanly | Minor errors, mostly runs | Does not run |
