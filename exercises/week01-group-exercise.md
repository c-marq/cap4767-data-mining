# Group Exercise 1 — Time Series Foundations
**CAP4767 Data Mining with Python | Week 1 | 10 Points**

---

## Objective
Explore a time series dataset as a team. Load the data, visualize the trend and seasonality, and interpret what the patterns mean for a real business decision. This exercise builds the foundational intuition you will need before we add forecasting models in Week 2.

---

## Your Dataset
**Australian Domestic Tourism** — quarterly holiday trips by region, 1998–2017.

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/exercises/week01-group-exercise.ipynb`

---

## Group Roles
Assign one role per person before you start. Roles rotate each week.

| Role | Responsibility |
|------|---------------|
| **Lead Coder** | Drives the notebook, shares screen, types the code |
| **Data Interpreter** | Narrates what each output means in plain English |
| **QA Reviewer** | Catches errors, checks outputs match checkpoints |
| **Presenter** | Owns the share-out at the end of class |

Groups of 3: Lead Coder covers QA Reviewer duties.

---

## Discussion First (5 minutes before opening Colab)
As a group, answer this before writing a single line of code:

> *"You work for a hotel chain in Queensland, Australia. A VP asks: 'Should we hire more seasonal staff this summer?' What data would you need to answer that question confidently — and what patterns would you look for?"*

Have the Presenter write a 2–3 sentence answer in the notebook's discussion cell. You will revisit this answer at the end.

---

## Tasks

**Step 1 — Load and inspect the data**
Run the setup cell. Then explore the dataset: how many rows, how many columns, what are the date ranges, and what does each column represent?

**Step 2 — Set the datetime index**
Convert the date column to datetime format and set it as the index. Confirm by printing the first and last five rows.

> 🛑 **Checkpoint:** Your index should be a DatetimeIndex. Run `df.index` — if you see integers, you have not set the index yet.

**Step 3 — Plot the raw time series**
Plot total quarterly trips over the full date range. Give the chart a title and labeled axes. What trend do you observe?

**Step 4 — Decompose the series**
Use `seasonal_decompose` to break the series into trend, seasonality, and residual components. Plot all four panels.

> 🛑 **Checkpoint:** You should see four stacked subplots. If you see an error about frequency, check that your datetime index has a consistent quarterly frequency.

**Step 5 — Interpret the components**
In a markdown cell, answer as a group:
- What is the direction of the trend?
- How strong is the seasonality? Which quarter peaks?
- Does the residual look random, or do you see a pattern?

**Step 6 — Test for stationarity**
Run the ADF test on the raw series. Is it stationary? What does that mean for forecasting?

**Step 7 — Revisit your discussion answer**
Return to the VP's question from the Discussion section. Does what you found in the data support hiring more seasonal staff? Update your answer with evidence from the decomposition.

---

## Share-Out (Last 10 minutes of class)
The **Presenter** walks the class through:
1. One visual that best tells the story of this dataset
2. Your answer to the VP's question — with evidence
3. One thing your group is still uncertain about

---

## Submission
Each student submits their own copy individually via Canvas.

**File naming:** `LastName_FirstName_Exercise01.ipynb`

**What to submit:** The completed notebook with all cells run and all markdown discussion cells filled in.

---

## Rubric

| Criterion | Excellent (10) | Good (7–8) | Needs Work (3–4) |
|-----------|---------------|-----------|-----------------|
| All tasks completed with outputs visible | All 7 steps complete, outputs clean | Most steps complete, minor gaps | Several steps missing or no outputs |
| Discussion and interpretation cells filled | Thoughtful, specific, evidence-based | Present but general | Missing or one word answers |
| Code runs without errors | Runs top to bottom cleanly | Minor errors, mostly runs | Does not run |
