# Group Exercise 7 — RFM Across Domains
**CAP4767 Data Mining with Python | Week 7 | 10 Points**

---

## Objective
Apply the RFM framework to a non-retail dataset and prove to yourself — and the class — that the logic of Recency, Frequency, and Magnitude is a universal pattern-discovery tool, not just a retail trick.

---

## Your Dataset
**UFO Sightings** — historical reporting data with date, location, duration, and shape fields.

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/exercises/week07-group-exercise.ipynb`

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

> *"We are about to apply RFM to UFO sightings. What would Recency, Frequency, and Magnitude (or 'Monetary') mean in this context? Who or what is the 'customer'? What business question could this answer?"*

Have the Presenter write your group's mapping in the notebook's discussion cell before touching any code.

---

## Tasks

**Step 1 — Load and assess data quality**
Run the setup cell. This dataset is intentionally messy. Document every data quality issue you find: missing values, inconsistent formats, impossible values, duplicates. Do not clean yet — just catalogue.

> 🛑 **Checkpoint:** You should find at least 4 distinct data quality issues. If you find fewer than 3, look harder — check dtypes, value counts on categorical columns, and the range of numeric fields.

**Step 2 — Clean the data**
Address the quality issues from Step 1. For each issue, document what you did and why in a markdown cell. What assumptions did you have to make?

**Step 3 — Map RFM to this domain**
Using the definitions your group agreed on in the Discussion, calculate:
- **Recency:** How recently was the last sighting reported from this location/state?
- **Frequency:** How many sightings have been reported from this location/state?
- **Magnitude:** What is the typical duration of sightings from this location/state?

**Step 4 — Calculate RFM scores and cluster**
Scale the three metrics. Run K-Means with your chosen k (use the Elbow Method). Profile each cluster.

> 🛑 **Checkpoint:** Your cluster profiles should look meaningfully different from each other. If all clusters have nearly identical means, try a different k or check your scaling.

**Step 5 — Name and interpret your segments**
Give each cluster a name that makes sense for this domain. What does each cluster represent? Is there a cluster that would interest a researcher? A journalist? A government agency?

**Step 6 — Reflect on the framework**
In a markdown cell, answer as a group:
- How well did RFM translate to this non-retail domain?
- What did you have to stretch or approximate to make it work?
- Name one other non-retail domain where you think this framework would apply cleanly — and one where it would not work at all.

**Step 7 — Bridge to your final project**
Each group member writes one sentence in the notebook: *"For my final project, I am considering applying this framework to _________ because _________."*

---

## Share-Out (Last 10 minutes of class)
The **Presenter** walks the class through:
1. Your RFM mapping — what did Recency, Frequency, and Magnitude mean in this context?
2. Your most interesting cluster — who would care about it and why?
3. One domain idea from your group for the final project

---

## Submission
Each student submits their own copy individually via Canvas.

**File naming:** `LastName_FirstName_Exercise07.ipynb`

**What to submit:** Completed notebook with all cells run, cluster profiles visible, and all markdown cells filled in including the final project bridge sentence.

---

## Rubric

| Criterion | Excellent (10) | Good (7–8) | Needs Work (3–4) |
|-----------|---------------|-----------|-----------------|
| RFM mapped and calculated correctly for this domain | Mapping is logical, all three metrics calculated, clusters profiled | Mapping present, one metric weak or missing | RFM not mapped to domain or not calculated |
| Data quality documentation | All issues identified and cleaning decisions documented | Most issues caught, some decisions not explained | No quality documentation |
| Reflection and final project bridge | Thoughtful reflection, specific domain idea with reasoning | Reflection present, bridge sentence vague | Missing reflection or bridge sentence |
