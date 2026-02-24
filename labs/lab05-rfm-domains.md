# Lab 5 — RFM Applied to a Non-Retail Domain
**CAP4767 Data Mining with Python | Individual Assignment | 20 Points**

---

## Overview
This is the bridge to your final project. You will apply the RFM framework to a domain of your choosing — outside of retail — and build the analytical foundation you can extend into your capstone. Choose your domain and dataset intentionally. You may use this exact dataset as the basis for your final project.

This lab is completed individually. Your notebook must reflect your own work and your own interpretation.

---

## Your Dataset
You choose. The only requirement is that it is **not retail or e-commerce transaction data** — you have already applied RFM there.

**Suggested domains:**

| Domain | Dataset Idea | RFM Mapping |
|--------|-------------|-------------|
| Healthcare | Patient visit records | Recency = days since last visit, Frequency = number of visits, Monetary = total charges |
| Education | LMS activity logs | Recency = days since last login, Frequency = number of logins, Monetary = assignments submitted |
| Sports | Player or team stats | Recency = days since last game, Frequency = games played, Monetary = points/goals scored |
| Social Media | Post engagement data | Recency = days since last post, Frequency = post count, Monetary = total likes/shares |
| Environmental | Sensor or event data | Recency = days since last event, Frequency = event count, Monetary = event magnitude |
| Public Safety | 311 or incident reports | Recency = days since last incident, Frequency = incident count, Monetary = severity or response time |

**Message via Canvas Inbox before starting if you are unsure whether your domain and dataset work for this lab.**

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/labs/lab05-rfm-domains.ipynb`

---

## Tasks

**Task 1 — Define your domain and RFM mapping**
Before loading any data, write a markdown cell that answers:
- What domain did you choose and why?
- What is the "entity" being analyzed (customer equivalent)?
- What does Recency, Frequency, and Magnitude mean specifically in your domain?
- What business or research question does this analysis answer?

This cell is graded. Vague answers lose points.

**Task 2 — Load and assess data quality**
Load your dataset. Document every quality issue you find. This dataset is likely not clean — that is expected and part of the exercise.

**Task 3 — Clean and prepare**
Address the quality issues. For each decision, explain your reasoning. What assumptions did you make? What did you discard and why?

**Task 4 — Calculate your RFM metrics**
Calculate the three metrics for each entity in your dataset. Print the top 10 entities by each metric separately. Do the top 10 lists make intuitive sense for your domain?

> **Reflection checkpoint:** If your Recency values are all 0 or all the same, check your snapshot date logic. If Frequency is all 1, check your groupby key.

**Task 5 — Score and segment**
Use either quintile scoring (1–5 per metric) or K-Means clustering to create segments. Choose the method that fits your domain better and justify your choice in a markdown cell.

**Task 6 — Profile and name your segments**
Calculate mean metric values per segment. Name each segment in a way that is meaningful for your domain. Create a summary table.

**Task 7 — Write your analysis and final project bridge**
In a final markdown cell (minimum 200 words), answer:
- What did you discover about your domain that would not have been obvious without this analysis?
- Which segment is most interesting or most actionable — and for whom?
- What are the limitations of applying RFM to this domain?
- How will you extend this analysis in your final project? What additional technique (forecasting, classification, market basket) could add value to this dataset?

---

## What to Document
The domain mapping in Task 1 is as important as the code. A clear, thoughtful mapping with well-reasoned segment names demonstrates genuine understanding of the framework. Process documentation throughout earns partial credit even when code has errors.

---

## Submission
Submit via Canvas.

**File naming:** `LastName_FirstName_Lab05.ipynb`

**What to submit:** Your completed Colab notebook downloaded as .ipynb. All cells must be run with outputs visible before downloading.

---

## Rubric

| Criterion | Excellent (20) | Good (14–16) | Needs Work (6–8) |
|-----------|---------------|-------------|-----------------|
| **RFM mapping and domain definition** (6 pts) | Mapping is specific and logical, entity defined clearly, business question stated | Mapping present, entity identified, question somewhat vague | Mapping missing or does not fit the domain |
| **Technical completeness** (8 pts) | All 7 tasks complete, metrics calculated correctly, segments profiled and named | Most tasks complete, segmentation or naming has minor gaps | RFM not calculated or no segmentation |
| **Analysis and final project bridge** (6 pts) | 200+ words, insight is domain-specific, limitations addressed, extension plan is concrete | Analysis present, insight somewhat general, extension mentioned but vague | Missing analysis or no final project bridge |
