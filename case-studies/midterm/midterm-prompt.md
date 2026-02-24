# Midterm Project — CAP4767 Data Mining

**Assigned:** Week 4, Session 1
**Due:** Week 5, Session 2 (present in class)
**Points:** 130

---

## The Scenario

You are a data analyst at a retail consulting firm. A client — a mid-size e-commerce company — has hired your team to help them stop losing money on customers who were once valuable but have quietly drifted away. They have transaction data. They have a hunch that not all customers are equal. But they have no system for acting on that knowledge.

Your job: build one.

---

## Your Deliverables

Submit three items to Canvas before class on the due date. Be prepared to present.

### 1. Colab Notebook (50 points)
Using the Online Retail II dataset (or an approved alternative), build the complete pipeline:

- Load and clean the data
- Calculate Customer Lifetime Value (CLTV)
- Build RFM scores (Recency, Frequency, Monetary)
- Apply K-Means clustering (justify your choice of k using the Elbow Method)
- Profile each cluster — who are these customers?
- Quantify the financial value of each segment

Every step must be documented. Use markdown cells to explain what you are doing and why.

### 2. Business Memo (40 points)
One page. Addressed to the client's VP of Marketing (non-technical audience).

Your memo must answer:
- How many customer segments did we find, and what defines each one?
- Which segment represents the greatest revenue risk?
- What specific actions should the client take for each segment?
- What is the estimated financial impact of acting vs. not acting?

No code. No jargon. Write for a reader who has never heard of K-Means.

### 3. Presentation (40 points)
5–8 minutes. Hard stop at 8 minutes.

Tell the story of your analysis. What did you find? What should the client do? Why does it matter?
Slides are optional — the notebook can be your visual aid.

---

## Dataset

**Primary:** Online Retail II — [Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)

**Approved alternatives** (request via Canvas before starting):
- Any dataset from the Dr. Lee CLTV article addendum
- Brazilian E-Commerce (Kaggle)
- Any dataset with customer ID, transaction date, and transaction value

---

## Rubric

| Component | Excellent | Good | Needs Work |
|-----------|-----------|------|------------|
| **Notebook — Pipeline completeness** (20 pts) | All steps present, clean, documented | Most steps present, minor gaps | Missing major steps or no documentation |
| | 20 | 14–16 | 6–8 |
| **Notebook — Code quality & explanation** (30 pts) | Every cell explained, logic is clear, output is clean | Most cells explained, some gaps | Little to no explanation |
| | 30 | 20–22 | 8–10 |
| **Memo — Clarity & audience** (20 pts) | No jargon, compelling narrative, clear recommendations | Mostly clear, minor jargon | Technical language, unclear recommendations |
| | 20 | 14–16 | 6–8 |
| **Memo — Business impact** (20 pts) | Quantified impact, specific actions per segment | Some quantification, general recommendations | No numbers, vague recommendations |
| | 20 | 14–16 | 6–8 |
| **Presentation — Delivery** (20 pts) | Confident, clear, within time limit | Mostly clear, slightly over/under time | Difficult to follow or significantly off time |
| | 20 | 14–16 | 6–8 |
| **Presentation — Findings** (20 pts) | Tells the full story, connects to business decision | Covers most findings, partial business connection | Only describes steps, no business context |
| | 20 | 14–16 | 6–8 |

**Total: 130 points**

---

> **Remember:** A student who documents what they tried, what failed, and what they adjusted earns more points than one who submits clean code with no explanation.
