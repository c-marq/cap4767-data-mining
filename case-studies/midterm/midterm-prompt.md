# Midterm Project: RFM + CLTV + K-Means. CAP4767 Data Mining

**Timing:** assigned with Modules 4-5. The techniques are taught in Modules 6-7 (demo, then Group Exercise 5), and the project is due after you have practiced them. Dates are in Canvas.
**Points:** 90 (Notebook 50, Memo 40). Individual work.

---

## The Scenario

You are a data analyst at a retail consulting firm. A client, a mid-size online retailer, has hired you to help them stop losing money on customers who were once valuable but have quietly drifted away. They have transaction data. They have a hunch that not all customers are equal. But they have no system for acting on that knowledge.

Your job: build one.

---

## Your Deliverables

Submit two files to Canvas by the due date.

### 1. Colab Notebook (50 points)
Using the Online Retail II dataset (sheet **"Year 2009-2010"**, see Dataset below), build the complete pipeline:

- Load and clean the data (cancellations, missing Customer IDs, zero or negative quantities and prices)
- Calculate Customer Lifetime Value (CLTV) for each customer
- Build RFM scores (Recency, Frequency, Monetary)
- Apply K-Means clustering (justify your choice of k using the Elbow Method)
- Profile each cluster: who are these customers?
- Quantify the financial value of each segment

Every step must be documented. Use markdown cells to explain what you are doing and why.

**CLTV:** total spend to date per customer (the way the Module 6 demo computes it) earns full credit, as long as you say that is what it is. A simple forward-looking estimate is welcome but not required.

### 2. Business Memo (40 points)
One page. Addressed to the client's VP of Marketing (non-technical audience).

Your memo must answer:
- How many customer segments did you find, and what defines each one?
- Which segment represents the greatest revenue risk?
- What specific actions should the client take for each segment?
- What is the estimated financial impact of acting vs. not acting?

No code. No jargon. Write for a reader who has never heard of K-Means.

There is no presentation for the midterm.

---

## Dataset

**Online Retail II** from the UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/502/online+retail+ii (CC BY 4.0).

- Download the zip, unzip it and upload `online_retail_II.xlsx` to Colab (or to your Google Drive).
- Use **only the sheet "Year 2009-2010"**: `pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2009-2010")`. Loading takes a minute or two.
- The other sheet ("Year 2010-2011") is the data from the class demo. Do not use it.
- The column names differ from the demo: `Invoice`, `Price` and `Customer ID` here are `InvoiceNo`, `UnitPrice` and `CustomerID` in the demo. Rename them if you reuse demo code.

Want a different dataset? It must have a customer ID, a transaction date and a transaction value, and most customers must buy more than once. Ask through Canvas Inbox before you start.

---

## Rubric

| Component | Excellent | Good | Needs Work |
|-----------|-----------|------|------------|
| **Notebook: Pipeline completeness** (20 pts) | All steps present, clean, documented | Most steps present, minor gaps | Missing major steps or no documentation |
| | 20 | 15 | 7 |
| **Notebook: Code quality & explanation** (30 pts) | Every cell explained, logic is clear, output is clean | Most cells explained, some gaps | Little to no explanation, or code does not run |
| | 30 | 21 | 9 |
| **Memo: Clarity & audience** (20 pts) | No jargon, compelling narrative, clear recommendations | Mostly clear, minor jargon | Technical language, unclear recommendations |
| | 20 | 15 | 7 |
| **Memo: Business impact** (20 pts) | Quantified impact, specific actions per segment | Some quantification, general recommendations | No numbers, vague recommendations |
| | 20 | 15 | 7 |

**Total: 90 points**

---

> **Remember:** A student who documents what they tried, what failed, and what they adjusted earns more points than one who submits clean code with no explanation.
