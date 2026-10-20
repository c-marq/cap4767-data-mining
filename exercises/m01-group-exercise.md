# Module 1 Group Exercise — Analyzing Time-Series Stock Data
**CAP4767 Data Mining with Python | Module 1 | 10 Points**

---

## Overview
After the Module 1 demo on Apple stock data and California wildfire records, your group applies the same time-series techniques (date ranges, reindexing, resampling, and rolling windows) to a stock dataset. Before coding, the group discusses what useful information you would lose, and what you might gain, by looking at stock data once per week instead of every day. The exercise is graded on completeness, correctness, and the group discussion response. Groups self-assign four roles: Lead Coder, Data Interpreter, QA Reviewer, and Presenter.

---

## Your Dataset
**Stocks (AAPL daily OHLC, 2020)**: daily Apple stock prices for 2020 with the columns `Date`, `Open`, `High`, `Low`, and `Close`. The setup cell loads it as `stockData`; do not modify it.

Load URL: `https://raw.githubusercontent.com/c-marq/cap4767-data-mining/refs/heads/main/data/stocks.csv`

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/exercises/m01_group_exercise.ipynb`

The starter notebook has the full instructions for every task.

---

## Tasks and Points

| Step | Tasks |
|------|-------|
| 1: Explore the Data | Display the first five rows of `stockData` |
| 2: Generate Date Ranges | Every other day in 2020 · every 3 hours in January 2020 · every other Friday in 2020 |
| 3: Reindex the Data | Set `Date` as the index · reindex to Fridays only (`stockDataFridays`) · plot `Close` |
| 4: Resample the Data | Downsample to monthly means · plot `Close` |
| 5: Compute a Rolling Window | 2-week (10 trading day) rolling average on `Close` (`stocksRolling`) · plot it |
| Share-Out | The Presenter shares one key observation with the class (about 1 minute) |

The exercise is worth **10 points** in total. The notebook does not assign points to individual steps.

---

## Submission
Each student submits their own copy individually via Canvas.

**File naming:** `LastName_FirstName_Exercise01.ipynb`

**What to submit:** The completed notebook with all cells run, all markdown discussion cells filled in, and your group members listed.
