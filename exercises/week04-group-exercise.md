# Group Exercise 4 — Customer Churn: Logistic Regression + Neural Network
**CAP4767 Data Mining with Python | Week 4 | 10 Points**

---

## Objective
Build two classification models to predict customer churn and compare them head-to-head. Your group will develop an opinion on when the added complexity of a neural network is actually worth it — and when it is not.

---

## Your Dataset
**IBM Telco Customer Churn** — same dataset from Week 3, now used for its intended purpose: predicting which customers will leave.

Starter notebook: `https://github.com/c-marq/cap4767-data-mining/blob/main/exercises/week04-group-exercise.ipynb`

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

> *"A neural network sounds more impressive than logistic regression. Should you always use the more complex model? What might you lose by doing so?"*

Have the Presenter write a 2–3 sentence answer in the notebook's discussion cell.

---

## Tasks

**Step 1 — Load, prepare, and handle class imbalance**
Run the setup cell. Check the churn rate in the dataset. Is it balanced? Discuss as a group whether class imbalance could affect your model — note your conclusion in a markdown cell.

**Step 2 — Feature engineering and scaling**
Encode categoricals, drop leakage columns, split 80/20, and scale. Use the same pipeline from Week 3 as your starting point.

> 🛑 **Checkpoint:** Print `y_train.value_counts()`. If one class is more than 3x the other, you have class imbalance. Note this — it will matter when you interpret your results.

**Step 3 — Build the logistic regression baseline**
Fit a LogisticRegression model. Print the classification report and confusion matrix. This is your baseline — every other model has to beat it to justify the added complexity.

**Step 4 — Build the neural network**
Build a Keras Sequential model with:
- Input layer matching your feature count
- Two hidden layers (try 26 and 15 neurons, ReLU activation)
- Output layer (1 neuron, sigmoid activation)

Compile with `adam` optimizer and `binary_crossentropy` loss. Train for 100 epochs with a 20% validation split.

> 🛑 **Checkpoint:** Plot training vs. validation loss. If validation loss starts rising while training loss keeps falling, your model is overfitting. Note this in a markdown cell.

**Step 5 — Evaluate the neural network**
Print the classification report for the neural network on the test set. Build a comparison table: logistic regression vs. neural network across accuracy, precision, recall, and F1 for the churn class.

**Step 6 — Calculate AUC-ROC for both models**
Plot both ROC curves on the same chart. Which model has a higher AUC?

**Step 7 — Declare a winner**
In a markdown cell, your group must take a position:
- Which model would you recommend to the telecom company?
- Does the performance difference justify the added complexity of the neural network?
- What would you need to see to change your recommendation?

---

## Share-Out (Last 10 minutes of class)
The **Presenter** walks the class through:
1. The comparison table — which model wins on which metric
2. The ROC curve plot
3. Your group's recommendation and the business reasoning behind it

---

## Submission
Each student submits their own copy individually via Canvas.

**File naming:** `LastName_FirstName_Exercise04.ipynb`

**What to submit:** Completed notebook with all cells run, comparison table visible, and all markdown discussion cells filled in.

---

## Rubric

| Criterion | Excellent (10) | Good (7–8) | Needs Work (3–4) |
|-----------|---------------|-----------|-----------------|
| Both models built and evaluated | Logistic and ANN complete with full comparison | One model complete, comparison has minor gaps | Only one model or no comparison |
| Business recommendation | Specific, metric-backed, addresses complexity tradeoff | Recommendation present, reasoning somewhat general | No recommendation or no reasoning |
| Code runs without errors | Runs top to bottom cleanly | Minor errors, mostly runs | Does not run |
