# Python Quick Reference — CAP4767 Data Mining

A condensed reference for the most common patterns used in this course.

---

## Loading Data

```python
import pandas as pd

# From URL (used in all course notebooks)
url = "https://raw.githubusercontent.com/c-marq/cap4767-data-mining/main/data/filename.csv"
df = pd.read_csv(url)

df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
```

---

## Time Series

```python
# Set datetime index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Resample and rolling average
df_monthly = df.resample('M').sum()
df['rolling_30'] = df['value'].rolling(window=30).mean()

# Seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df['value'])
result.plot()

# ADF stationarity test — p > 0.05 = non-stationary
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(df['value'])
print(f"p-value: {adf_result[1]:.4f}")

# SARIMAX with auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
params = auto_arima(train, seasonal=True, m=4, suppress_warnings=True)
model = SARIMAX(train, order=params.order, seasonal_order=params.seasonal_order)
result = model.fit(disp=False)
forecast = result.forecast(steps=12)

# Prophet
from prophet import Prophet
train_df = pd.DataFrame({'ds': train.index, 'y': train.values})
model = Prophet(yearly_seasonality=True)
model.fit(train_df)
future = model.make_future_dataframe(periods=12, freq='QS')
forecast = model.predict(future)
```

---

## Regression & Classification

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Linear regression
lr = LinearRegression().fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")

# Logistic regression
clf = LogisticRegression(random_state=42).fit(X_train_scaled, y_train)
print(classification_report(y_test, clf.predict(X_test_scaled)))
```

---

## Neural Networks (Keras)

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(26, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1,  activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_split=0.2, verbose=0)
```

---

## K-Means Clustering

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)

# Elbow method
wcss = [KMeans(n_clusters=k, random_state=42).fit(X_scaled).inertia_ for k in range(1, 11)]

# Fit optimal k
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
df.groupby('Cluster')[['Recency','Frequency','Monetary']].mean().round(1)
```

---

## Market Basket Analysis

```python
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
df_encoded = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)

frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
rules.sort_values('lift', ascending=False).head(10)
```

---

## Evaluation Metrics Cheat Sheet

| Metric | Use When | Notes |
|--------|----------|-------|
| RMSE | Regression | Penalizes large errors |
| R² | Regression | % of variance explained |
| Accuracy | Classification | Balanced classes only |
| Precision | Classification | Minimize false positives |
| Recall | Classification | Minimize false negatives |
| F1 | Classification | Imbalanced classes |
| AUC-ROC | Classification | Overall discrimination |
| Support | MBA | Itemset frequency |
| Confidence | MBA | Rule reliability |
| Lift | MBA | Strength vs. random |
