*PART I — TIME SERIES: UNDERSTANDING AND FORECASTING PATTERNS OVER TIME*
# Chapter 2 — Time Series Forecasting: SARIMAX and Prophet

<img width="1408" height="768" alt="image" src="https://github.com/user-attachments/assets/40aa072d-e44d-4102-9aa9-0304c57bcd14" />

<!-- [IMAGE: images/ch02/fig-2-0-analyst-forecast-comparison.png]
Alt text: A data analyst seated at a wide curved monitor displaying a time series chart with historical data points on the left and two diverging forecast lines — one amber, one blue — extending into a shaded forecast region on the right, her hand hovering between the two projected paths.
Nano Banana Pro Prompt: "A woman in her early 30s wearing a dark navy blazer, seated at a wide curved monitor in a modern open-plan office at night. On the screen, a large time series chart dominates the display: a solid white line of historical data points traces a rising wave pattern with clear seasonal peaks and valleys across the left two-thirds of the screen. At the point where the solid line ends, two dashed forecast lines diverge into a lightly shaded region labeled with a small translucent placard reading 'Forecast' — one line glows warm amber and curves upward, the other glows electric blue and curves slightly lower, both continuing the seasonal wave pattern but separating gradually as they extend to the right edge of the screen. The analyst's right hand hovers between the two lines on the screen, fingers slightly apart as if weighing the two paths. Her desk is minimal — a single coffee mug and a small notebook. Floor-to-ceiling windows behind her show a nighttime city skyline with blue and white building lights. The lighting is cool blue from the monitor washing across her face and hands, mixed with warm amber from a desk lamp on the left, creating a split-tone effect. Style: modern tech editorial illustration with crisp vector-influenced rendering, subtle gradients, and a dark navy-and-electric-blue color palette with amber accents. Composition is a medium shot from a slight over-the-shoulder angle, placing the monitor's time series chart and the analyst's hovering hand in equal visual weight. No text other than the single word 'Forecast' on the translucent placard on the screen."
-->

---

## 2.1 The Business Problem

It's January in Miami Beach, and the revenue management team at a 400-room oceanfront hotel is building next year's budget. Every decision cascading from that budget — how many housekeepers to hire, when to schedule the $2.3 million lobby renovation, what room rate to set for Art Basel weekend versus a sleepy Tuesday in September — depends on one number: how many guests will walk through the door each quarter.

The team has sixteen years of quarterly booking data. They can *see* the patterns. Summer is always strong. Q1 always spikes with snowbirds and spring breakers. Q3 dips when locals escape the heat and international visitors stay home. But seeing the pattern and *quantifying* it precisely enough to bet millions of dollars on it are two very different things.

A forecast that runs 10% too high means overstaffing, excess food inventory, and rooms prepped for guests who never arrive. At the hotel's average cost structure, that's roughly $380,000 in wasted operational spend per quarter. A forecast that runs 10% too low is worse — it means turning away guests during peak season. At $350 per night with an average 2.3-night stay, just 50 turned-away bookings per week across a peak quarter costs the hotel over $1 million in lost revenue, plus the lifetime value of guests who book with a competitor instead and never come back.

The question isn't *whether* to forecast. The question is how, and how much to trust the number you get. In this chapter, you'll learn two fundamentally different approaches to time series forecasting — SARIMAX (the classical statistical method) and Prophet (Meta's modern alternative) — and you'll learn how to put them side by side and decide which one to trust with your budget.

In Chapter 1, you learned to *read* a time series — to decompose it, test for stationarity, and spot the trend and seasonal patterns hidden in the noise. That was diagnosis. This is treatment.

---

## 2.2 The Concept

### From Description to Prediction

Everything you built in Chapter 1 was preparation for this moment. You learned that time series data has three components — trend, seasonality, and residuals. You learned to test for stationarity with the Augmented Dickey-Fuller test. You used rolling windows to smooth noise and seasonal decomposition to pull the series apart and examine each piece.

All of that was *descriptive*. You were reading the story the data had already told. Forecasting asks a harder question: *what happens next?*

Think of it this way. Chapter 1 was studying the weather history for Miami — you learned that August is hot, January is mild, and hurricanes cluster in September and October. Forecasting is the 10-day forecast on your phone: it uses those same historical patterns, projects them forward, and becomes less certain the further out you look. The five-day forecast is pretty reliable. The ten-day forecast is a rough guide. The thirty-day forecast is barely better than guessing. The same principle applies to every time series model you'll build.

```mermaid
graph LR
    ch1_trend["Ch 1: Trend Detection"] --> forecast_input["Forecasting<br/>Input Layer"]
    ch1_season["Ch 1: Seasonal Decomposition"] --> forecast_input
    ch1_station["Ch 1: Stationarity (ADF Test)"] --> forecast_input
    ch1_roll["Ch 1: Rolling Windows"] --> forecast_input
    forecast_input --> sarimax["SARIMAX Model"]
    forecast_input --> prophet["Prophet Model"]
    sarimax --> comparison["Model Comparison"]
    prophet --> comparison
    comparison --> decision["Business Decision"]
```

**Figure 2.1: From Chapter 1 to Chapter 2** — Everything you learned in Chapter 1 (trend detection, seasonal decomposition, stationarity testing, rolling windows) becomes input to the forecasting models you'll build in this chapter.

The conceptual leap is small but important: instead of summarizing the past, we're using the past to generate *specific numerical predictions* about the future — predictions precise enough that a hotel revenue manager can base a multimillion-dollar budget on them.

### ARIMA: Three Components, One Model

ARIMA stands for **AutoRegressive Integrated Moving Average**. That's a mouthful, but each word describes a specific modeling strategy. Let's take them one at a time.

**AR — AutoRegressive (p).** The autoregressive component predicts the current value by looking at a fixed number of previous values, called *lags*. If p = 3, the model learns a relationship between today's value and the three most recent past values — essentially asking, "Given where we were yesterday, the day before, and the day before that, where are we likely to be today?" The model learns coefficients for each lag, quantifying exactly how much influence each past observation has. Think of it like predicting tomorrow morning's traffic on I-95 by checking the last three mornings. If Monday, Tuesday, and Wednesday were all heavy, Thursday probably will be too.

**I — Integrated (d).** The integrated component handles non-stationarity through *differencing*. Remember from Chapter 1 that most forecasting models require stationary data — a stable mean and variance over time. Differencing subtracts each value from the one before it: `difference = current value − previous value`. If your original series has an upward trend, differencing removes that trend by converting the series into *changes* rather than *levels*. If d = 1, we difference once. If d = 2, we difference the already-differenced series a second time. This is the cleanup step — it makes the data stationary so the AR and MA components can do their jobs.

**MA — Moving Average (q).** The moving average component learns from past *forecast errors*, not past values. If the model predicted 5,200 trips last quarter but the actual number was 5,400, that's an error of +200. The MA component uses those past errors to adjust future predictions. If q = 2, the model considers the last two prediction errors when making its next forecast. This is the self-correction mechanism — the model gets better by learning from its own mistakes.

```mermaid
graph TD
    raw["Raw Time Series<br/>(non-stationary)"] --> diff["Differencing (d)<br/>Remove trend"]
    diff --> stationary["Stationary Series"]
    stationary --> ar["AR Component (p)<br/>Learn from past values"]
    stationary --> ma["MA Component (q)<br/>Learn from past errors"]
    ar --> combine["Combined Forecast"]
    ma --> combine
    combine --> output["Prediction"]
```

**Figure 2.2: How ARIMA Works** — Differencing (I) makes the series stationary. Then autoregression (AR) learns from past values and moving average (MA) corrects from past errors. Together, they produce a forecast.

Here's the key insight: these three components are not three separate models bolted together. They're three *strategies* that work simultaneously within one unified mathematical framework. The parameters (p, d, q) tell the model how much to lean on each strategy.

💡 **Key Insight:** The (p, d, q) parameters aren't arbitrary tuning knobs — each one represents a fundamentally different approach to modeling the data. AR says "the past predicts the future." I says "remove the trend first." MA says "learn from your mistakes." The art of ARIMA is finding the right balance.

### SARIMA: Adding Seasonality

ARIMA handles trend and short-term patterns, but it can't capture repeating seasonal cycles. SARIMA (**Seasonal ARIMA**) fixes this by adding a second set of parameters — (P, D, Q, m) — that operate on the seasonal level.

If ARIMA is predicting next week's I-95 traffic from recent days, SARIMA adds another layer: "But also, every year during Art Basel week, traffic spikes 40% — and that spike has happened every December for the last fifteen years." The seasonal parameters model dependencies that repeat at a fixed interval.

The four seasonal parameters work just like their non-seasonal counterparts, but across seasons instead of adjacent observations. **P** is the seasonal autoregressive order — how many *past seasons* to learn from. **D** is the seasonal differencing order — removing the seasonal trend. **Q** is the seasonal moving average order — correcting from past *seasonal* prediction errors. And **m** is the length of the seasonal cycle: m = 4 for quarterly data (four quarters per year), m = 12 for monthly data (twelve months per year), m = 7 for daily data with weekly patterns.

```mermaid
graph LR
    subgraph non_seasonal ["Non-Seasonal (p, d, q)"]
        ar_ns["AR: Learn from<br/>recent values"]
        i_ns["I: Difference to<br/>remove trend"]
        ma_ns["MA: Correct from<br/>recent errors"]
    end
    subgraph seasonal ["Seasonal (P, D, Q, m)"]
        ar_s["Seasonal AR: Learn<br/>from past seasons"]
        i_s["Seasonal I: Remove<br/>seasonal trend"]
        ma_s["Seasonal MA: Correct<br/>seasonal errors"]
    end
    non_seasonal --> merge["Merged<br/>Forecast"]
    seasonal --> merge
    merge --> output["Final Prediction"]
```

**Figure 2.3: SARIMA's Dual Structure** — Non-seasonal parameters handle short-term patterns; seasonal parameters handle repeating cycles. Both tracks merge into a single forecast.

⚠️ **Common Pitfall:** Students often think "seasonal" just means summer versus winter. It doesn't. Seasonality is *any* repeating pattern at a fixed interval. A restaurant with higher Friday sales has weekly seasonality (m = 7). A retailer with December spikes has yearly seasonality (m = 12). A quarterly tourism dataset has m = 4. The value of m must match the actual cycle in your data.

### SARIMAX: External Variables

**SARIMAX** (Seasonal ARIMA with eXogenous variables) adds one more capability: it can incorporate external variables that influence the forecast but aren't part of the time series itself. Think of exogenous variables as outside forces — a hurricane warning, a major convention scheduled downtown, a new competitor opening, or a global pandemic. These events affect your time series but can't be predicted by looking at the series alone.

In this chapter, we won't use exogenous variables because the Australian tourism dataset doesn't include them. But it's important to know the capability exists. When you encounter a dataset where external events clearly matter — and you will — SARIMAX is ready for that.

> **Scenario: The Renovation Window**
>
> A boutique hotel chain with three properties in Wynwood wants to renovate 15 rooms at their flagship location. The renovation will take a full quarter — those rooms will be completely offline, representing a revenue loss of roughly $189,000 at their average rate. The general manager needs to identify the quarter with the lowest predicted occupancy over the next two years so the renovation causes minimal damage.
>
> She runs SARIMAX on four years of quarterly booking data. The model's (1,1,1)(1,1,1,4) parameters capture both the gradual upward trend (Wynwood is getting more popular) and the seasonal dip every Q3 (summer heat drives tourists to the coast). The model forecasts that Q3 2027 will have the lowest occupancy of any quarter in the next eight — and it'll be 12% lower than Q3 2026 because a competitor is opening a new property nearby (she adds this as an exogenous variable). The renovation gets scheduled, and the revenue team adjusts their annual budget accordingly.
>
> *Connection:* The ARIMA parameter values aren't abstract math. They directly determine which quarter the GM picks — and whether the hotel risks $189,000 or $340,000 in lost revenue.

### auto_arima: Let the Algorithm Search

With six parameters to set (p, d, q, P, D, Q) plus the seasonal period m, manually testing every combination would take hours. The `auto_arima` function from the `pmdarima` library automates this search.

auto_arima systematically tests parameter combinations and evaluates each one using the **Akaike Information Criterion (AIC)**. AIC measures model quality with a built-in penalty for complexity — if adding another parameter doesn't meaningfully improve the fit, AIC punishes the more complex model. This prevents overfitting: a model with 12 parameters that's only slightly better than a model with 4 parameters will have a *worse* AIC because the extra complexity isn't justified.

Think of it like a sous chef systematically working through spice combinations for a new dish. She doesn't just find the combination that tastes best — she also penalizes recipes that use 15 spices when 5 would do, because complexity means higher ingredient costs, longer prep time, and more things that can go wrong. A lower AIC is better, just like a simpler recipe that still tastes great is better than an unnecessarily complicated one.

🔧 **Pro Tip:** auto_arima doesn't guarantee the absolute best model — it searches within a defined parameter space and uses AIC to rank candidates. Think of it as finding an excellent solution efficiently, not an exhaustive search of every possibility. For this course, that's exactly what we need.

### Prophet: Meta's Modern Alternative

Prophet, developed by Meta (formerly Facebook), takes a completely different approach to forecasting. While SARIMAX models the time series through its own autocorrelation structure, Prophet treats the forecast as a *curve-fitting* problem with three additive components:

**Trend:** A piecewise linear function (or logistic growth curve) that captures the long-term direction. Prophet automatically detects *changepoints* — moments where the growth rate shifts. A hotel's booking trend might grow steadily for two years, plateau when the market saturates, then spike again when a nearby convention center opens. Prophet handles these shifts natively.

**Seasonality:** Captured using Fourier series — mathematical functions that can represent any repeating pattern. You don't need to specify m = 4 like SARIMAX. Prophet's yearly seasonality can capture quarterly patterns, monthly patterns, or complex shapes that don't fit neatly into a single cycle length.

**Holidays:** Modeled as one-time or recurring events that create predictable bumps or dips. Art Basel, Ultra Music Festival, spring break — if you tell Prophet about them, it will learn their effect.

The key hyperparameters you'll set when building a Prophet model are: `changepoint_prior_scale` (how sensitive the model is to trend changes — higher means more flexible), `seasonality_prior_scale` (how strong the seasonal effects are — higher means more flexible seasonality), `seasonality_mode` (additive or multiplicative — whether seasonal effects add a fixed amount or scale proportionally with the trend), and `changepoint_range` (what proportion of the training data can contain changepoints — default 0.8 means the last 20% is assumed to follow the same trend).

If SARIMAX is a classically trained chef who follows a precise recipe with exact measurements and specific techniques, Prophet is a talented home cook who eyeballs the ingredients, adjusts on the fly, and often produces excellent results with less effort. The classically trained chef is more precise when conditions are perfect — clean data, stable patterns, well-behaved seasonality. But the home cook adapts better when the kitchen is messy — missing data, trend changes, irregular holidays.

```mermaid
graph TD
    subgraph sarimax_approach ["SARIMAX Approach"]
        s_ar["Autoregression<br/>(past values)"]
        s_diff["Differencing<br/>(remove trend)"]
        s_ma["Moving Average<br/>(past errors)"]
        s_season["Seasonal Layer<br/>(P, D, Q, m)"]
    end
    subgraph prophet_approach ["Prophet Approach"]
        p_trend["Piecewise Linear<br/>Trend"]
        p_season["Fourier Series<br/>Seasonality"]
        p_holiday["Holiday<br/>Effects"]
    end
    sarimax_approach --> forecast1["SARIMAX<br/>Forecast"]
    prophet_approach --> forecast2["Prophet<br/>Forecast"]
    forecast1 --> compare["Compare<br/>RMSE, R²"]
    forecast2 --> compare
    compare --> choose["Choose Best<br/>for Context"]
```

**Figure 2.4: Two Approaches, One Goal** — SARIMAX models autocorrelation structure; Prophet fits additive components. Both produce forecasts that can be compared on the same metrics.

### Choosing Between Models

Here's the honest truth that someone always asks about: there is no universally better model. The right choice depends on your data and your situation.

SARIMAX tends to win when the data has stable, well-behaved seasonality and a smooth trend — exactly the kind of clean quarterly pattern you see in tourism data — as long as the model is told about the trend. Section 2.4 shows what happens when it isn't. It's also more interpretable: the (p, d, q) parameters tell you *how* the model is learning, and the coefficients are statistically testable.

Prophet tends to win when the data has multiple seasonalities at different frequencies (daily + weekly + yearly), when the trend changes abruptly at known points, when there are gaps or irregular observations, or when you need to get a reasonable forecast quickly without deep statistical expertise.

> **Scenario: Two Models, One Decision**
>
> A car rental company at Fort Lauderdale-Hollywood International Airport runs both SARIMAX and Prophet on their weekly fleet demand data. SARIMAX says Q3 demand will be 12% above Q2. Prophet says 18%. The fleet manager has to decide how many vehicles to pre-position at the airport lot — each unused car sitting idle costs $45/day in depreciation and lot fees, but each turned-away customer costs $280 in lost rental revenue plus the lifetime value risk of a customer who downloads a competitor's app.
>
> The two models agree on *direction* (Q3 is up) but disagree on *magnitude* (12% vs 18%). The fleet manager doesn't pick one number — she plans for the range. She pre-positions enough vehicles for the 12% scenario (the conservative estimate), keeps a reserve fleet at the regional depot that can be deployed within 48 hours if demand tracks closer to the 18% scenario, and sets a decision trigger: "If Week 2 of Q3 exceeds the 12% trajectory by more than 5%, activate the reserve fleet."
>
> *Connection:* Model comparison isn't about finding the "right" number. It's about understanding the range of plausible outcomes and making a risk-adjusted decision.

### Evaluating Forecast Quality: RMSE and R²

You'll evaluate your forecasts with two complementary metrics.

**RMSE (Root Mean Squared Error)** measures the average magnitude of prediction errors *in the original units of your data*. If your model forecasts quarterly visitor nights (in millions) and RMSE = 2.8, that means your predictions are off by about 2.8 million visitor nights per quarter on average. RMSE is what the business cares about — it answers "how wrong could this forecast be?" and lets the revenue manager build a buffer into the budget.

**R² (R-squared)** measures what proportion of the variance in the data the model explains, on a scale from 0 to 1 (or 0% to 100%). An R² of 0.87 means the model captures 87% of the patterns in the data — the remaining 13% is noise or unexplained variation. R² is what the analyst cares about — it answers "how much of what's happening does this model actually understand?" and enables fair comparison between models.

You need both. A model can have a decent R² but a huge RMSE if the data has large values. A model can have a small RMSE but a low R² if the data has very little variance. Together, they give you the complete picture.

📊 **By The Numbers:** In a 2023 benchmarking study of forecasting methods on quarterly economic time series, SARIMA-family models achieved median R² values of 0.82–0.91 on tourism data, while Prophet achieved 0.78–0.88. The gap narrowed considerably on messy, irregularly-spaced data where Prophet's flexibility became an advantage.

---

## 2.3 The Data

We'll work with the **Australian International Tourism Quarterly** dataset — the same data used in our source article, and a natural companion to the South Florida tourism context that frames this chapter. Australia and South Florida share strikingly similar tourism dynamics: strong seasonal patterns driven by weather and holidays, a long-term growth trend fueled by expanding international travel, and periodic disruptions (bushfires in Australia, hurricanes in Florida) that temporarily break the pattern.

The dataset contains 48 quarterly observations spanning 1999 through 2010 — twelve full years of tourism data.

```python
# Load the Australian Tourism Quarterly dataset
import pandas as pd

url = "https://raw.githubusercontent.com/c-marq/cap4767-data-mining/main/data/australia-international-visitor-nights-quarterly.csv"
df = pd.read_csv(url, parse_dates=["Quarter"], index_col="Quarter")
df = df.asfreq("QS")  # Enforce quarterly start frequency

print(f"Shape: {df.shape}")
print(f"Date range: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
print(f"\nFirst 8 rows:")
print(df.head(8))
```

```
# Expected Output:
# Shape: (48, 1)
# Date range: 1999-01 to 2010-10
#
# First 8 rows:
#             VisitorNights
# Quarter
# 1999-01-01        30.0525
# 1999-04-01        19.1485
# 1999-07-01        25.3177
# 1999-10-01        27.5914
# 2000-01-01        32.0765
# 2000-04-01        23.4880
# 2000-07-01        28.4759
# 2000-10-01        35.1238
```

Each row represents one quarter. The `VisitorNights` column contains the total number of nights, in millions, that international tourists spent in Australia during that quarter, as reported by Tourism Research Australia. Notice the pattern even in these first eight rows: Q2 (April) is the lowest quarter in both years, and the quarters on either side of the Australian summer, Q4 (October) and Q1 (January), run highest. Across all twelve years, Q1 is the peak quarter in eleven of them and Q2 is the low point in every one. That's the seasonal signature we'll be modeling.

The data is clean — no missing values, no outliers that need removal. The main preparation step is ensuring the datetime index has the correct frequency (`QS` for quarter start), which we've already done. This dataset is ready for modeling.

🤔 **Think About It:** Why would Q1 bring the most visitor nights to Australia and Q2 the fewest? Remember — Australia is in the Southern Hemisphere. Their Q1 (January–March) is summer, and Q2 (April–June) runs into autumn and early winter. Australia's summer falls during the Northern Hemisphere's winter, so a visitor from Europe, North America, or northern Asia can trade cold weather for a beach. That creates a seasonal pattern that *lines up with* South Florida's snowbird-driven Q1 peak rather than inverting it.

---

## 2.4 The Demo

### Setup

Before we build anything, let's get our environment ready. We need a few libraries beyond what Chapter 1 used — specifically `pmdarima` for automated ARIMA parameter search and `prophet` for Meta's forecasting model. Both require installation in Colab.

```python
# ============================================
# Chapter 2 Setup: Install and Import Libraries
# Run this cell first — installations may take 1-2 minutes
# ============================================

# Install forecasting libraries not included in default Colab
!pip install pmdarima prophet -q

# Core data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt

# Time series analysis (from Chapter 1)
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Forecasting — NEW in Chapter 2
from pmdarima import auto_arima                          # Automated ARIMA parameter search
from statsmodels.tsa.statespace.sarimax import SARIMAX   # Classical seasonal forecasting
from prophet import Prophet                               # Meta's modern forecasting model

# Model evaluation
from sklearn.metrics import mean_squared_error, r2_score

# Suppress verbose Prophet logging
import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore")

print("✅ All libraries loaded successfully")
```

Now let's load our data and confirm what Chapter 1 already told us about this series:

```python
# Load the Australian Tourism dataset
url = "https://raw.githubusercontent.com/c-marq/cap4767-data-mining/main/data/australia-international-visitor-nights-quarterly.csv"
df = pd.read_csv(url, parse_dates=["Quarter"], index_col="Quarter")
df = df.asfreq("QS")

# Quick stationarity check — confirming Chapter 1 findings
adf_result = adfuller(df["VisitorNights"], autolag="AIC")
print(f"ADF Test p-value: {adf_result[1]:.4f}")
if adf_result[1] < 0.05:
    print("→ Data is stationary")
else:
    print("→ Data is non-stationary (as expected — trend + seasonality present)")
```

```
# Expected Output:
# ADF Test p-value: 0.9984
# → Data is non-stationary (as expected — trend + seasonality present)
```

Good — the ADF test confirms what we already know from Chapter 1: this series has a trend and seasonality that make it non-stationary. The ARIMA differencing component (d) will handle this.

### Example 1: Finding the Right Parameters with auto_arima

The first step in building a SARIMAX model is choosing the (p, d, q)(P, D, Q, m) parameters. Rather than guessing, we'll let `auto_arima` search systematically.

```python
# ============================================
# Example 2.1: Automated Parameter Search
# Purpose: Find optimal SARIMAX parameters using AIC
# Prerequisites: pmdarima installed, df loaded
# ============================================

# Step 1: Run auto_arima with seasonal=True and m=4 (quarterly data)
# auto_arima tests many (p,d,q)(P,D,Q,m) combinations
# and selects the one with the lowest AIC score
stepwise_model = auto_arima(
    df["VisitorNights"],
    seasonal=True,     # We know seasonality exists from Chapter 1
    m=4,               # Quarterly data = 4 periods per year
    suppress_warnings=True,
    error_action="ignore",
    trace=False        # Set to True if you want to see every combination tested
)

# Step 2: Extract the selected parameters
order = stepwise_model.order            # (p, d, q)
seasonal_order = stepwise_model.seasonal_order  # (P, D, Q, m)

print(f"Best non-seasonal order (p, d, q): {order}")
print(f"Best seasonal order (P, D, Q, m): {seasonal_order}")
print(f"AIC: {stepwise_model.aic():.2f}")
print(f"\nInterpretation:")
print(f"  AR lags (p={order[0]}): model uses {order[0]} recent observation(s)")
print(f"  Differencing (d={order[1]}): data differenced {order[1]} time(s) for stationarity")
print(f"  MA terms (q={order[2]}): model corrects from {order[2]} recent error(s)")
print(f"  Seasonal AR (P={seasonal_order[0]}): learns from {seasonal_order[0]} past season(s)")
print(f"  Seasonal differencing (D={seasonal_order[1]}): seasonal pattern differenced {seasonal_order[1]} time(s)")
print(f"  Seasonal MA (Q={seasonal_order[2]}): corrects from {seasonal_order[2]} seasonal error(s)")
print(f"  Seasonal period (m={seasonal_order[3]}): cycle repeats every {seasonal_order[3]} quarters")
```

```
# Expected Output:
# Best non-seasonal order (p, d, q): (1, 0, 0)
# Best seasonal order (P, D, Q, m): (1, 1, 0, 4)
# AIC: 207.53
#
# Interpretation:
#   AR lags (p=1): model uses 1 recent observation(s)
#   Differencing (d=0): data differenced 0 time(s) for stationarity
#   MA terms (q=0): model corrects from 0 recent error(s)
#   Seasonal AR (P=1): learns from 1 past season(s)
#   Seasonal differencing (D=1): seasonal pattern differenced 1 time(s)
#   Seasonal MA (Q=0): corrects from 0 seasonal error(s)
#   Seasonal period (m=4): cycle repeats every 4 quarters
```

Read those results carefully. auto_arima chose a relatively simple model: one AR lag, no differencing needed at the non-seasonal level (because the seasonal differencing already handles trend), and no MA component. On the seasonal side, it uses one season of history and one round of seasonal differencing. The AIC penalized more complex alternatives — extra parameters didn't improve the fit enough to justify the added complexity.

⚠️ **Common Pitfall:** If you forget `seasonal=True`, auto_arima returns a plain ARIMA model with no seasonal components. The forecast will capture the trend but completely miss the seasonal ups and downs — producing a straight line where you expect waves. Always set `seasonal=True` when your Chapter 1 decomposition showed seasonality.

🔧 **Pro Tip:** Set `trace=True` to watch auto_arima test each parameter combination in real time. It's excellent for building intuition about how many combinations exist and how AIC changes across them.

**Try It Yourself:**
1. Change `m=4` to `m=12`. What parameters does auto_arima select? Why does the model change? (Hint: you're telling it to look for a 12-quarter cycle that doesn't exist in quarterly data.)
2. Set `trace=True` and run again. Count how many models auto_arima evaluates. How many have similar AIC values?
3. Try `seasonal=False`. Compare the AIC to the seasonal model. Which is lower (better)?

### Example 2: Building and Evaluating the SARIMAX Forecast

Now we'll use those parameters to fit a SARIMAX model, forecast, and evaluate. The critical step here is the train/test split — and for time series, you cannot randomly split. The training set must come *before* the test set in time. Randomly shuffling would leak future information into the model, making your accuracy metrics meaninglessly optimistic.

```python
# ============================================
# Example 2.2: SARIMAX Forecast with Temporal Train/Test Split
# Purpose: Fit SARIMAX, forecast the test period, evaluate with RMSE and R²
# Prerequisites: Example 2.1 completed (order and seasonal_order defined)
# ============================================

# Step 1: Temporal train/test split
# First 36 quarters (9 years) for training, last 12 quarters (3 years) for testing
train_size = 36
train_data = df["VisitorNights"].iloc[:train_size]
test_data = df["VisitorNights"].iloc[train_size:]

print(f"Training period: {train_data.index[0].to_period('Q').strftime('%Y-Q%q')} to {train_data.index[-1].to_period('Q').strftime('%Y-Q%q')}")
print(f"Testing period:  {test_data.index[0].to_period('Q').strftime('%Y-Q%q')} to {test_data.index[-1].to_period('Q').strftime('%Y-Q%q')}")
print(f"Train size: {len(train_data)} quarters | Test size: {len(test_data)} quarters")

# Step 2: Fit SARIMAX using auto_arima's selected parameters
sarimax_model = SARIMAX(
    train_data,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,   # Allow flexible stationarity constraints
    enforce_invertibility=False    # Allow flexible invertibility constraints
)
sarimax_result = sarimax_model.fit(disp=False)  # disp=False suppresses convergence output

# Step 3: Forecast the test period
sarimax_forecast = sarimax_result.forecast(steps=len(test_data))

# Step 4: Evaluate
sarimax_rmse = np.sqrt(mean_squared_error(test_data, sarimax_forecast))
sarimax_r2 = r2_score(test_data, sarimax_forecast)

print(f"\n--- SARIMAX Evaluation ---")
print(f"RMSE: {sarimax_rmse:.1f} million visitor nights")
print(f"R²:   {sarimax_r2:.4f}")
print(f"\nInterpretation:")
print(f"  The model's predictions are off by ~{sarimax_rmse:.0f} million visitor nights per quarter on average.")
print(f"  The model explains {sarimax_r2*100:.1f}% of the variation in visitor nights.")

# Step 5: Plot the results
plt.figure(figsize=(12, 5))
plt.plot(train_data.index, train_data, label="Training Data", color="steelblue")
plt.plot(test_data.index, test_data, label="Actual (Test)", color="darkorange")
plt.plot(test_data.index, sarimax_forecast, label="SARIMAX Forecast", 
         linestyle="--", color="green")
plt.title("SARIMAX Forecast vs Actual Visitor Nights")
plt.xlabel("Quarter")
plt.ylabel("Visitor nights (millions)")
plt.legend()
plt.tight_layout()
plt.show()
```

```
# Expected Output:
# Training period: 1999-Q1 to 2007-Q4
# Testing period:  2008-Q1 to 2010-Q4
# Train size: 36 quarters | Test size: 12 quarters
#
# --- SARIMAX Evaluation ---
# RMSE: 6.1 million visitor nights
# R²:   0.4405
#
# Interpretation:
#   The model's predictions are off by ~6 million visitor nights per quarter on average.
#   The model explains 44.1% of the variation in visitor nights.
#
# [Plot: blue training line, orange test line, green dashed forecast line
#  with the same seasonal pattern, running below the actual data]
```

Look at the forecast plot. The green dashed line follows the seasonal shape of the orange actual line, peaking in Q1 and bottoming out in Q2, but it runs below the actual line in all twelve test quarters, and the gap widens each year: about 4.5 million visitor nights in 2008 Q1, 7.5 million in 2009 Q1, and 12.3 million in 2010 Q1. The model gets the seasonal *pattern* right but misses the *growth*. An RMSE of ~6.1 million visitor nights on a series that ranges from roughly 19 to 60 million, with an R² of only 0.44, is a weak result. Keep that widening gap in mind; Example 3 shows where it comes from.

⚠️ **Common Pitfall:** Never use `train_test_split()` from scikit-learn on time series data. That function shuffles randomly, which destroys the temporal ordering. For time series, always split by time: everything before a cutoff date is training, everything after is testing.

**Try It Yourself:**
1. Change `train_size` to 40 (10 years training, 2 years testing). Does more training data improve RMSE?
2. Manually set `order=(1,1,1)` and `seasonal_order=(1,1,1,4)` instead of using auto_arima's choice. Is it better or worse? Check the RMSE.
3. Add `print(sarimax_result.summary())` after the fit to see the full statistical summary — coefficients, standard errors, and p-values for each parameter.

### Example 3 / Full Pipeline: SARIMAX vs Prophet Head-to-Head

Now we bring Prophet into the picture and answer the question that matters: which model should the hotel revenue manager trust?

```python
# ============================================
# Example 2.3: Full Pipeline — SARIMAX vs Prophet Comparison
# Purpose: Fit both models on the same data, compare fairly, forecast the future
# Prerequisites: Examples 2.1 and 2.2 completed
# ============================================

# --- PART A: Prophet Forecast on the Same Train/Test Split ---

# Step 1: Prepare data in Prophet's required format
# Prophet requires columns named 'ds' (datestamp) and 'y' (value)
train_prophet = pd.DataFrame({
    "ds": train_data.index,
    "y": train_data.values
})
test_prophet = pd.DataFrame({
    "ds": test_data.index,
    "y": test_data.values
})

# Step 2: Configure and fit Prophet
# These parameters control the model's flexibility
prophet_model = Prophet(
    changepoint_prior_scale=0.05,    # Moderate trend flexibility
    seasonality_prior_scale=10.0,    # Default — let seasonality be flexible
    holidays_prior_scale=0.01,       # Low — no major holiday effects in this data
    seasonality_mode="multiplicative",  # Seasonal effects scale with the trend
    changepoint_range=0.8,           # Changepoints allowed in first 80% of data
    yearly_seasonality=True,         # Yes — clear annual pattern
    weekly_seasonality=False,        # No — quarterly data has no weekly pattern
    daily_seasonality=False          # No — quarterly data has no daily pattern
)
prophet_model.fit(train_prophet)

# Step 3: Create future dataframe and predict
future = prophet_model.make_future_dataframe(periods=len(test_data), freq="QS")
prophet_predictions = prophet_model.predict(future)

# Extract only the test period predictions
prophet_forecast = prophet_predictions["yhat"].iloc[-len(test_data):].values

# Step 4: Evaluate Prophet
prophet_rmse = np.sqrt(mean_squared_error(test_data.values, prophet_forecast))
prophet_r2 = r2_score(test_data.values, prophet_forecast)

# --- PART B: Side-by-Side Comparison ---

# Step 5: Build comparison table
print("=" * 55)
print("    MODEL COMPARISON: SARIMAX vs Prophet")
print("=" * 55)
print(f"{'Metric':<25} {'SARIMAX':>12} {'Prophet':>12}")
print("-" * 55)
print(f"{'RMSE (million nights)':<25} {sarimax_rmse:>12.1f} {prophet_rmse:>12.1f}")
print(f"{'R²':<25} {sarimax_r2:>12.4f} {prophet_r2:>12.4f}")
print("-" * 55)

# Determine winner
if sarimax_rmse < prophet_rmse:
    print(f"\n→ SARIMAX wins on RMSE by {prophet_rmse - sarimax_rmse:.1f} million visitor nights")
else:
    print(f"\n→ Prophet wins on RMSE by {sarimax_rmse - prophet_rmse:.1f} million visitor nights")

if sarimax_r2 > prophet_r2:
    print(f"→ SARIMAX wins on R² by {(sarimax_r2 - prophet_r2)*100:.2f} percentage points")
else:
    print(f"→ Prophet wins on R² by {(prophet_r2 - sarimax_r2)*100:.2f} percentage points")

# Step 6: Combined forecast plot
plt.figure(figsize=(14, 6))
plt.plot(train_data.index, train_data, label="Training Data", color="steelblue", alpha=0.7)
plt.plot(test_data.index, test_data, label="Actual (Test)", color="darkorange", linewidth=2)
plt.plot(test_data.index, sarimax_forecast, label="SARIMAX Forecast", 
         linestyle="--", color="green", linewidth=2)
plt.plot(test_data.index, prophet_forecast, label="Prophet Forecast", 
         linestyle="--", color="red", linewidth=2)
plt.title("SARIMAX vs Prophet: Quarterly Tourism Forecast")
plt.xlabel("Quarter")
plt.ylabel("Visitor nights (millions)")
plt.legend()
plt.tight_layout()
plt.show()

# --- PART C: Future Forecast with SARIMAX ---

# Step 7: Retrain SARIMAX on ALL data and forecast 8 quarters ahead
full_data = df["VisitorNights"]
future_quarters = 8

# Retrain SARIMAX on full dataset
final_model = SARIMAX(
    full_data,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
final_result = final_model.fit(disp=False)
future_forecast = final_result.forecast(steps=future_quarters)

# Create future date index
future_index = pd.date_range(
    start=full_data.index[-1] + pd.tseries.offsets.QuarterBegin(),
    periods=future_quarters,
    freq="QS"
)

# Step 8: Plot the complete picture — historical + future
plt.figure(figsize=(14, 6))
plt.plot(full_data.index, full_data, label="Historical Data", color="steelblue")
plt.plot(future_index, future_forecast, label="Future Forecast (8 Quarters)", 
         linestyle="--", color="red", linewidth=2, marker="o", markersize=5)
plt.axvline(x=full_data.index[-1], color="gray", linestyle=":", alpha=0.7, label="Forecast Start")
plt.title("Tourism Forecast: 8 Quarters Into the Future")
plt.xlabel("Quarter")
plt.ylabel("Visitor nights (millions)")
plt.legend()
plt.tight_layout()
plt.show()

# Print future forecast values
print("\n--- Future Quarterly Forecast ---")
for date, value in zip(future_index, future_forecast):
    print(f"  {date.to_period('Q').strftime('%Y Q%q')}: {value:,.1f} million predicted visitor nights")
```

```
# Expected Output:
# =======================================================
#     MODEL COMPARISON: SARIMAX vs Prophet
# =======================================================
# Metric                         SARIMAX      Prophet
# -------------------------------------------------------
# RMSE (million nights)              6.1          2.3
# R²                              0.4405       0.9205
# -------------------------------------------------------
#
# → Prophet wins on RMSE by 3.8 million visitor nights
# → Prophet wins on R² by 48.00 percentage points
#
# [Plot 1: Combined forecast — blue training, orange actual test,
#  green dashed SARIMAX, red dashed Prophet, both tracking the seasonal pattern;
#  red stays close to the actual line, green runs below it]
#
# [Plot 2: Full historical data + 8-quarter red dashed future forecast
#  repeating the seasonal oscillation at about the 2010 level]
#
# --- Future Quarterly Forecast ---
#   2011 Q1: 59.8 million predicted visitor nights
#   2011 Q2: 35.5 million predicted visitor nights
#   2011 Q3: 44.6 million predicted visitor nights
#   2011 Q4: 46.7 million predicted visitor nights
#   2012 Q1: 60.1 million predicted visitor nights
#   2012 Q2: 35.5 million predicted visitor nights
#   2012 Q3: 44.9 million predicted visitor nights
#   2012 Q4: 47.0 million predicted visitor nights
```

Look at that comparison table. Prophet wins on both metrics here: an RMSE of 2.3 million visitor nights against SARIMAX's 6.1, and an R² of 0.92 against 0.44. If you expected SARIMAX to win on clean, stable quarterly data, the plot shows why it didn't. Visitor nights grew year after year from 2003 on, and Prophet's piecewise linear trend carries that growth into the test years. Our SARIMAX model handles the seasonal pattern through seasonal differencing, but the `SARIMAX()` call has no trend term, so its forecasts stay close to the last training year (2007) while the actual series keeps climbing. The model auto_arima fitted did include one, an intercept that acts as the growth term (print `stepwise_model.summary()` to see it), but we passed only its orders to `SARIMAX()`. Choosing a model isn't only about what the data looks like; it's also about whether the model, as you set it up, can represent what the data is doing.

But notice that both models get the seasonal *shape* right. In the combined plot, both green and red dashed lines rise and fall with the same quarterly rhythm as the actual data, high in Q1 and low in Q2. They disagree on level, not on the pattern. That's still an important finding: the seasonal pattern is real, both models confirm it, and the hotel revenue team can plan around it with confidence.

Part C projects 8 quarters ahead with SARIMAX retrained on all 48 quarters. The same retrain-on-everything step works for Prophet: fit it on the full dataset, then call `make_future_dataframe(periods=8, freq="QS")`.

⚠️ **Common Pitfall:** Prophet requires very specific column names — `ds` for dates and `y` for values. If you pass `Date` or `VisitorNights` instead, you'll get `ValueError: Dataframe must have columns "ds" and "y" with the dates and values respectively.` This catches everyone the first time.

> **Scenario: When the Pattern Breaks**
>
> A restaurant group in Brickell used SARIMAX to forecast weekend dinner covers and nailed it for 18 months — every Friday and Saturday within 8% of the prediction. Then a buzzy new competitor opened two blocks away, and suddenly the model's RMSE tripled. Forecasted covers: 180. Actual covers: 112.
>
> The model wasn't wrong about the *past*. The underlying pattern changed. The restaurant's analyst faced three options: retrain on only the most recent six months of data (losing the seasonal intelligence from prior years), add the competitor's opening date as an exogenous variable in SARIMAX (letting the model learn the disruption's effect), or switch to Prophet, whose built-in changepoint detection might adapt faster to the new reality.
>
> She chose option two — adding the competitor as an exogenous variable — and RMSE dropped back to 11% within two quarters as the model learned the new normal.
>
> *Connection:* Forecast evaluation isn't something you do once when you build the model. It's ongoing. A spike in RMSE isn't just bad luck — it's a signal that something in the real world has changed, and your model needs to adapt.

**Try It Yourself:**
1. Change Prophet's `seasonality_mode` from `"multiplicative"` to `"additive"`. How does it affect RMSE? (Hint: multiplicative means seasonal effects *scale* with the trend — if the trend doubles, the seasonal swings double too. Additive means they stay the same size regardless of the trend level.)
2. Set `changepoint_prior_scale=0.5` instead of 0.05. What happens to Prophet's forecast? Does it become more or less volatile?
3. Change `future_quarters` to 20 (5 years). Look at how the forecast uncertainty grows the further out you project. Is a 5-year quarterly forecast useful?

🌎 **Real-World Application:** Cruise lines operating out of PortMiami use exactly this kind of model comparison to forecast passenger volume. Royal Caribbean's revenue management team has publicly discussed using both ARIMA-family models and machine learning approaches, choosing the right tool based on the route, season, and booking horizon. A 6-month forecast for Caribbean routes uses different methods than a 2-year forecast for new itineraries in Alaska.

---

## 2.5 Interpreting the Results

Let's step back from the code and read our outputs the way a hotel revenue manager would.

**What the SARIMAX parameters tell us.** The auto_arima search selected (1,0,0)(1,1,0,4). In plain language: the model uses one recent quarter to predict the next (AR=1), doesn't need non-seasonal differencing (d=0), doesn't use error correction at the short-term level (MA=0), learns from one past year's same-quarter value (seasonal AR=1), applies one round of seasonal differencing to remove the year-over-year seasonal trend (seasonal D=1), and doesn't use seasonal error correction (seasonal Q=0). The model is relatively simple — which is a good sign. It means the seasonal pattern in this data is clean enough that a straightforward model captures it. What our SARIMAX forecast missed was the growth, not the seasonality.

**What the comparison table tells us.** Prophet predicted quarterly visitor nights within ~2.3 million on average, capturing 92% of the test period's variance. SARIMAX was much less precise at ~6.1 million and 44% variance explained. Both models agree on the seasonal pattern, with Q1 peaks and Q2 lows, but only Prophet carried the upward trend into the test years; SARIMAX under-forecast every one of the twelve test quarters. The ~3.8 million gap in RMSE is large: SARIMAX's typical error is more than two and a half times Prophet's. If you were making a one-time decision, the two models would agree on *which* quarters are busy but not on *how busy* they will be.

**What the future forecast tells us.** The 8-quarter projection keeps the familiar seasonal oscillation: Q1 runs about 24 million visitor nights above Q2 in both forecast years, with Q3 and Q4 in between. But it shows almost no growth. The model projects 2012 Q4 at roughly 47.0 million visitor nights, essentially the same as the last observed Q4 in 2010, and 2012 Q1 at 60.1 million, less than 1% above 2010 Q1. That fails the sniff test: annual visitor nights rose every year from 2003 to 2010. It's the same missing growth term that cost SARIMAX the comparison, and a reminder to check any forecast against recent history before you hand it to a budget committee.

**Connecting back to the hotel.** For our Miami Beach hotel revenue manager, these results translate directly to action. The seasonal pattern is real — confirmed by both models — so staffing and inventory should follow the same quarterly rhythm. The upward trend means year-over-year growth in demand, justifying the lobby renovation investment, as long as the forecast you budget with actually carries that growth forward; Prophet's did and our SARIMAX's did not. And the difference between the two models' forecasts provides a natural *range* for budgeting: plan for the conservative estimate, budget a contingency for the optimistic one.

💡 **Key Insight:** The most valuable output of a forecasting exercise isn't a single number — it's the *range* between different credible models. When SARIMAX says 46.7 million visitor nights and Prophet says 51.5 million for the same quarter (2008 Q1, the first quarter of our test period), the revenue team doesn't pick one. They plan for something in that range and prepare contingencies for both ends.

---

## 2.6 Business Takeaways

Before this chapter, the revenue team had a time series and a gut feeling about seasonal patterns. Now they have specific quarterly predictions they can build a budget around, with quantified uncertainty (RMSE) that tells them exactly how much buffer to carry.

Here's what the business can *do* now:

**Set quarterly staffing levels.** The forecast shows Q1 running 68–69% above Q2. Housekeeping contracts, front desk scheduling, and restaurant staffing can follow that pattern instead of the flat annual average that wastes money in slow quarters and burns out staff in busy ones.

**Time capital expenditures.** The model identifies the lowest-demand quarter in the forecast horizon. That's when the 15-room renovation should happen — minimizing revenue loss during the construction period.

**Allocate marketing spend dynamically.** If Q2 is forecast to be the weakest quarter, that's where incremental marketing spend has the most room to move the needle. Spending heavily in Q1, when guests are already coming, is less efficient.

**Build an honest budget range.** Instead of one number, the revenue team can present a "SARIMAX scenario" and a "Prophet scenario" — a conservative and a moderate projection — and build operational plans for both. This is more honest and more useful than false precision.

But this raises a new question. Everything we've predicted so far has been about *the same variable* — future tourism from past tourism, future visitor nights from past visitor nights. We're looking at one column and projecting it forward. What if the hotel wants to predict something *different* — not how many guests will arrive, but which specific guests are likely to cancel their reservation? Or what price a competitor's room should sell for given its amenities? That requires predicting one variable *from other variables* — and it's the domain of **supervised learning**, which begins in the next chapter with the most foundational predictive model in all of statistics: **regression**.

---

## 2.7 Chapter Lab

### Lab 1: Time Series Exploration & Forecasting

Lab 1 applies the full Chapter 1 and Chapter 2 pipeline to the **Florida Hotel Occupancy** dataset (quarterly, 2005–2024). Choose **one numeric column** as your forecasting target, any column except `occupancy_rate_pct`, and take it from exploration through a SARIMAX vs Prophet comparison and a future forecast. The data is quarterly, so use m = 4. The lab is worth **20 points**.

- Starter notebook: [labs/lab01_forecasting.ipynb](https://github.com/c-marq/cap4767-data-mining/blob/main/labs/lab01_forecasting.ipynb)
- Instructions and point breakdown: [labs/lab01-forecasting.md](https://github.com/c-marq/cap4767-data-mining/blob/main/labs/lab01-forecasting.md)

### Extra Practice (Optional): Monthly U.S. Electricity Production

For more practice at a different frequency, run the same pipeline on the Federal Reserve's monthly industrial production index for U.S. electric and gas utilities (FRED series IPG2211A2N, index 2017 = 100, not seasonally adjusted). The file has 408 monthly observations from January 1985 through December 2018, in two columns, `Month` and `Production`. It has a winter peak, a second peak in summer, and a long-term growth trend that levels off after the mid-2000s. Use m = 12 for monthly seasonality. Load it the same way as the tourism data, with `parse_dates=["Month"], index_col="Month"` and `.asfreq("MS")`.

- URL: `https://raw.githubusercontent.com/c-marq/cap4767-data-mining/main/data/us-electricity-production-monthly.csv`

---

## 2.8 Multiple Choice Questions

**1.** What does the "I" in ARIMA stand for, and what does it do?

a) Interpolated — fills in missing values in the time series
b) Integrated — applies differencing to make the series stationary ✓
c) Iterative — repeats the model training process multiple times
d) Inverse — reverses the order of the time series before modeling

**2.** In SARIMA notation (p, d, q)(P, D, Q, m), what does the parameter m represent?

a) The number of months in the dataset
b) The maximum number of parameters the model can use
c) The length of the seasonal cycle in periods ✓
d) The number of models auto_arima will evaluate

**3.** For quarterly tourism data, what value of m should you use in a SARIMA model?

a) m = 1
b) m = 4 ✓
c) m = 12
d) m = 52

**4.** What metric does auto_arima use to balance model fit quality against model complexity?

a) RMSE — Root Mean Squared Error
b) R² — Coefficient of Determination
c) AIC — Akaike Information Criterion ✓
d) MSE — Mean Squared Error

**5.** Why can't you use scikit-learn's `train_test_split()` function on time series data?

a) It doesn't support pandas DataFrames
b) It requires at least 1,000 data points
c) It shuffles randomly, breaking temporal dependencies and causing data leakage ✓
d) It only works for classification tasks, not regression

**6.** A SARIMAX model produces RMSE = 300 trips on quarterly data that ranges from 3,000 to 6,000 trips. What does this mean in practical terms?

a) The model is 300% accurate
b) The model's predictions are off by about 300 trips per quarter on average ✓
c) The model will be exactly 300 trips wrong every quarter
d) The model needs at least 300 data points to work

**7.** Prophet requires input data in a specific format. What must the two required columns be named?

a) `date` and `value`
b) `time` and `y`
c) `ds` and `y` ✓
d) `timestamp` and `target`

**8.** What is the primary advantage of Prophet over SARIMAX?

a) Prophet always produces more accurate forecasts
b) Prophet handles messy data, multiple seasonalities, and trend changes more flexibly ✓
c) Prophet doesn't require any hyperparameter settings
d) Prophet can forecast categorical variables

**9.** In Prophet, what does the `seasonality_mode="multiplicative"` setting mean?

a) Multiple seasonal patterns are modeled simultaneously
b) The seasonal effect is multiplied by itself each year
c) Seasonal effects scale proportionally with the trend level ✓
d) The model multiplies SARIMAX and Prophet forecasts together

**10.** An analyst runs both SARIMAX and Prophet on the same dataset. SARIMAX has RMSE = 248, R² = 0.87. Prophet has RMSE = 313, R² = 0.81. Which of the following is the most appropriate conclusion?

a) SARIMAX is always the better model
b) Prophet is broken and should not be used
c) SARIMAX performs better on this dataset, but Prophet may be preferred in other contexts ✓
d) The models are essentially identical in performance

---

## 2.9 Reflection Questions

**1.** Think about a dataset from your own job, hobby, or daily life that has a time-based pattern — it could be sales, attendance, website traffic, workout frequency, or anything else. What would Recency, Frequency, and the seasonal pattern look like in that data? Which forecasting approach (SARIMAX or Prophet) do you think would be more appropriate, and why?

**2.** Prophet beat our SARIMAX model on the Australian tourism dataset. But imagine the tourism data included a sudden disruption — like a global pandemic in 2020 that dropped visitor nights to near zero for two quarters before a gradual recovery. How would each model handle this disruption? Which model's approach to trend changes (SARIMAX's differencing vs Prophet's changepoint detection) would adapt faster?

**3.** Consider a scenario where a forecast is used to make a high-stakes business decision — like the hotel renovation timing or the car rental fleet positioning from the Scenario Boxes. When a model produces a point forecast (a single number), how should a decision-maker think about the uncertainty around that number? Is a forecast that's sometimes wrong still useful? When would you *not* trust a forecast, even if the RMSE and R² look good?

---

## Key Terms

| Term | Definition |
|------|-----------|
| **ARIMA** | AutoRegressive Integrated Moving Average — a time series model combining autoregression (past values), differencing (trend removal), and moving average (past error correction) |
| **SARIMA** | Seasonal ARIMA — extends ARIMA with seasonal parameters (P, D, Q, m) that capture repeating patterns at fixed intervals |
| **SARIMAX** | Seasonal ARIMA with eXogenous variables — SARIMA plus the ability to include external variables that influence the forecast |
| **Prophet** | Meta's additive forecasting model that decomposes time series into trend (piecewise linear), seasonality (Fourier series), and holiday effects |
| **Autoregression (AR)** | Modeling the current value as a function of a fixed number of previous values (lags) |
| **Differencing** | Subtracting each value from the previous one to remove trends and achieve stationarity |
| **Moving Average (MA)** | Modeling the current value as a function of past forecast errors (residuals) |
| **Seasonal Order (P, D, Q, m)** | The seasonal counterparts to ARIMA's (p, d, q), operating across seasonal cycles of length m |
| **AIC (Akaike Information Criterion)** | A metric that balances model fit quality against complexity — lower is better; penalizes unnecessary parameters |
| **auto_arima** | A function from the pmdarima library that automatically searches for optimal ARIMA/SARIMA parameters using AIC |
| **Changepoint** | A point in time where the trend's growth rate shifts — Prophet detects these automatically |
| **Fourier Series** | Mathematical functions used by Prophet to model repeating seasonal patterns of any shape |
| **Exogenous Variable** | An external variable (not part of the time series itself) that influences the forecast — e.g., holidays, events, weather |
| **Forecast Horizon** | The number of future periods the model is asked to predict — accuracy decreases as the horizon lengthens |
| **Temporal Train/Test Split** | Splitting time series data by time (all training data before all test data) to preserve temporal ordering and prevent data leakage |
| **Stationarity** | A time series property where the statistical properties (mean, variance) remain constant over time — required by most ARIMA-family models (review: Chapter 1) |

---

*Next Chapter: Chapter 3 — Regression: From Linear Relationships to Logistic Classification*

In Chapter 3, we cross into **Part II: Supervised Learning**. So far, we've predicted future values of the *same* variable — tourism from past tourism, visitor nights from past visitor nights. That's forecasting: extending a single time series into the future. But what if we want to predict something *different entirely* — like what a house should sell for based on its square footage, bedrooms, and neighborhood? Or whether a customer will churn based on their contract type and monthly charges? That requires a fundamentally different approach: using *multiple input variables* to predict a *target outcome*. It starts with the simplest, most interpretable predictive model in statistics — and the one every more complex model is compared against.
