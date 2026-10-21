# Dataset Directory — CAP4767 Data Mining

Every dataset the readings and notebooks use is stored in this folder, so the course does not depend on outside links. Notebooks load these files from `https://raw.githubusercontent.com/c-marq/cap4767-data-mining/main/data/<file>`. Do not modify the setup cells.

The copies here are for non-commercial teaching. Each dataset keeps its original license, listed below; where no license is stated, rights stay with the original owner. Please cite the source when you reuse a dataset.

## Datasets

| File | Used in | What it is | Source | License |
|---|---|---|---|---|
| `airline-passengers.csv` | Chapter 1 | Monthly international airline passengers (thousands), 1949–1960, 144 rows | Box & Jenkins (1976), via the Time Series Data Library; copy from [jbrownlee/Datasets](https://github.com/jbrownlee/Datasets) | Not stated |
| `australia-international-visitor-nights-quarterly.csv` | Chapters 1–2 | Visitor nights (millions) spent by international tourists in Australia, 1999 Q1–2010 Q4, 48 rows | `austourists` in R package [fpp2](https://cran.r-project.org/package=fpp2) (Hyndman); Tourism Research Australia | GPL-3 (package) |
| `us-electricity-production-monthly.csv` | Chapter 2 (extra practice) | Industrial production index, electric and gas utilities (2017 = 100), Jan 1985–Dec 2018, 408 rows; downloaded 2026-10-20 | Federal Reserve Board, [FRED IPG2211A2N](https://fred.stlouisfed.org/series/IPG2211A2N) | Public domain, citation requested |
| `AAPL.csv` | Module 1 demo | Apple daily stock prices, 2020, 253 rows | Yahoo Finance historical download | Not stated (Yahoo terms restrict redistribution) |
| `stocks.csv` | Module 1 group exercise | Same Apple 2020 prices as `AAPL.csv`, fewer columns | Yahoo Finance | Not stated |
| `acresBurned.csv` | Module 1 demo | Daily acres burned by U.S. wildfires, 1992–2015 (daily total of the fire records) | USDA Forest Service, [FPA FOD](https://doi.org/10.2737/RDS-2013-0009.4) (Short 2017) | U.S. government data, usable without permission; citation requested |
| `tourism.csv` | Module 2 demo | Australian domestic overnight trips by region, state and purpose, 1998 Q1–2016 Q4 | `tourism` in R package [tsibble](https://tsibble.tidyverts.org/reference/tourism.html); Tourism Research Australia | GPL-3 (package) |
| `rsafsna_retail_sales.csv` | Module 2 demo (retail) | U.S. advance retail sales, monthly, not seasonally adjusted | U.S. Census Bureau, [FRED RSAFSNA](https://fred.stlouisfed.org/series/RSAFSNA) | Public domain, citation requested |
| `florida_hotel_occupancy.csv` | Module 2 group exercise, Lab 1 | Quarterly Florida hotel and visitor measures, 2005–2024, 80 rows | Not documented (see note) | — |
| `housingData.csv` | Chapter 3, Module 3 demo and group exercise | King County, WA home sales, May–July 2014, 4,600 rows | Kaggle [shree1992/housedata](https://www.kaggle.com/datasets/shree1992/housedata) | Not stated |
| `cars.csv` | Chapter 3, Module 3 demo | Car prices and specifications, 205 rows | Kaggle [hellbuoy/car-price-prediction](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction); based on UCI Automobile (Schlimmer 1985) | Not stated (UCI original: CC BY 4.0) |
| `graduate-admissions.csv` | Chapter 3, Module 3 demo | Graduate admissions (admit, gre, gpa, rank), 400 rows, hypothetical | [UCLA OARC](https://stats.oarc.ucla.edu/r/dae/logit-regression/) logistic regression example | Not stated |
| `insurance.csv` | Module 3 insurance demo, Lab 2 | Medical insurance charges, 1,338 rows | Kaggle [mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance) (from Lantz, *Machine Learning with R*) | Open Database License |
| `california-housing.csv` | Chapter 3, Lab 2 | California census block groups, 1990, 20,640 rows | StatLib, Pace & Barry (1997); as distributed by scikit-learn | Not stated |
| `auto-mpg.csv` | Chapter 3, Lab 2 | Car fuel economy, 398 rows | [UCI Auto MPG](https://doi.org/10.24432/C5859H) (Quinlan 1993); as distributed by seaborn | CC BY 4.0 |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Chapters 4–5, Modules 4–5 | IBM sample telecom customers, 7,043 rows (7,032 after dropping blank `TotalCharges`) | IBM sample data via Kaggle [blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | Not stated |
| `Churn_Modelling.csv` | Lab 3 | Bank customers, 10,000 rows, target `Exited` | Kaggle [shrutimechlearn/churn-modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) | Not stated |
| `BankChurners.csv` | Lab 3 | Credit card customers, 10,127 rows, target `Attrition_Flag` | Kaggle [sakshigoyal7/credit-card-customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) | CC0 |
| `online-retail.xlsx` | Chapters 6–8, Modules 6–8 | UK online retailer invoice lines, Dec 2010–Dec 2011, 541,909 rows | [UCI Online Retail](https://doi.org/10.24432/C5BW33) (Chen 2015); copy from [fenago/datasets](https://github.com/fenago/datasets) | CC BY 4.0 |
| `olist_*.csv` (5 files) | Modules 6–7 group exercise, Lab 4 | Brazilian e-commerce orders, customers, items, payments, products | Kaggle [olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | CC BY-NC-SA 4.0 (non-commercial) |
| `restaurant-1-orders.csv` | Modules 6–7 group exercise, Lab 4 | Takeaway restaurant order lines, 2015–2019, 74,818 rows | Kaggle [henslersoftware/19560-indian-takeaway-orders](https://www.kaggle.com/datasets/henslersoftware/19560-indian-takeaway-orders) | Not stated |
| `gym_members_exercise_tracking.csv` | Chapter 9, Module 9 group exercise, Lab 5 | Gym members' workout measures, 973 rows | Kaggle [valakhorasani/gym-members-exercise-dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset) | Apache 2.0 |
| `Blood Transfusion Service data.csv` | Chapter 9, Module 9 group exercise, Lab 5 | Blood donors' recency, frequency and monetary measures, 748 rows | [UCI Blood Transfusion Service Center](https://doi.org/10.24432/C5GS39) (Yeh 2008), reformatted copy | CC BY 4.0 |
| `ufo-sightings.csv` | Chapter 9, Module 9 demo and group exercise | NUFORC UFO sighting reports, geocoded, 80,332 rows, no header row | NUFORC reports processed by S. Axel, [planetsig/ufo-reports](https://github.com/planetsig/ufo-reports) | Not stated |
| `mortality_data.csv` | Not used by current materials | U.S. childhood death rates by age group, 1900–2018 | CDC/NCHS, [Childhood Mortality Rates](https://catalog.data.gov/dataset/nchs-childhood-mortality-rates) | Public domain (U.S. government) |

**Not stored here:** the Instacart Online Grocery dataset (Lab 4, Option A) is too large for this repository; students download it from [Kaggle](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset).

## Notes

- **Florida hotel occupancy:** source unknown; it appears to be simulated teaching data, and the exercise and lab say so. Its structure suggests generated data: `revpar_usd` equals occupancy × average daily rate in every row, and `domestic_visitors_k` repeats a ceiling value of 42,000.
- **Online Retail vs. Online Retail II:** the chapters use the original UCI Online Retail dataset (2010–2011). The midterm prompt links Online Retail II on Kaggle, a larger version (2009–2011) with different column names (`Invoice`, `Price`, `Customer ID`).
- **FRED series** are revised over time, so a fresh download can differ slightly from the copies here.
