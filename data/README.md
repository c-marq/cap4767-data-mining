# Dataset Directory — CAP4767 Data Mining

All datasets used in this course are listed below with their source, description, and the chapter they appear in.

| Dataset | Source | Used In | Description |
|---------|--------|---------|-------------|
| Australian Tourism | Kaggle | Chapter 2 | Quarterly holiday trips — trend + seasonality |
| Australian International Visitor Nights (`australia-international-visitor-nights-quarterly.csv`) | R package fpp/fpp2 (`austourists`); Tourism Research Australia | Chapter 2 | 48 quarters, 1999 Q1–2010 Q4: nights (millions) international tourists spent in Australia — trend + seasonality |
| U.S. Electricity Production (`us-electricity-production-monthly.csv`) | FRED (IPG2211A2N) | Chapter 2 (optional extra practice) | 408 months, Jan 1985–Dec 2018: industrial production index for electric and gas utilities (2017 = 100, not seasonally adjusted) |
| IBM Telco Customer Churn | Kaggle | Chapters 4–5 | 7,032 customers, binary churn target |
| Online Retail II | Kaggle / UCI | Chapters 6–8 | 500K+ transactions, UK retailer |
| UFO Reports | GitHub (planetsig) | Chapter 9 | Historical sightings — data quality lesson |

> Datasets too large for GitHub are loaded directly from their source URLs inside each notebook setup cell.
> Do not modify the setup cells.

## Sources

- **australia-international-visitor-nights-quarterly.csv**: `austourists` from the R packages fpp/fpp2 (Rob J. Hyndman), GPL-3; original data from Tourism Research Australia. The file keeps the first 48 quarters (1999 Q1–2010 Q4) of the series, dated by quarter start.
- **us-electricity-production-monthly.csv**: FRED series IPG2211A2N (Industrial Production: Utilities: Electric and Gas Utilities, NAICS 2211,2), Board of Governors of the Federal Reserve System, retrieved from FRED, Federal Reserve Bank of St. Louis. Downloaded 2026-10-20; FRED revises this series, so later downloads may differ slightly.

Licenses for both datasets are to be confirmed in the dataset review.
