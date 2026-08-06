# Retail Revenue Concentration: RFM Segmentation Meets Cohort Retention

📄 **[Read the full report (PDF)](outputs/docs/Customer_Segmentation_Retention_Report.pdf)** — includes strategic recommendations and business impact estimates

## Overview
End-to-end customer analytics pipeline on the Online Retail II dataset, covering data ingestion, cleaning, revenue analytics, RFM segmentation, cohort retention, and a cross-analysis connecting customer segments to purchase behavior over time.

## Dataset
- **Source:** [Online Retail II, UCI ML Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Size:** 1,067,371 transactions (Dec 2009 – Dec 2011), 805,620 after cleaning
- **Fields:** Invoice, StockCode, Description, Quantity, InvoiceDate, UnitPrice, Customer ID, Country

## Workflow
The pipeline runs as a single script (`customer_segmentation_analysis.py`), structured in six sections:
1. **Data Loading** — UCI API with a direct HTTP download fallback
2. **Data Cleaning** — type fixes, removing cancellations and missing Customer IDs
3. **Revenue Analytics** — summary stats, revenue by country, monthly trend
4. **RFM Segmentation** — Recency / Frequency / Monetary scoring into 6 segments
5. **Cohort Retention Analysis** — monthly retention heatmap by first-purchase cohort
6. **RFM × Cohort Cross-Analysis** — tenure and order frequency by segment

## Key Findings
- **Total Revenue:** £17,743,429 across 36,975 invoices, 5,881 customers, 41 countries (Dec 2009 – Dec 2011)
- **UK dominance:** £14.7M (83% of revenue); top international market is the Netherlands
- **Champions segment:** 25.2% of customers (1,482) drive 69.3% of revenue (£12.3M)
- **Pareto effect:** top 20% of customers by monetary value generate 77.3% of total revenue
- **Month-1 retention crisis:** 78.8% of first-time buyers churn within 30 days — the single largest leakage point
- **Cohort retention:** stabilizes around 18-22% after month 1, gradually declining to 15-18% by month 24
- **At Risk insight:** longest average tenure (20.0 months) of any segment, but only 4.9 orders — a recoverable, zero-acquisition-cost win-back opportunity
- **Frequency vs. tenure:** Champions and At Risk have nearly identical tenure (18.7 vs. 20.0 months); order frequency (15.7 vs. 4.9) is the real differentiator — this is a habit-formation gap, not a new-customer problem

## Tech Stack
- Python — pandas, numpy, matplotlib, seaborn, openpyxl, requests
- Built on [Zerve](https://www.zerve.ai) — canvas-based, reproducible notebook environment

## Repository Structure
```
├── customer_segmentation_analysis.py   # full pipeline, single script
├── outputs/
│   ├── figures/                        # generated charts
│   │   ├── revenue_by_country.png
│   │   ├── monthly_revenue_trend.png
│   │   ├── rfm_segment_revenue.png
│   │   ├── cohort_retention_heatmap.png
│   │   └── tenure_frequency_by_segment.png
│   └── docs/
│       └── Customer_Segmentation_Retention_Report.pdf
└── README.md
```

## Key Visuals
![Revenue by Country](outputs/figures/revenue_by_country.png)
![Monthly Revenue Trend](outputs/figures/monthly_revenue_trend.png)
![Total Revenue by RFM Segment](outputs/figures/rfm_segment_revenue.png)
![Cohort Retention Heatmap](outputs/figures/cohort_retention_heatmap.png)
![Tenure vs Frequency by Segment](outputs/figures/tenure_frequency_by_segment.png)

## How to Run
```bash
pip install -r requirements.txt
python customer_segmentation_analysis.py
```

## Author
Suman Kumar Oraon
