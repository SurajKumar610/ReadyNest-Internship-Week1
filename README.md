# Dataset Analysis & Reporting Platform (DARP)

DARP is a complete, modular, and professional dataset analysis and reporting platform built with Python and Streamlit. It automates loading, cleaning, analyzing, visualizing, forecasting, spatial plotting, statistical comparison, and document reporting workflows.

## 🚀 Features

1. **Flexible Data Loading (Module 1)**: Supports CSV, Excel, JSON file uploads, REST API integrations, and database extraction via SQLAlchemy URI connections.
2. **Robust Data Cleansing (Module 2)**: Missing value imputations (Mean/Median/Mode), duplicate removal, IQR/Z-score outlier capping or removal, and text-trimming standardization.
3. **Descriptive Stats Summary (Module 3)**: Automatic mean, median, mode, variance, range, skewness, kurtosis, and missing percentage metrics for all numerical attributes.
4. **Interactive Univariate Analysis (Module 4)**: Histograms, Kernel Density Estimate (KDE) violin plots, bar charts, and donut charts built using Plotly.
5. **Interactive Bivariate Relationships (Module 5 & 6)**: Numerical vs. Numerical (scatter plots with OLS trendlines and Pearson/Spearman correlation metrics), Numerical vs. Categorical (box/violin plots), and Categorical vs. Categorical (stacked bar charts with Chi-Square tests of independence).
6. **Key Insights Discovery Engine (Module 7)**: Automated business findings, recommendations, and actionable items extracted from statistical properties and business columns (revenue, category, date, customer age, churn, etc.).
7. **Feature Engineering (Module 9)**: Date part extraction, numerical scaling (Standard/MinMax), label/one-hot encoding, and custom ratio metrics.
8. **Time Series Forecasting (Module 10)**: Exponential Smoothing, Moving Average, and ARIMA models with evaluation metrics (RMSE, MAE, MAPE) and forecast trend lines.
9. **Interactive Geographic Maps (Module 11)**: Scatter Maps (Plotly Express Mapbox), Marker Clusters, and Heatmaps (Folium).
10. **Machine Learning Predictive Modeling (Module 13)**: Regressions (Linear, Ridge, Decision Tree, Random Forest) and Classifications (Logistic, Decision Tree, Random Forest) with scorecard metrics (R², MAE, RMSE; Accuracy, Precision, Recall, F1) and driver weights (feature importances).
11. **A/B Testing significance (Module 14)**: Independent Two-Sample T-tests (for numeric indicators) and Chi-Square tests (for binary conversion indicators) showing lift and statistical significance p-value analysis.
12. **Automated Export Reports Pack (Module 15)**: One-click export download of PDF executive briefings (ReportLab), multi-worksheet Excel files (OpenPyXL), and PowerPoint slides (python-pptx).

---

## 🛠️ Project Directory Structure

```text
Dataset-Analysis-Reporting/
│
├── data/
│   ├── raw/
│   └── cleaned/
│
├── notebooks/                 # Jupyer notebook sandboxes
│
├── src/                       # Core python analytics modules
│   ├── data_loading.py        # Import CSV, Excel, JSON, SQL, APIs
│   ├── data_cleaning.py       # Imputers, outliers, type cast, text trim
│   ├── descriptive_statistics.py # Calculations of moments & dispersion
│   ├── univariate_analysis.py # Frequency tables & single variable charts
│   ├── bivariate_analysis.py  # Co-relations, Chi-Square, crosstabs
│   ├── visualization.py       # Advanced charts (radar, sankey, waterfall, treemap)
│   ├── insight_extraction.py  # Statistics & business rules insight engine
│   ├── feature_engineering.py # Encoders, scalers, date parts, custom ratios
│   ├── forecasting.py         # time-series ARIMA/Exponential Smoothing
│   ├── geo_mapping.py         # coordinates plotting & Folium maps
│   ├── predictive_modeling.py # ML training, predictions & validations
│   ├── ab_testing.py          # control/variant hypothesis checks
│   └── reporting.py           # Document export (PDF, Excel, PPT)
│
├── dashboard/
│   └── streamlit_app.py       # Streamlit UI Dashboard code
│
├── reports/                   # Output storage for reports
│   ├── pdf/
│   ├── excel/
│   └── ppt/
│
├── models/                    # Serialized machine learning models
│
├── requirements.txt           # Package specifications
├── README.md                  # Project overview and run guide
└── main.py                    # Orchestrator CLI entry point
```

---

## ⚙️ Installation and Setup

1. **Create Virtual Environment**:
   It is recommended to run this in a virtual environment. You can initialize one using `uv` or `venv`:
   ```bash
   uv venv
   ```
   *or*
   ```bash
   python -m venv .venv
   ```

2. **Activate the Virtual Environment**:
   - **Windows**:
     ```powershell
     .venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```
   *or*
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Platform

You can run this platform in two modes:

### 1. Interactive Web Dashboard (Recommended)
This launches a beautiful, state-of-the-art interactive dashboard directly in your browser.
```bash
python main.py --dashboard
```
*or*
```bash
streamlit run dashboard/streamlit_app.py
```
This loads a comprehensive dummy dataset automatically if you do not upload a file, allowing you to test the entire dashboard's functionality immediately.

### 2. Command-Line Orchestrator (CLI Mode)
Clean, calculate stats, extract insights, and generate all output reports directly from the terminal for any dataset:
```bash
python main.py --input "data/raw/sales.csv" --output-dir "reports" --forecast-col "Sales" --time-col "Date"
```
Check the `reports/` folder to view the generated PDF, Excel, and PPT files.
