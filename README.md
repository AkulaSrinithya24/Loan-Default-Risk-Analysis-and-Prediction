# Loan Default Risk Analysis and Prediction
### IBM SkillsBuild Data Analytics Academic Internship Project

## 🚀 Live Demo

**Streamlit Application:**  
https://loan-default-risk-analysis-and-prediction-ayrecyonj3zrdgrij3w4.streamlit.app/prediction

**GitHub Repository:**  
https://github.com/AkulaSrinithya24/Loan-Default-Risk-Analysis-and-Prediction

---

## Project Overview

This project delivers a complete, end-to-end data analytics and machine learning pipeline applied to a real-world loan default dataset. The objective is to understand the key drivers of loan default, quantify risk across borrower segments, surface actionable business insights, and build a predictive model capable of flagging high-risk applications before disbursement.

The work is structured across eight phases — from raw data loading through to an interactive Streamlit dashboard and a formal project report — following professional data analytics best practices.

---

## Dataset

| Property | Value |
|---|---|
| File | `data/Loan_default.csv` |
| Rows | 255,347 |
| Columns | 18 |
| Target column | `Default` (0 = No Default, 1 = Default) |
| Overall default rate | 11.61% |
| Missing values | None |
| Duplicate rows | None |

### Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| LoanID | String | Unique loan identifier (dropped before modelling) |
| Age | Numeric | Borrower age (18–69) |
| Income | Numeric | Annual income ($) |
| LoanAmount | Numeric | Loan size ($) |
| CreditScore | Numeric | Credit score (300–849) |
| MonthsEmployed | Numeric | Months at current employer |
| NumCreditLines | Numeric | Number of open credit lines |
| InterestRate | Numeric | Loan interest rate (%) |
| LoanTerm | Numeric | Loan term in months |
| DTIRatio | Numeric | Debt-to-income ratio |
| Education | Categorical | High School / Bachelor's / Master's / PhD |
| EmploymentType | Categorical | Full-time / Part-time / Self-employed / Unemployed |
| MaritalStatus | Categorical | Single / Married / Divorced |
| LoanPurpose | Categorical | Auto / Business / Education / Home / Other |
| HasMortgage | Binary | Whether borrower has an existing mortgage |
| HasDependents | Binary | Whether borrower has dependents |
| HasCoSigner | Binary | Whether the loan has a co-signer |

---

## Project Structure

```
Loan_Default_Risk_Analytics/
│
├── data/
│   └── Loan_default.csv              ← Raw dataset (255,347 rows)
│
├── notebooks/                        ← Step-by-step analytical notebooks
│   ├── 01_data_loading_understanding.ipynb
│   ├── 02_data_quality_cleaning.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_kpi_risk_analysis.ipynb
│   ├── 05_feature_engineering.ipynb
│   └── 06_model_training_evaluation.ipynb
│
├── src/                              ← Reusable Python backend modules
│   ├── config.py                     ← All paths, constants, hyper-parameters
│   ├── data_loader.py                ← Load & validate raw data
│   ├── data_cleaner.py               ← Quality checks & cleaning pipeline
│   ├── eda.py                        ← EDA chart functions
│   ├── kpi.py                        ← Portfolio & segment KPI computation
│   ├── risk_analysis.py              ← Risk scoring, tiers, key drivers
│   ├── feature_engineering.py        ← sklearn preprocessing pipeline
│   ├── model.py                      ← Train, evaluate, save, inference
│   └── insights.py                   ← Business insights & recommendations
│
├── models/                           ← Saved model artefacts (after training)
│   ├── deployment_model.joblib       ← Lightweight model for deployment
│   ├── .gitkeep
├── scripts/
│   └── create_deployment_model.py    ← Creates lightweight deployment model
│
├── outputs/
│   ├── figures/                      ← Generated charts (PNG)
│   └── tables/                       ← Generated summary tables (CSV)
│
├── dashboard/                        ← Streamlit multi-page application
│   ├── app.py                        ← Home page & entry point
│   └── pages/
│       ├── 01_overview.py            ← Portfolio KPIs
│       ├── 02_eda.py                 ← Exploratory Data Analysis
│       ├── 03_risk_analysis.py       ← Risk scoring & insights
│       ├── 04_model_performance.py   ← Model metrics & evaluation plots
│       └── 05_prediction.py          ← Single-loan prediction form
│
├── report/
│   └── Loan_Default_Risk_Report.docx ← Full project report (Word)
│
├── requirements.txt
├── run_pipeline.py                   ← One-shot: clean → train → save
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip

### 1. Navigate to the project directory
```bash
cd Loan_Default_Risk_Analytics
```

### 2. Create and activate a virtual environment (recommended)
```bash
# Create
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS / Linux
source venv/bin/activate
```

### 3. Install all dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train the full pipeline (data → model → saved artefacts)
```bash
python run_pipeline.py
```
This loads and cleans the data, trains both a Logistic Regression baseline and a Random Forest model, evaluates them, and saves all artefacts to `models/`.

### Launch the interactive Streamlit dashboard
```bash
streamlit run dashboard/app.py
```
Then open `http://localhost:8501` in your browser.

### Run individual Jupyter notebooks
```bash
jupyter notebook notebooks/01_data_loading_understanding.ipynb
```
Run notebooks in order (01 → 06) for a guided walkthrough of the full project.

---

## Project Phases

| # | Phase | Key Files | Status |
|---|---|---|---|
| 1 | Project scaffold & configuration | `src/config.py`, `requirements.txt`, `README.md` | ✅ Complete |
| 2 | Data loading & understanding | `src/data_loader.py`, Notebook 01 | ✅ Complete |
| 3 | Data quality & cleaning | `src/data_cleaner.py`, Notebook 02 | ✅ Complete |
| 4 | Exploratory Data Analysis | `src/eda.py`, Notebook 03 | ✅ Complete |
| 5 | KPIs & Risk Analysis | `src/kpi.py`, `src/risk_analysis.py`, `src/insights.py`, Notebook 04 | ✅ Complete |
| 6 | Feature engineering & ML model | `src/feature_engineering.py`, `src/model.py`, Notebooks 05–06 | ✅ Complete |
| 7 | Streamlit dashboard | `dashboard/` | ✅ Complete |
| 8 | Report & documentation | `report/`, `README.md` | ✅ Complete |

---

## Key Findings

### Data Quality
- 255,347 rows, 18 columns — **zero missing values, zero duplicates**
- All categorical values within expected domains
- All numeric values within sensible bounds

### Exploratory Data Analysis
| Finding | Detail |
|---|---|
| Overall default rate | **11.61%** (class imbalance ~8:1) |
| Strongest numeric correlate | Age (|r| = 0.168) |
| Highest-risk employment | Unemployed (13.55% default rate) |
| Lowest-risk employment | Full-time (9.46% default rate) |
| Highest-risk loan purpose | Business (12.33%) |
| Age insight | Younger borrowers (18–25) default at **20.76%** vs 5.50% for 56–69 |

### KPIs
| KPI | Value |
|---|---|
| Total portfolio value | $32.6 billion |
| Defaulted loan value | $4.3 billion (13.15% of portfolio) |
| Avg loan amount | $127,579 |
| Avg interest rate | 13.49% |
| Avg credit score | 574 |

### Risk Tiers (Heuristic Score)
| Tier | Loans | Default Rate |
|---|---|---|
| Low | 63,245 | 5.73% |
| Medium | 89,714 | 9.74% |
| High | 63,679 | 14.05% |
| Very High | 38,709 | **21.56%** |

### Top Key Drivers of Default (by relative mean difference)
1. **Interest Rate** — 20.6% higher for defaulters
2. **Age** — 17.7% lower for defaulters (younger = more risk)
3. **Months Employed** — 17.3% lower for defaulters
4. **Loan Amount** — 15.3% higher for defaulters
5. **Income** — 14.4% lower for defaulters

### Machine Learning Model
| Metric | Random Forest | Logistic Regression |
|---|---|---|
| ROC-AUC | *run pipeline to populate* | *run pipeline to populate* |
| Avg Precision | *run pipeline to populate* | *run pipeline to populate* |
| F1 (Default) | *run pipeline to populate* | *run pipeline to populate* |
| Decision threshold | 0.40 | 0.40 |

---

## Business Recommendations

1. **Tighten underwriting for Unemployed borrowers** — 13.55% default rate vs 9.46% for Full-time
2. **Apply stricter criteria to borrowers aged 18–35** — default rates 2–4× higher than older segments
3. **Use the ML model as a pre-approval score** — flag applications with predicted probability ≥ 40%
4. **Require co-signers for Very High-risk tier** — 21.56% default rate represents concentrated portfolio risk
5. **Review Business loan collateral requirements** — highest default rate among loan purposes
6. **Set differentiated credit score floors** by employment type and loan purpose

---

## Technology Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| pandas | ≥ 2.0 | Data manipulation |
| NumPy | ≥ 1.24 | Numerical computing |
| matplotlib | ≥ 3.7 | Visualisation |
| seaborn | ≥ 0.12 | Statistical charts |
| scikit-learn | ≥ 1.3 | ML preprocessing & models |
| joblib | ≥ 1.3 | Model persistence |
| Streamlit | ≥ 1.32 | Interactive dashboard |
| python-docx | ≥ 1.1 | Report generation |
| Jupyter | ≥ 1.0 | Analytical notebooks |

---

### Deployment Model

For Streamlit deployment, a lightweight Random Forest deployment model is provided:

`models/deployment_model.joblib`

This model is optimized for deployment size while preserving the preprocessing pipeline and prediction functionality required by the dashboard.

---

## Author

Akula Srinithya
**IBM SkillsBuild Data Analytics Academic Internship**
Project: *Loan Default Risk Analysis and Prediction Using Data Analytics and Machine Learning*

---

## License

Submitted as part of an IBM SkillsBuild academic internship programme.
Dataset and analysis are for educational purposes only.
