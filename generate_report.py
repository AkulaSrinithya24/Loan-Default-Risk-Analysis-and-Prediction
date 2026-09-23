"""
generate_report.py
------------------
Generates the full project report as a Word (.docx) file.

Run from the project root:
    python generate_report.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime


# ── Helpers ────────────────────────────────────────────────────────────────────

def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return h


def add_paragraph(doc, text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    return p


def add_table(doc, headers, rows, style="Light List Accent 1"):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = style
    table.alignment = WD_TABLE_ALIGNMENT.LEFT

    # Header row
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        hdr[i].paragraphs[0].runs[0].bold = True

    # Data rows
    for r_idx, row in enumerate(rows):
        cells = table.rows[r_idx + 1].cells
        for c_idx, val in enumerate(row):
            cells[c_idx].text = str(val)

    return table


def add_horizontal_rule(doc):
    doc.add_paragraph("─" * 80)


# ── Report content ─────────────────────────────────────────────────────────────

def build_report(output_path: Path):
    doc = Document()

    # Page margins
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(2.5)
        section.right_margin  = Cm(2.5)

    # ── Cover ──────────────────────────────────────────────────────────────────
    doc.add_paragraph()
    title = doc.add_heading("Loan Default Risk Analysis and Prediction", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph("Using Data Analytics and Machine Learning")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].font.size = Pt(14)
    subtitle.runs[0].bold = True

    doc.add_paragraph()

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run("IBM SkillsBuild Data Analytics Academic Internship\n").bold = True
    meta.add_run(f"Date: {datetime.date.today().strftime('%B %d, %Y')}\n")
    meta.add_run("Dataset: Loan_default.csv  |  255,347 records  |  18 features")

    doc.add_page_break()

    # ── 1. Executive Summary ───────────────────────────────────────────────────
    add_heading(doc, "1. Executive Summary")
    add_paragraph(doc, (
        "This report presents a complete end-to-end data analytics and machine learning "
        "project focused on predicting and understanding loan default risk. Using a dataset "
        "of 255,347 loan records, the project applies structured phases of data loading, "
        "quality checking, exploratory analysis, KPI computation, risk segmentation, and "
        "machine learning modelling to derive actionable business intelligence."
    ))
    add_paragraph(doc, (
        "The overall portfolio default rate is 11.61%, representing 29,653 defaulted loans "
        "with a combined value of $4.3 billion out of a $32.6 billion total portfolio. "
        "Key risk drivers include Interest Rate, Age, Months Employed, Loan Amount, and Income. "
        "Unemployed borrowers and younger borrowers (18-25) carry disproportionately high default rates. "
        "A Random Forest classifier is trained to predict default probability, enabling "
        "risk-based pre-approval scoring."
    ))

    doc.add_paragraph()

    # ── 2. Project Objectives ─────────────────────────────────────────────────
    add_heading(doc, "2. Project Objectives")
    objectives = [
        "Understand the structure, quality, and statistical properties of the loan dataset.",
        "Identify and quantify key drivers of loan default.",
        "Compute portfolio-level and segment-level KPIs.",
        "Segment borrowers into risk tiers using a heuristic risk score.",
        "Build and evaluate a machine learning model for default prediction.",
        "Surface actionable business insights and recommendations.",
        "Deliver an interactive Streamlit dashboard for stakeholder use.",
    ]
    for i, obj in enumerate(objectives, 1):
        p = doc.add_paragraph(style="List Number")
        p.add_run(obj)

    doc.add_paragraph()

    # ── 3. Dataset Description ────────────────────────────────────────────────
    add_heading(doc, "3. Dataset Description")
    add_paragraph(doc, (
        "The dataset contains 255,347 loan records with 18 columns, including a binary target "
        "variable (Default: 0 = No Default, 1 = Default). The dataset is clean with no missing "
        "values and no duplicate records."
    ))

    add_heading(doc, "3.1 Feature Summary", level=2)
    add_table(doc,
        headers=["Feature", "Type", "Range / Values", "Description"],
        rows=[
            ["Age",             "Numeric",     "18 – 69",                           "Borrower age"],
            ["Income",          "Numeric",     "$15,000 – $150,000",                "Annual income"],
            ["LoanAmount",      "Numeric",     "$5,000 – $250,000",                 "Loan size"],
            ["CreditScore",     "Numeric",     "300 – 849",                         "Credit score"],
            ["MonthsEmployed",  "Numeric",     "0 – 119",                           "Months at current job"],
            ["NumCreditLines",  "Numeric",     "1 – 4",                             "Open credit lines"],
            ["InterestRate",    "Numeric",     "2.0 – 25.0%",                       "Loan interest rate"],
            ["LoanTerm",        "Numeric",     "12 / 24 / 36 / 48 / 60 months",    "Loan term"],
            ["DTIRatio",        "Numeric",     "0.10 – 0.90",                       "Debt-to-income ratio"],
            ["Education",       "Categorical", "High School / Bachelor's / Master's / PhD", "Education level"],
            ["EmploymentType",  "Categorical", "Full-time / Part-time / Self-employed / Unemployed", "Employment type"],
            ["MaritalStatus",   "Categorical", "Single / Married / Divorced",       "Marital status"],
            ["LoanPurpose",     "Categorical", "Auto / Business / Education / Home / Other", "Loan purpose"],
            ["HasMortgage",     "Binary",      "Yes / No",                          "Existing mortgage"],
            ["HasDependents",   "Binary",      "Yes / No",                          "Has dependents"],
            ["HasCoSigner",     "Binary",      "Yes / No",                          "Co-signer present"],
            ["Default",         "Target",      "0 / 1",                             "Loan defaulted (1=Yes)"],
        ]
    )

    doc.add_paragraph()

    # ── 4. Data Quality ───────────────────────────────────────────────────────
    add_heading(doc, "4. Data Quality Assessment")
    add_paragraph(doc, (
        "A comprehensive data quality check was performed covering missing values, duplicate rows, "
        "invalid categorical values, out-of-range numerics, invalid target values, and negative values "
        "in non-negative columns."
    ))
    add_table(doc,
        headers=["Check", "Result"],
        rows=[
            ["Missing values",          "None detected"],
            ["Duplicate rows",          "None detected"],
            ["Invalid categorical values", "None detected"],
            ["Out-of-range numerics",   "None detected"],
            ["Invalid target values",   "None detected"],
            ["Negative values",         "None detected"],
            ["Overall QA status",       "PASSED — 0 issues"],
        ]
    )
    add_paragraph(doc, (
        "\nCleaning steps applied: binary Yes/No columns encoded to 1/0; LoanID column dropped; "
        "numeric features clipped to domain bounds (defensive); index reset."
    ))

    doc.add_paragraph()

    # ── 5. Exploratory Data Analysis ─────────────────────────────────────────
    add_heading(doc, "5. Exploratory Data Analysis")

    add_heading(doc, "5.1 Target Variable", level=2)
    add_paragraph(doc, (
        "The dataset is imbalanced: 88.39% of loans did not default (225,694) and 11.61% defaulted "
        "(29,653). This ~8:1 class imbalance is handled in the model using class_weight='balanced'."
    ))

    add_heading(doc, "5.2 Numeric Feature Correlations with Default", level=2)
    add_table(doc,
        headers=["Feature", "|Pearson r| with Default", "Direction"],
        rows=[
            ["Age",            "0.168", "Lower age -> higher risk"],
            ["InterestRate",   "0.131", "Higher rate -> higher risk"],
            ["Income",         "0.099", "Lower income -> higher risk"],
            ["MonthsEmployed", "0.097", "Less tenure -> higher risk"],
            ["LoanAmount",     "0.087", "Larger loan -> higher risk"],
            ["CreditScore",    "0.034", "Lower score -> higher risk"],
            ["NumCreditLines", "0.028", "More lines -> slightly higher risk"],
            ["DTIRatio",       "0.019", "Higher ratio -> higher risk"],
            ["LoanTerm",       "0.001", "Negligible"],
        ]
    )

    add_heading(doc, "5.3 Default Rate by Employment Type", level=2)
    add_table(doc,
        headers=["Employment Type", "Default Rate (%)", "Total Loans"],
        rows=[
            ["Unemployed",     "13.55%", "63,824"],
            ["Part-time",      "11.97%", "64,161"],
            ["Self-employed",  "11.46%", "63,706"],
            ["Full-time",       "9.46%", "63,656"],
        ]
    )

    add_heading(doc, "5.4 Default Rate by Age Band", level=2)
    add_table(doc,
        headers=["Age Band", "Default Rate (%)", "Total Loans"],
        rows=[
            ["18-25", "20.76%", "39,016"],
            ["26-35", "16.08%", "49,408"],
            ["36-45", "11.55%", "49,220"],
            ["46-55",  "8.45%", "49,148"],
            ["56-69",  "5.50%", "68,555"],
        ]
    )

    add_heading(doc, "5.5 Default Rate by Loan Purpose", level=2)
    add_table(doc,
        headers=["Loan Purpose", "Default Rate (%)"],
        rows=[
            ["Business",  "12.33%"],
            ["Auto",      "11.88%"],
            ["Education", "11.65%"],
            ["Other",     "11.44%"],
            ["Home",      "11.03%"],
        ]
    )

    doc.add_paragraph()

    # ── 6. KPI Analysis ───────────────────────────────────────────────────────
    add_heading(doc, "6. Portfolio KPI Analysis")
    add_table(doc,
        headers=["KPI", "Value"],
        rows=[
            ["Total Loans",              "255,347"],
            ["Total Defaulted",          "29,653  (11.61%)"],
            ["Total Non-Defaulted",      "225,694  (88.39%)"],
            ["Total Loan Value",         "$32,576,880,572"],
            ["Defaulted Loan Value",     "$4,285,312,531  (13.15%)"],
            ["Avg Loan Amount",          "$127,579"],
            ["Median Loan Amount",       "$127,556"],
            ["Avg Credit Score",         "574"],
            ["Avg Annual Income",        "$82,499"],
            ["Avg Interest Rate",        "13.49%"],
            ["Avg DTI Ratio",            "0.500"],
            ["Avg Loan Term",            "36.0 months"],
            ["Avg Months Employed",      "59.5"],
        ]
    )

    doc.add_paragraph()

    # ── 7. Risk Analysis ──────────────────────────────────────────────────────
    add_heading(doc, "7. Risk Analysis")

    add_heading(doc, "7.1 Heuristic Risk Score", level=2)
    add_paragraph(doc, (
        "A weighted heuristic risk score (0-100) was computed for each loan using the following "
        "components: Credit Score (inverse, 25%), DTI Ratio (20%), Interest Rate (20%), "
        "Income (inverse, 15%), Months Employed (inverse, 10%), Loan Amount (10%)."
    ))

    add_heading(doc, "7.2 Risk Tier Summary", level=2)
    add_table(doc,
        headers=["Risk Tier", "Loans", "Default Rate (%)", "Avg Risk Score", "Default Value ($)"],
        rows=[
            ["Low",       "63,245",  "5.73%",  "34.0", "$414,565,617"],
            ["Medium",    "89,714",  "9.74%",  "47.4", "$1,177,462,475"],
            ["High",      "63,679", "14.05%",  "57.8", "$1,315,602,683"],
            ["Very High", "38,709", "21.56%",  "69.2", "$1,377,681,756"],
        ]
    )

    add_heading(doc, "7.3 Key Numeric Drivers", level=2)
    add_table(doc,
        headers=["Feature", "Mean (Default)", "Mean (No Default)", "Rel Diff (%)", "Risk Direction"],
        rows=[
            ["InterestRate",   "15.896", "13.177", "20.6%", "Higher -> more risk"],
            ["Age",            "40.7",   "44.0",   "17.7%", "Lower -> more risk"],
            ["MonthsEmployed", "51.3",   "61.3",   "17.3%", "Lower -> more risk"],
            ["LoanAmount",     "147,346","127,983", "15.3%", "Higher -> more risk"],
            ["Income",         "70,600", "84,103",  "14.4%", "Lower -> more risk"],
            ["NumCreditLines", "2.60",   "2.50",    "4.0%",  "Higher -> more risk"],
            ["CreditScore",    "559",    "576",     "2.9%",  "Lower -> more risk"],
            ["DTIRatio",       "0.514",  "0.500",   "2.8%",  "Higher -> more risk"],
        ]
    )

    doc.add_paragraph()

    # ── 8. Machine Learning Model ─────────────────────────────────────────────
    add_heading(doc, "8. Machine Learning Model")

    add_heading(doc, "8.1 Preprocessing Pipeline", level=2)
    add_table(doc,
        headers=["Column Group", "Transformer", "Columns"],
        rows=[
            ["Numeric (9)",     "StandardScaler",  "Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio"],
            ["Categorical (4)", "OrdinalEncoder",  "Education, EmploymentType, MaritalStatus, LoanPurpose"],
            ["Binary (3)",      "Passthrough",     "HasMortgage, HasDependents, HasCoSigner"],
        ]
    )

    add_heading(doc, "8.2 Models Trained", level=2)
    add_table(doc,
        headers=["Model", "Parameters", "Role"],
        rows=[
            ["Logistic Regression", "class_weight=balanced, max_iter=1000", "Baseline"],
            ["Random Forest",       "n_estimators=200, class_weight=balanced", "Primary model"],
        ]
    )

    add_heading(doc, "8.3 Training Configuration", level=2)
    add_table(doc,
        headers=["Setting", "Value"],
        rows=[
            ["Train / Test Split",    "80% / 20% (stratified)"],
            ["Train rows",            "204,277"],
            ["Test rows",             "51,070"],
            ["Random seed",           "42"],
            ["Decision threshold",    "0.40 (tuned for class imbalance)"],
            ["Leakage prevention",    "Preprocessor fitted on training data only"],
        ]
    )

    add_heading(doc, "8.4 Model Performance", level=2)
    add_paragraph(doc, (
        "Run python run_pipeline.py to populate the metrics below. "
        "The table format is ready to fill in after training."
    ), italic=True)
    add_table(doc,
        headers=["Metric", "Logistic Regression", "Random Forest"],
        rows=[
            ["ROC-AUC",          "–", "–"],
            ["Avg Precision",    "–", "–"],
            ["F1 (Default)",     "–", "–"],
            ["Accuracy",         "–", "–"],
        ]
    )

    doc.add_paragraph()

    # ── 9. Business Insights ──────────────────────────────────────────────────
    add_heading(doc, "9. Business Insights & Recommendations")

    insights = [
        ("High", "Portfolio Risk",
         "Overall portfolio default rate is 11.61%.",
         "Tighten underwriting criteria to reduce portfolio default rate below 8%."),
        ("High", "Key Numeric Driver",
         "Interest Rate is the strongest numeric driver of default (20.6% relative difference).",
         "Incorporate Interest Rate as a primary screening variable in loan approval workflows."),
        ("High", "Employment Risk",
         "Unemployed borrowers default at 13.55% vs 9.46% for Full-time borrowers.",
         "Apply stricter income verification and lower LTV ratios for Unemployed applicants."),
        ("Medium", "Loan Purpose Risk",
         "Business loans show the highest default rate (12.33%).",
         "Review collateral requirements and repayment terms for Business loans."),
        ("High", "Risk Tier Concentration",
         "Very High-risk tier: 38,709 loans at 21.56% default rate.",
         "Consider loan caps or mandatory co-signer requirements for Very High-risk borrowers."),
        ("Medium", "Credit Score Opportunity",
         "Defaulters average credit score 559 vs 576 for non-defaulters.",
         "Set minimum credit score thresholds differentiated by loan purpose and employment type."),
        ("Medium", "Interest Rate Policy",
         "Higher rates correlate with higher default — indicative of risk pricing.",
         "Ensure interest rate bands reflect risk tiers. Avoid rate escalation for high-risk borrowers."),
        ("High", "Predictive Modelling",
         "Random Forest can predict default risk before loan disbursement.",
         "Deploy the ML model as a pre-approval scoring tool; flag applications above 40% probability."),
    ]

    for priority, category, finding, recommendation in insights:
        add_heading(doc, f"[{priority}] {category}", level=2)
        p = doc.add_paragraph()
        p.add_run("Finding: ").bold = True
        p.add_run(finding)
        p2 = doc.add_paragraph()
        p2.add_run("Recommendation: ").bold = True
        p2.add_run(recommendation)

    doc.add_paragraph()

    # ── 10. Dashboard ─────────────────────────────────────────────────────────
    add_heading(doc, "10. Streamlit Dashboard")
    add_paragraph(doc, (
        "An interactive Streamlit dashboard has been built with five pages:"
    ))
    add_table(doc,
        headers=["Page", "Content"],
        rows=[
            ["Overview & KPIs",    "8 portfolio KPI cards, class balance charts, interactive segment KPI analysis"],
            ["EDA",                "Feature distributions, correlation heatmap, categorical/binary default rates, cross-feature heatmaps"],
            ["Risk Analysis",      "Risk tier summary, heuristic score plots, key drivers chart, business insights"],
            ["Model Performance",  "ROC curve, PR curve, confusion matrix, feature importance, threshold analysis"],
            ["Prediction",         "16-feature loan input form with probability gauge, risk tier label, approval recommendation"],
        ]
    )
    add_paragraph(doc, "\nTo launch: streamlit run dashboard/app.py")

    doc.add_paragraph()

    # ── 11. Project Architecture ──────────────────────────────────────────────
    add_heading(doc, "11. Project Architecture & Technology Stack")
    add_table(doc,
        headers=["Component", "Technology", "Purpose"],
        rows=[
            ["Data processing",     "Python, pandas, NumPy", "Load, clean, transform data"],
            ["Visualisation",       "matplotlib, seaborn",   "Charts for EDA, KPIs, model evaluation"],
            ["Machine learning",    "scikit-learn",          "Preprocessing pipeline, RF, LR models"],
            ["Model persistence",   "joblib",                "Save and load trained model artefacts"],
            ["Dashboard",           "Streamlit",             "Interactive multi-page web application"],
            ["Notebooks",           "Jupyter",               "Step-by-step analytical walkthroughs"],
            ["Report",              "python-docx",           "Automated Word document generation"],
        ]
    )

    doc.add_paragraph()

    # ── 12. Conclusions ───────────────────────────────────────────────────────
    add_heading(doc, "12. Conclusions")
    add_paragraph(doc, (
        "This project successfully demonstrates a professional, end-to-end data analytics "
        "workflow applied to a real-world loan default problem. Key conclusions are:"
    ))
    conclusions = [
        "The dataset is of high quality — no cleaning issues were found, enabling immediate analysis.",
        "Loan default is driven primarily by Interest Rate, Age, Employment Tenure, Loan Amount, and Income.",
        "Unemployed and younger borrowers carry significantly higher default risk than the portfolio average.",
        "The heuristic risk score effectively stratifies loans into tiers with monotonically increasing default rates.",
        "A Random Forest model trained with balanced class weights can predict default probability at the individual loan level.",
        "Deploying the model as a pre-approval scoring tool — flagging applications at or above a 40% predicted probability — has direct commercial value in reducing portfolio default exposure.",
    ]
    for c in conclusions:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(c)

    doc.add_paragraph()

    # ── 13. Appendix ──────────────────────────────────────────────────────────
    add_heading(doc, "13. Appendix — File Reference")
    add_table(doc,
        headers=["File", "Description"],
        rows=[
            ["src/config.py",              "Central configuration — all paths, constants, model parameters"],
            ["src/data_loader.py",         "Load and validate raw CSV"],
            ["src/data_cleaner.py",        "Quality checks and cleaning pipeline"],
            ["src/eda.py",                 "Reusable EDA chart functions"],
            ["src/kpi.py",                 "Portfolio and segment KPI computations"],
            ["src/risk_analysis.py",       "Risk scoring, tier assignment, key driver analysis"],
            ["src/insights.py",            "Business insights and recommendations"],
            ["src/feature_engineering.py", "scikit-learn preprocessing pipeline builder"],
            ["src/model.py",               "Model training, evaluation, persistence, inference"],
            ["run_pipeline.py",            "One-shot pipeline: load -> clean -> train -> save"],
            ["dashboard/app.py",           "Streamlit home page"],
            ["dashboard/pages/01_overview.py",  "KPI dashboard page"],
            ["dashboard/pages/02_eda.py",       "EDA page"],
            ["dashboard/pages/03_risk_analysis.py", "Risk analysis page"],
            ["dashboard/pages/04_model_performance.py", "Model evaluation page"],
            ["dashboard/pages/05_prediction.py","Prediction interface page"],
        ]
    )

    doc.add_paragraph()
    add_paragraph(doc,
        f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
        "IBM SkillsBuild Data Analytics Academic Internship",
        italic=True, size=9
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    print(f"Report saved -> {output_path}")


if __name__ == "__main__":
    build_report(Path("report/Loan_Default_Risk_Report.docx"))
