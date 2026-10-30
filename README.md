# Breast Cancer Cohort Explorer

An interactive Streamlit dashboard based on the analysis in EDA/Breast_Cancer.ipynb. It explores age, tumor size, clinical and nodal stage, histologic grade, race, marital status, and differentiation.

## Run locally on Windows

From this project folder:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    streamlit run app.py

If PowerShell blocks activation, run the app directly without activating:

    .\.venv\Scripts\python.exe -m streamlit run app.py

## Add the dataset

The notebook reads Breast_Cancer-selected-columns.csv from a Google Colab path. That CSV is not included in this project. Upload it in the dashboard sidebar, or place it in the project root, EDA, or data folder. The dashboard also checks for Breast_Cancer.csv in the project root and data.

The CSV should contain these columns:

- Age
- Race
- Marital Status
- N Stage
- 6th Stage
- differentiate
- Grade
- A Stage
- Tumor Size

T Stage is optional. Fully empty columns are excluded from the analysis, matching the notebook's handling of its entirely missing T Stage field. Exact duplicate rows are removed by default, also matching the notebook.

## Dashboard views

- **Overview:** cohort metrics, stage and grade counts, age and tumor size distributions, and average tumor size by stage and grade.
- **Stage & grade:** grade mix within each stage, tumor size by grade, age across stages, and the age versus tumor size relationship.
- **Explore records:** race and marital status distributions, filtered record table, and CSV download.
- **Data quality:** missing values, unique values, source row count, and duplicate removal summary.

All charts respond to the sidebar filters. Findings describe the records in the supplied dataset; they are not population estimates or clinical guidance.
