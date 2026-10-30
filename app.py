from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Breast Cancer Cohort Explorer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

REQUIRED_COLUMNS = {
    "Age",
    "Race",
    "Marital Status",
    "N Stage",
    "6th Stage",
    "differentiate",
    "Grade",
    "A Stage",
    "Tumor Size",
}
STAGE_ORDER = ["IIA", "IIB", "IIIA", "IIIB", "IIIC"]
GRADE_COLORS = {
    "1": "#3f8f86",
    "2": "#6e83bd",
    "3": "#d77b6b",
    "anaplastic; Grade IV": "#7b568b",
}
ACCENT = "#b64c68"
INK = "#222b3b"
MUTED = "#718096"


st.markdown(
    """
    <style>
      :root { --ink: #222b3b; --muted: #718096; --accent: #b64c68; }
      .stApp { background: #f5f7fb; color: var(--ink); }
      [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e7eaf1; }
      [data-testid="stSidebar"] > div:first-child { padding-top: 1.35rem; }
      .block-container { padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1440px; }
      .hero {
        background: linear-gradient(115deg, #ffffff 0%, #fff7f8 100%);
        border: 1px solid #ece7eb; border-radius: 18px; padding: 1.45rem 1.7rem;
        margin: 0.15rem 0 1.15rem 0; box-shadow: 0 5px 20px rgba(35, 43, 59, 0.035);
      }
      .eyebrow { color: var(--accent); font-size: 0.74rem; font-weight: 750;
        letter-spacing: .13em; text-transform: uppercase; margin-bottom: .35rem; }
      .hero h1 { color: var(--ink); font-size: 2rem; line-height: 1.15;
        margin: 0 0 .45rem 0; font-weight: 760; }
      .hero p { color: var(--muted); margin: 0; font-size: .98rem; }
      .section-title { color: var(--ink); font-size: 1.15rem; font-weight: 700;
        margin: .35rem 0 .7rem 0; }
      .subtle { color: var(--muted); font-size: .88rem; }
      .note {
        border-left: 4px solid var(--accent); background: #fff; border-radius: 7px;
        padding: .9rem 1rem; color: #4d586c; margin: .6rem 0 1rem 0;
      }
      div[data-testid="stMetric"] {
        background: #fff; border: 1px solid #e8ebf2; border-radius: 13px;
        padding: .95rem 1rem; box-shadow: 0 4px 16px rgba(35, 43, 59, .035);
      }
      div[data-testid="stMetricLabel"] p { color: var(--muted); font-size: .82rem; }
      div[data-testid="stMetricValue"] { color: var(--ink); }
      div[data-testid="stTabs"] button { font-weight: 650; }
      .footer { border-top: 1px solid #e6e9ef; color: var(--muted);
        font-size: .78rem; margin-top: 2rem; padding-top: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def find_local_dataset() -> Path | None:
    """Find the notebook's expected CSV in common project locations."""
    project_dir = Path(__file__).resolve().parent
    candidates = [
        project_dir / "Breast_Cancer-selected-columns.csv",
        project_dir / "EDA" / "Breast_Cancer-selected-columns.csv",
        project_dir / "data" / "Breast_Cancer-selected-columns.csv",
        project_dir / "Breast_Cancer.csv",
        project_dir / "data" / "Breast_Cancer.csv",
    ]
    return next((path for path in candidates if path.is_file()), None)


@st.cache_data(show_spinner=False)
def read_csv_bytes(contents: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(contents))


def tidy_data(raw: pd.DataFrame, remove_duplicates: bool) -> tuple[pd.DataFrame, int, list[str]]:
    frame = raw.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.loc[:, ~frame.isna().all()]
    missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    duplicate_count = int(frame.duplicated().sum())
    if remove_duplicates:
        frame = frame.drop_duplicates().copy()
    for column in ("Age", "Tumor Size"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, duplicate_count, missing_columns


def format_number(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}"


def grade_order(values: list[str]) -> list[str]:
    preferred = ["1", "2", "3", "anaplastic; Grade IV"]
    return [value for value in preferred if value in values] + [
        value for value in values if value not in preferred
    ]


def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Cohort filters")

    age_values = data["Age"].dropna()
    if age_values.empty:
        age_range = None
        st.sidebar.caption("No numeric age values are available to filter.")
    else:
        age_min, age_max = int(age_values.min()), int(age_values.max())
        if age_min == age_max:
            age_range = (age_min, age_max)
            st.sidebar.caption(f"Age: {age_min}")
        else:
            age_range = st.sidebar.slider(
                "Age range",
                min_value=age_min,
                max_value=age_max,
                value=(age_min, age_max),
                step=1,
            )

    result = data.copy()
    if age_range is not None:
        result = result[result["Age"].between(*age_range)]

    categorical_filters = [
        ("6th Stage", "Clinical stage"),
        ("Grade", "Histologic grade"),
        ("N Stage", "Nodal stage"),
        ("Race", "Race"),
        ("Marital Status", "Marital status"),
        ("A Stage", "Extension"),
        ("differentiate", "Differentiation"),
    ]
    for column, label in categorical_filters:
        values = result[column].dropna().astype(str).unique().tolist()
        if column == "6th Stage":
            values.sort(key=lambda value: STAGE_ORDER.index(value) if value in STAGE_ORDER else 99)
        elif column == "Grade":
            values = grade_order(values)
        else:
            values.sort()
        selected = st.sidebar.multiselect(label, options=values, default=values)
        result = result[result[column].astype(str).isin(selected)]

    return result


def make_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=INK, size=12),
        margin=dict(l=12, r=12, t=42, b=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white", font_color=INK),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e3e7ee", zeroline=False)
    fig.update_yaxes(gridcolor="#edf0f5", zeroline=False)
    return fig


def show_empty_chart(message: str = "No records match the selected filters.") -> None:
    st.info(message)


st.sidebar.markdown(
    """
    <div style="padding:0 .1rem 1rem .1rem">
      <div style="font-weight:800;font-size:1.05rem;color:#222b3b">COHORT EXPLORER</div>
      <div style="font-size:.78rem;color:#718096;margin-top:.2rem">Breast cancer dataset</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("### Dataset")
uploaded = st.sidebar.file_uploader("Upload the notebook CSV", type=["csv"])
remove_duplicates = st.sidebar.checkbox(
    "Remove duplicate rows", value=True, help="Matches the notebook's duplicate-removal step."
)

local_path = find_local_dataset() if uploaded is None else None
source_name = uploaded.name if uploaded is not None else (str(local_path.name) if local_path else None)
load_error = None
raw_data = None
if uploaded is not None:
    try:
        raw_data = read_csv_bytes(uploaded.getvalue())
    except Exception as error:
        load_error = f"Could not read the uploaded CSV: {error}"
elif local_path is not None:
    try:
        raw_data = read_csv_bytes(local_path.read_bytes())
    except Exception as error:
        load_error = f"Could not read {local_path.name}: {error}"

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Interactive cohort analysis</div>
      <h1>Breast Cancer Cohort Explorer</h1>
      <p>Explore the notebook dataset through stage, tumor size, age, grade and cohort characteristics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_error:
    st.error(load_error)
    st.stop()

if raw_data is None:
    st.info(
        "The notebook references Breast_Cancer-selected-columns.csv, but that CSV is not in this project yet. "
        "Upload it in the sidebar, or place it beside app.py, in EDA, or in data to populate the dashboard."
    )
    st.markdown('<div class="section-title">What the notebook reports</div>', unsafe_allow_html=True)
    ref1, ref2, ref3 = st.columns(3)
    ref1.metric("Source rows", "4,024")
    ref2.metric("Rows after notebook cleaning", "3,519")
    ref3.metric("Columns used for analysis", "9")
    st.markdown(
        """
        <div class="note">
          The notebook drops the fully empty T Stage field and removes duplicate rows. Its CSV was stored at a
          Colab path and is not included in this workspace, so record-level charts will appear after you upload
          the file. The dashboard does not generate substitute patient records.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Expected columns:** Age, Race, Marital Status, N Stage, 6th Stage, differentiate, Grade, A Stage, Tumor Size.")
    st.markdown(
        '<div class="footer">Exploratory descriptive analysis · Patterns in this file are not population estimates or clinical guidance.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

data, duplicate_count, missing_columns = tidy_data(raw_data, remove_duplicates)
if missing_columns:
    st.error("This CSV does not match the notebook schema. Missing columns: " + ", ".join(missing_columns))
    st.caption("Upload the selected-columns file used by the notebook, including Age, stage, grade and Tumor Size fields.")
    st.stop()

st.sidebar.caption(f"Loaded: {source_name}")
st.sidebar.caption(f"{len(raw_data):,} source rows · {len(data):,} after cleaning")
filtered = apply_filters(data)

st.markdown('<div class="section-title">Cohort snapshot</div>', unsafe_allow_html=True)
metric_cols = st.columns(5)
metric_values = [
    ("Records", f"{len(filtered):,}"),
    ("Median age", f"{format_number(filtered['Age'].median(), 0)} years"),
    ("Median tumor size", f"{format_number(filtered['Tumor Size'].median(), 0)} mm"),
    ("Mean tumor size", f"{format_number(filtered['Tumor Size'].mean(), 1)} mm"),
    ("Stages represented", f"{filtered['6th Stage'].nunique():,}"),
]
for column, (label, value) in zip(metric_cols, metric_values):
    column.metric(label, value)

if duplicate_count:
    removed_note = f"{duplicate_count:,} duplicate rows removed, following the notebook cleaning step." if remove_duplicates else f"{duplicate_count:,} duplicate rows are present; duplicate removal is turned off."
else:
    removed_note = "No exact duplicate rows were found after empty columns were removed."
st.caption(
    f"Showing {len(filtered):,} of {len(data):,} cleaned records from {source_name}. "
    f"{removed_note} Entirely blank columns are excluded."
)

overview_tab, stage_tab, explore_tab, quality_tab = st.tabs(
    ["Overview", "Stage & grade", "Explore records", "Data quality"]
)

with overview_tab:
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="section-title">Records by clinical stage</div>', unsafe_allow_html=True)
        if filtered.empty:
            show_empty_chart()
        else:
            stage_counts = (
                filtered.groupby(["6th Stage", "Grade"], dropna=False)
                .size()
                .reset_index(name="Records")
            )
            stage_order = [stage for stage in STAGE_ORDER if stage in filtered["6th Stage"].astype(str).unique()]
            stage_order += sorted(set(filtered["6th Stage"].astype(str)) - set(stage_order))
            fig = px.bar(
                stage_counts,
                x="6th Stage",
                y="Records",
                color="Grade",
                category_orders={"6th Stage": stage_order, "Grade": grade_order(filtered["Grade"].astype(str).unique().tolist())},
                color_discrete_map=GRADE_COLORS,
                barmode="stack",
                labels={"6th Stage": "Clinical stage"},
                hover_data={"Records": ":,"},
            )
            st.plotly_chart(make_layout(fig, 390), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown('<div class="section-title">Tumor size distribution</div>', unsafe_allow_html=True)
        if filtered.empty:
            show_empty_chart()
        else:
            fig = px.histogram(
                filtered,
                x="Tumor Size",
                nbins=30,
                color_discrete_sequence=[ACCENT],
                marginal="box",
                labels={"Tumor Size": "Tumor size (mm)", "count": "Records"},
                hover_data=["Age", "Grade"],
            )
            fig.update_traces(marker_line_color="white", marker_line_width=0.5)
            st.plotly_chart(make_layout(fig, 390), use_container_width=True, config={"displayModeBar": False})

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Age profile</div>', unsafe_allow_html=True)
        if filtered.empty:
            show_empty_chart()
        else:
            fig = px.histogram(
                filtered,
                x="Age",
                nbins=20,
                color_discrete_sequence=["#6e83bd"],
                labels={"Age": "Age (years)", "count": "Records"},
            )
            fig.update_traces(marker_line_color="white", marker_line_width=0.5)
            st.plotly_chart(make_layout(fig, 320), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown('<div class="section-title">Average tumor size by stage and grade</div>', unsafe_allow_html=True)
        if filtered.empty:
            show_empty_chart()
        else:
            means = filtered.groupby(["6th Stage", "Grade"], as_index=False)["Tumor Size"].mean()
            fig = px.line(
                means,
                x="6th Stage",
                y="Tumor Size",
                color="Grade",
                markers=True,
                category_orders={"6th Stage": STAGE_ORDER, "Grade": grade_order(filtered["Grade"].astype(str).unique().tolist())},
                color_discrete_map=GRADE_COLORS,
                labels={"6th Stage": "Clinical stage", "Tumor Size": "Mean tumor size (mm)"},
            )
            st.plotly_chart(make_layout(fig, 320), use_container_width=True, config={"displayModeBar": False})

with stage_tab:
    st.markdown('<div class="section-title">Grade composition within each stage</div>', unsafe_allow_html=True)
    st.caption("Each row is normalized to 100%, making stage groups easier to compare.")
    if filtered.empty:
        show_empty_chart()
    else:
        composition = pd.crosstab(filtered["6th Stage"], filtered["Grade"], normalize="index") * 100
        row_order = [stage for stage in STAGE_ORDER if stage in composition.index.astype(str)]
        row_order += [stage for stage in composition.index.astype(str) if stage not in row_order]
        composition.index = composition.index.astype(str)
        composition = composition.reindex(row_order)
        composition = composition.reindex(columns=grade_order(composition.columns.astype(str).tolist()), fill_value=0)
        heat = px.imshow(
            composition,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale=["#f8edf0", "#dfa0ae", "#a63f5b"],
            labels={"x": "Histologic grade", "y": "Clinical stage", "color": "Share (%)"},
            zmin=0,
            zmax=100,
        )
        st.plotly_chart(make_layout(heat, 390), use_container_width=True, config={"displayModeBar": False})

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Tumor size by grade</div>', unsafe_allow_html=True)
        if filtered.empty:
            show_empty_chart()
        else:
            fig = px.box(
                filtered,
                x="Grade",
                y="Tumor Size",
                color="Grade",
                category_orders={"Grade": grade_order(filtered["Grade"].astype(str).unique().tolist())},
                color_discrete_map=GRADE_COLORS,
                points="outliers",
                labels={"Tumor Size": "Tumor size (mm)"},
            )
            st.plotly_chart(make_layout(fig, 350), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown('<div class="section-title">Age across clinical stages</div>', unsafe_allow_html=True)
        if filtered.empty:
            show_empty_chart()
        else:
            fig = px.box(
                filtered,
                x="6th Stage",
                y="Age",
                color="6th Stage",
                category_orders={"6th Stage": STAGE_ORDER},
                color_discrete_sequence=px.colors.qualitative.Pastel,
                points=False,
                labels={"6th Stage": "Clinical stage", "Age": "Age (years)"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(make_layout(fig, 350), use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-title">Age and tumor size</div>', unsafe_allow_html=True)
    scatter_data = filtered.dropna(subset=["Age", "Tumor Size"])
    if scatter_data.empty:
        show_empty_chart()
    else:
        fig = px.scatter(
            scatter_data,
            x="Age",
            y="Tumor Size",
            color="Grade",
            symbol="6th Stage",
            color_discrete_map=GRADE_COLORS,
            opacity=0.68,
            hover_data=["Race", "Marital Status", "N Stage"],
            labels={"Age": "Age (years)", "Tumor Size": "Tumor size (mm)", "6th Stage": "Clinical stage"},
        )
        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
        st.plotly_chart(make_layout(fig, 430), use_container_width=True, config={"displayModeBar": False})
        correlation = scatter_data[["Age", "Tumor Size"]].corr().iloc[0, 1]
        st.caption(f"Pearson correlation between age and tumor size in the filtered records: {format_number(correlation, 2)}.")

with explore_tab:
    st.markdown('<div class="section-title">Cohort characteristics</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("**Race**")
        if filtered.empty:
            show_empty_chart()
        else:
            race_counts = filtered["Race"].value_counts().rename_axis("Race").reset_index(name="Records")
            fig = px.bar(
                race_counts.sort_values("Records"),
                x="Records",
                y="Race",
                orientation="h",
                color_discrete_sequence=["#6e83bd"],
                labels={"Records": "Records"},
                text="Records",
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(make_layout(fig, 280), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown("**Marital status**")
        if filtered.empty:
            show_empty_chart()
        else:
            status_counts = filtered["Marital Status"].value_counts().rename_axis("Marital Status").reset_index(name="Records")
            fig = px.bar(
                status_counts.sort_values("Records"),
                x="Records",
                y="Marital Status",
                orientation="h",
                color_discrete_sequence=[ACCENT],
                labels={"Records": "Records"},
                text="Records",
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(make_layout(fig, 280), use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-title">Filtered records</div>', unsafe_allow_html=True)
    st.caption("The table follows the sidebar filters. Use the download button to export this view.")
    st.download_button(
        "Download filtered records",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="breast_cancer_filtered_records.csv",
        mime="text/csv",
        disabled=filtered.empty,
    )
    st.dataframe(filtered, use_container_width=True, hide_index=True, height=430)

with quality_tab:
    st.markdown('<div class="section-title">Data quality and cleaning</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note">The notebook reports T Stage as completely empty, removes it, then removes exact duplicate rows. '
        'This app applies the same steps when the matching file is loaded.</div>',
        unsafe_allow_html=True,
    )
    null_summary = raw_data.copy()
    null_summary.columns = [str(column).strip() for column in null_summary.columns]
    quality = pd.DataFrame(
        {
            "Column": null_summary.columns,
            "Missing records": [int(null_summary[column].isna().sum()) for column in null_summary.columns],
            "Missing (%)": [
                (float(null_summary[column].isna().mean() * 100) if len(null_summary) else 0)
                for column in null_summary.columns
            ],
            "Unique values": [int(null_summary[column].nunique(dropna=True)) for column in null_summary.columns],
        }
    ).sort_values(["Missing (%)", "Column"], ascending=[False, True])
    st.dataframe(
        quality.style.format({"Missing (%)": "{:.1f}%"}),
        use_container_width=True,
        hide_index=True,
    )
    q1, q2, q3 = st.columns(3)
    q1.metric("Source records", f"{len(raw_data):,}")
    q2.metric("Exact duplicate rows", f"{duplicate_count:,}")
    q3.metric("Rows in analysis", f"{len(data):,}")
    blank_columns = [column for column in raw_data.columns if raw_data[column].isna().all()]
    if blank_columns:
        st.caption("Fully empty columns excluded from analysis: " + ", ".join(map(str, blank_columns)))

st.markdown(
    '<div class="footer">Exploratory descriptive analysis · Cohort patterns describe this dataset only and are not population estimates or clinical guidance.</div>',
    unsafe_allow_html=True,
)
