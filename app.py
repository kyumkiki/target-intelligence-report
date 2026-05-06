import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import io
import json
import GEOparse


# =====================================================
# PAGE SETUP
# =====================================================

st.set_page_config(
    page_title="Target Intelligence Report",
    layout="wide"
)

GDC_API = "https://api.gdc.cancer.gov"


TCGA_PROJECTS = {
    "Adrenocortical Carcinoma": "TCGA-ACC",
    "Bladder Cancer": "TCGA-BLCA",
    "Breast Cancer": "TCGA-BRCA",
    "Cervical Cancer": "TCGA-CESC",
    "Cholangiocarcinoma": "TCGA-CHOL",
    "Colon Cancer": "TCGA-COAD",
    "Diffuse Large B-cell Lymphoma": "TCGA-DLBC",
    "Esophageal Cancer": "TCGA-ESCA",
    "Glioblastoma": "TCGA-GBM",
    "Head and Neck Cancer": "TCGA-HNSC",
    "Kidney Chromophobe": "TCGA-KICH",
    "Kidney Clear Cell Carcinoma": "TCGA-KIRC",
    "Kidney Papillary Carcinoma": "TCGA-KIRP",
    "Acute Myeloid Leukemia": "TCGA-LAML",
    "Lower Grade Glioma": "TCGA-LGG",
    "Liver Cancer": "TCGA-LIHC",
    "Lung Adenocarcinoma": "TCGA-LUAD",
    "Lung Squamous Cell Carcinoma": "TCGA-LUSC",
    "Mesothelioma": "TCGA-MESO",
    "Ovarian Cancer": "TCGA-OV",
    "Pancreatic Cancer": "TCGA-PAAD",
    "Pheochromocytoma / Paraganglioma": "TCGA-PCPG",
    "Prostate Cancer": "TCGA-PRAD",
    "Rectal Cancer": "TCGA-READ",
    "Sarcoma": "TCGA-SARC",
    "Melanoma": "TCGA-SKCM",
    "Stomach Cancer": "TCGA-STAD",
    "Testicular Germ Cell Tumor": "TCGA-TGCT",
    "Thyroid Cancer": "TCGA-THCA",
    "Thymoma": "TCGA-THYM",
    "Endometrial Cancer": "TCGA-UCEC",
    "Uterine Carcinosarcoma": "TCGA-UCS",
    "Uveal Melanoma": "TCGA-UVM",
}


IMMUNE_MARKERS = [
    "CD8A",
    "CD4",
    "FOXP3",
    "PDCD1",
    "CD274",
    "LAG3",
    "TIGIT",
    "GZMB",
    "IFNG",
]


# =====================================================
# GDC / TCGA FUNCTIONS
# =====================================================

def make_gdc_file_filter(project_id, sample_types):
    return {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [project_id],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_type",
                    "value": ["Gene Expression Quantification"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.analysis.workflow_type",
                    "value": ["STAR - Counts"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "cases.samples.sample_type",
                    "value": sample_types,
                },
            },
        ],
    }


@st.cache_data(show_spinner=False)
def get_gdc_files(project_id, sample_types, max_files=10):
    fields = [
        "file_id",
        "file_name",
        "cases.submitter_id",
        "cases.case_id",
        "cases.samples.sample_type",
    ]

    params = {
        "filters": json.dumps(make_gdc_file_filter(project_id, sample_types)),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": str(max_files),
    }

    r = requests.get(f"{GDC_API}/files", params=params, timeout=60)

    if r.status_code != 200:
        st.error(f"GDC API Error: {r.status_code}")
        st.code(r.text)
        return pd.DataFrame()

    hits = r.json().get("data", {}).get("hits", [])
    rows = []

    for h in hits:
        case = h.get("cases", [{}])[0]
        sample = case.get("samples", [{}])[0]

        rows.append({
            "file_id": h.get("file_id"),
            "file_name": h.get("file_name"),
            "Patient": case.get("submitter_id"),
            "case_id": case.get("case_id"),
            "sample_type": sample.get("sample_type"),
        })

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def download_gdc_file(file_id):
    r = requests.get(f"{GDC_API}/data/{file_id}", timeout=120)

    if r.status_code != 200:
        return None

    return r.text


def extract_gene_values(file_text, genes):
    if file_text is None:
        return {}

    try:
        df = pd.read_csv(io.StringIO(file_text), sep="\t", comment="#")
    except Exception:
        return {}

    if "gene_name" not in df.columns:
        return {}

    expr_col = None

    for col in ["tpm_unstranded", "fpkm_unstranded", "unstranded"]:
        if col in df.columns:
            expr_col = col
            break

    if expr_col is None:
        return {}

    result = {}

    for gene in genes:
        gene_upper = gene.upper()
        row = df[df["gene_name"].astype(str).str.upper() == gene_upper]

        if not row.empty:
            try:
                raw_value = float(row.iloc[0][expr_col])
                log_value = np.log2(raw_value + 1)
                result[gene_upper] = log_value
            except Exception:
                result[gene_upper] = np.nan

    return result


@st.cache_data(show_spinner=False)
def get_tcga_expression(project_id, target_gene, tumor_n=10, normal_n=0):
    tumor_files = get_gdc_files(project_id, ["Primary Tumor"], tumor_n)

    if normal_n > 0:
        normal_files = get_gdc_files(project_id, ["Solid Tissue Normal"], normal_n)
    else:
        normal_files = pd.DataFrame()

    all_files = pd.concat([tumor_files, normal_files], ignore_index=True)

    if all_files.empty:
        return pd.DataFrame()

    genes_to_extract = [target_gene] + IMMUNE_MARKERS
    rows = []

    for _, file_row in all_files.iterrows():
        file_text = download_gdc_file(file_row["file_id"])
        gene_values = extract_gene_values(file_text, genes_to_extract)

        target_value = gene_values.get(target_gene.upper())

        if target_value is None or pd.isna(target_value):
            continue

        row = {
            "Patient": file_row["Patient"],
            "Group": "Tumor" if file_row["sample_type"] == "Primary Tumor" else "Normal",
            "Sample_Type": file_row["sample_type"],
            "Target_Expression": target_value,
        }

        for marker in IMMUNE_MARKERS:
            row[marker] = gene_values.get(marker.upper(), np.nan)

        rows.append(row)

    return pd.DataFrame(rows)


def make_clinical_filter(project_id):
    return {
        "op": "in",
        "content": {
            "field": "project.project_id",
            "value": [project_id],
        },
    }


@st.cache_data(show_spinner=False)
def get_tcga_clinical(project_id, max_cases=1000):
    fields = [
        "submitter_id",
        "diagnoses.vital_status",
        "diagnoses.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.tumor_stage",
    ]

    params = {
        "filters": json.dumps(make_clinical_filter(project_id)),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": str(max_cases),
    }

    r = requests.get(f"{GDC_API}/cases", params=params, timeout=60)

    if r.status_code != 200:
        return pd.DataFrame()

    hits = r.json().get("data", {}).get("hits", [])
    rows = []

    for h in hits:
        diagnosis = h.get("diagnoses", [{}])[0]

        vital_status = diagnosis.get("vital_status")
        days_to_death = diagnosis.get("days_to_death")
        days_to_last_follow_up = diagnosis.get("days_to_last_follow_up")
        tumor_stage = diagnosis.get("tumor_stage")

        if days_to_death is not None:
            os_time = days_to_death
        else:
            os_time = days_to_last_follow_up

        if os_time is None:
            continue

        os_event = 1 if str(vital_status).lower() == "dead" else 0

        rows.append({
            "Patient": h.get("submitter_id"),
            "OS_time": float(os_time),
            "OS_event": os_event,
            "Vital_Status": vital_status,
            "Tumor_Stage": tumor_stage,
        })

    return pd.DataFrame(rows)


# =====================================================
# ANALYSIS FUNCTIONS
# =====================================================

def expression_analysis(expr_df):
    tumor_df = expr_df[expr_df["Group"] == "Tumor"]
    normal_df = expr_df[expr_df["Group"] == "Normal"]

    tumor_mean = tumor_df["Target_Expression"].mean() if len(tumor_df) > 0 else np.nan
    normal_mean = normal_df["Target_Expression"].mean() if len(normal_df) > 0 else np.nan

    if pd.notna(normal_mean) and normal_mean > 0:
        fc = tumor_mean / normal_mean
    else:
        fc = np.nan

    if pd.isna(fc):
        tumor_expression_score = 3 if tumor_mean >= 3 else 2 if tumor_mean >= 1 else 1
        selectivity_score = 2
    else:
        tumor_expression_score = 5 if tumor_mean >= 4 else 4 if tumor_mean >= 3 else 2 if tumor_mean >= 1 else 0
        selectivity_score = 5 if fc >= 3 else 3 if fc >= 1.5 else 1 if fc >= 1.1 else 0

    return tumor_mean, normal_mean, fc, tumor_expression_score, selectivity_score


def make_survival_groups(expr_df, clinical_df, cutoff_method):
    tumor_df = expr_df[expr_df["Group"] == "Tumor"].copy()

    merged = pd.merge(tumor_df, clinical_df, on="Patient", how="inner")

    if merged.empty:
        return pd.DataFrame()

    if cutoff_method == "Median":
        cutoff = merged["Target_Expression"].median()
    elif cutoff_method == "Upper quartile":
        cutoff = merged["Target_Expression"].quantile(0.75)
    else:
        cutoff = merged["Target_Expression"].median()

    merged["Target_Group"] = np.where(
        merged["Target_Expression"] >= cutoff,
        "High",
        "Low",
    )

    return merged


def km_curve(df, group_name):
    temp = df[df["Target_Group"] == group_name].copy()
    temp = temp.sort_values("OS_time")

    if temp.empty:
        return pd.DataFrame()

    survival = 1.0
    times = []
    survivals = []

    for t in sorted(temp["OS_time"].unique()):
        at_risk = len(temp[temp["OS_time"] >= t])
        events = len(temp[(temp["OS_time"] == t) & (temp["OS_event"] == 1)])

        if at_risk > 0:
            survival = survival * (1 - events / at_risk)

        times.append(t)
        survivals.append(survival)

    return pd.DataFrame({
        "Time": times,
        "Survival": survivals,
        "Group": group_name,
    })


def survival_analysis_score(survival_df):
    if survival_df.empty:
        return 0, "Not available"

    high = survival_df[survival_df["Target_Group"] == "High"]
    low = survival_df[survival_df["Target_Group"] == "Low"]

    if high.empty or low.empty:
        return 0, "Not available"

    high_event_rate = high["OS_event"].mean()
    low_event_rate = low["OS_event"].mean()

    if low_event_rate == 0 and high_event_rate > 0:
        score = 5
        text = "High expression group shows higher death-event rate."
    elif high_event_rate >= low_event_rate * 1.5:
        score = 5
        text = "High expression group shows markedly worse survival trend."
    elif high_event_rate > low_event_rate:
        score = 3
        text = "High expression group shows weak poor-survival trend."
    else:
        score = 1
        text = "No clear poor-survival trend."

    return score, text


def immune_correlation(expr_df):
    tumor_df = expr_df[expr_df["Group"] == "Tumor"].copy()
    rows = []

    for marker in IMMUNE_MARKERS:
        if marker not in tumor_df.columns:
            continue

        temp = tumor_df[["Target_Expression", marker]].dropna()

        if len(temp) < 3:
            corr = np.nan
        else:
            corr = temp["Target_Expression"].corr(temp[marker])

        rows.append({
            "Immune_marker": marker,
            "Correlation_with_target": corr,
        })

    return pd.DataFrame(rows)


def immune_score_from_correlation(immune_df):
    if immune_df.empty:
        return 0

    valid = immune_df["Correlation_with_target"].dropna()

    if valid.empty:
        return 0

    max_abs_corr = valid.abs().max()

    if max_abs_corr >= 0.5:
        return 3
    elif max_abs_corr >= 0.3:
        return 2
    elif max_abs_corr >= 0.2:
        return 1
    else:
        return 0


def grade_from_score(score):
    if score >= 85:
        return "A"
    elif score >= 75:
        return "B+"
    elif score >= 65:
        return "B"
    elif score >= 55:
        return "C+"
    elif score >= 45:
        return "C"
    else:
        return "D"


def recommendation_from_grade(grade):
    if grade in ["A", "B+"]:
        return "Proceed to target validation and modality-specific experimental validation."
    elif grade in ["B", "C+"]:
        return "Proceed selectively. Prioritize safety, internalization, and validation datasets."
    elif grade == "C":
        return "Hold or validate only if strategic rationale is strong."
    else:
        return "Not recommended without stronger biological and business evidence."


def calculate_master_score(
    tumor_expression_score,
    selectivity_score,
    survival_score,
    immune_score,
    recurrence_score,
    metastasis_score,
    localization_score,
    internalization_score,
    payload_score,
    biomarker_score,
    safety_score,
    dependency_score,
    competition_score,
    market_score,
):
    biology_score = (
        tumor_expression_score
        + selectivity_score
        + survival_score
        + immune_score
        + recurrence_score
        + metastasis_score
        + dependency_score
    )

    modality_score = (
        localization_score
        + internalization_score
        + payload_score
        + biomarker_score
    )

    final_score = (
        biology_score
        + modality_score
        + safety_score
        + competition_score
        + market_score
    )

    final_score = max(0, min(100, final_score))

    return biology_score, modality_score, final_score


@st.cache_data(show_spinner=False)
def get_geo_info(gse_id):
    gse = GEOparse.get_GEO(geo=gse_id, destdir="./geo_cache", silent=True)

    rows = []

    for gsm_name, gsm in gse.gsms.items():
        rows.append({
            "GSM": gsm_name,
            "Title": gsm.metadata.get("title", [""])[0],
            "Source": gsm.metadata.get("source_name_ch1", [""])[0],
            "Characteristics": "; ".join(gsm.metadata.get("characteristics_ch1", [])),
        })

    return pd.DataFrame(rows)


def make_report(
    target,
    cancer,
    project_id,
    tumor_mean,
    normal_mean,
    fc,
    biology_score,
    modality_score,
    safety_score,
    competition_score,
    market_score,
    final_score,
    grade,
    recommendation,
    survival_text,
):
    normal_text = "Not available" if pd.isna(normal_mean) else f"{normal_mean:.3f}"
    fc_text = "Not available" if pd.isna(fc) else f"{fc:.3f}"

    return f"""
Target Intelligence Report v0.5

Target: {target}
Cancer: {cancer}
TCGA Project: {project_id}

1. Expression
- Data source: TCGA/GDC RNA-seq STAR Counts
- Expression scale: log2(TPM + 1), or fallback to available GDC expression column
- Tumor mean expression: {tumor_mean:.3f}
- Normal mean expression: {normal_text}
- Tumor/Normal ratio: {fc_text}

Interpretation:
Expression is the first evidence layer. For ADC/ApDC development, tumor expression and tumor selectivity are both important.

2. Survival
- Result: {survival_text}

Interpretation:
Survival association is used as a disease relevance indicator. A target linked to poor prognosis may support stronger biological rationale.

3. Immune
- Immune marker correlation is calculated from the same TCGA tumor RNA-seq samples.

Interpretation:
Immune association helps understand whether the target is linked to tumor microenvironment biology.

4. GEO Validation
- GEO metadata can be reviewed for recurrence, metastasis, drug resistance, or treatment response datasets.

Interpretation:
GEO is best used as focused validation, while TCGA is used as broad baseline biology.

5. ADC/ApDC Suitability
- Biology score: {biology_score}/30
- Modality score: {modality_score}/20
- Safety score: {safety_score}/20

Interpretation:
A good ADC/ApDC target should have tumor expression, surface localization, internalization potential, disease relevance, and acceptable normal tissue safety.

6. Competition and Marketability
- Competition score: {competition_score}/10
- Marketability score: {market_score}/20

Interpretation:
A biologically strong target may still be commercially weak if competition is excessive or market opportunity is small.

7. Final Target Intelligence Grade
- Final score: {final_score}/100
- Grade: {grade}
- Recommendation: {recommendation}

Overall Meaning:
This report converts public cancer data and expert-defined development criteria into a structured target intelligence score.
"""


# =====================================================
# UI
# =====================================================

st.title("Target Intelligence Report v0.5")
st.write("어떤 target gene이라도 입력하여 TCGA 기반 타겟 리포트를 생성하는 고급형 MVP")

st.sidebar.header("1. Target Input")

target = st.sidebar.text_input("Target gene symbol", "ALCAM")
cancer = st.sidebar.selectbox("Cancer type", list(TCGA_PROJECTS.keys()))
project_id = TCGA_PROJECTS[cancer]

st.sidebar.write(f"TCGA Project: `{project_id}`")

tumor_n = st.sidebar.slider("Tumor sample number", 3, 50, 10)
normal_n = st.sidebar.slider("TCGA normal sample number", 0, 30, 0)

cutoff_method = st.sidebar.selectbox(
    "Survival cutoff method",
    ["Median", "Upper quartile"],
)

st.sidebar.markdown("---")
st.sidebar.header("2. GEO Validation")

gse_id = st.sidebar.text_input("GEO GSE ID", "GSE26712")

recurrence_score = st.sidebar.slider(
    "Recurrence evidence score",
    0,
    3,
    1,
)

metastasis_score = st.sidebar.slider(
    "Metastasis evidence score",
    0,
    3,
    1,
)

st.sidebar.markdown("---")
st.sidebar.header("3. Modality Suitability")

localization_score = st.sidebar.slider(
    "Cell-surface / membrane localization",
    0,
    5,
    4,
)

internalization_score = st.sidebar.slider(
    "Internalization evidence",
    0,
    5,
    3,
)

payload_score = st.sidebar.slider(
    "Payload compatibility",
    0,
    5,
    3,
)

biomarker_score = st.sidebar.slider(
    "Patient stratification / biomarker feasibility",
    0,
    5,
    3,
)

st.sidebar.markdown("---")
st.sidebar.header("4. Safety / Dependency")

safety_score = st.sidebar.slider(
    "Normal tissue safety score",
    0,
    20,
    10,
)

dependency_score = st.sidebar.slider(
    "Cancer dependency score",
    0,
    5,
    1,
)

st.sidebar.markdown("---")
st.sidebar.header("5. Competition / Market")

competition_score = st.sidebar.slider(
    "Competitive white-space score",
    0,
    10,
    5,
)

market_score = st.sidebar.slider(
    "Marketability score",
    0,
    20,
    12,
)


if st.sidebar.button("Generate Advanced Report"):

    with st.spinner("TCGA/GDC expression 데이터를 불러오는 중입니다."):
        expr_df = get_tcga_expression(project_id, target, tumor_n, normal_n)

    if expr_df.empty:
        st.error(
            "Target expression 데이터를 불러오지 못했습니다. Gene symbol, 암종, sample 수를 확인하세요."
        )
        st.stop()

    tumor_mean, normal_mean, fc, tumor_expression_score, selectivity_score = expression_analysis(expr_df)

    with st.spinner("TCGA clinical survival 데이터를 불러오는 중입니다."):
        clinical_df = get_tcga_clinical(project_id)

    survival_df = make_survival_groups(expr_df, clinical_df, cutoff_method)
    survival_score_value, survival_text = survival_analysis_score(survival_df)

    immune_df = immune_correlation(expr_df)
    immune_score = immune_score_from_correlation(immune_df)

    biology_score, modality_score, final_score = calculate_master_score(
        tumor_expression_score=tumor_expression_score,
        selectivity_score=selectivity_score,
        survival_score=survival_score_value,
        immune_score=immune_score,
        recurrence_score=recurrence_score,
        metastasis_score=metastasis_score,
        localization_score=localization_score,
        internalization_score=internalization_score,
        payload_score=payload_score,
        biomarker_score=biomarker_score,
        safety_score=safety_score,
        dependency_score=dependency_score,
        competition_score=competition_score,
        market_score=market_score,
    )

    grade = grade_from_score(final_score)
    recommendation = recommendation_from_grade(grade)

    st.header(f"{target} in {cancer}")
    st.caption(f"Data source: TCGA/GDC project {project_id}")

    tabs = st.tabs([
        "Executive Summary",
        "Expression",
        "Survival",
        "Immune",
        "GEO",
        "Score Breakdown",
        "Scoring Criteria",
        "Report",
    ])

    with tabs[0]:
        st.subheader("Executive Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Final Score", f"{final_score}/100")
        col2.metric("Grade", grade)
        col3.metric("Biology", f"{biology_score}/30")
        col4.metric("Modality", f"{modality_score}/20")

        st.success(recommendation)

        summary_df = pd.DataFrame({
            "Category": [
                "Biology",
                "Modality suitability",
                "Safety",
                "Competition",
                "Marketability",
            ],
            "Score": [
                biology_score,
                modality_score,
                safety_score,
                competition_score,
                market_score,
            ],
            "Max": [
                30,
                20,
                20,
                10,
                20,
            ],
        })

        st.dataframe(summary_df)

        fig, ax = plt.subplots()
        ax.bar(summary_df["Category"], summary_df["Score"])
        ax.set_ylabel("Score")
        ax.set_title("Target Intelligence Score Breakdown")
        plt.xticks(rotation=30)
        st.pyplot(fig)

    with tabs[1]:
        st.subheader("1. TCGA Expression")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Tumor mean", f"{tumor_mean:.3f}")
        col2.metric("Normal mean", "NA" if pd.isna(normal_mean) else f"{normal_mean:.3f}")
        col3.metric("Tumor/Normal ratio", "NA" if pd.isna(fc) else f"{fc:.3f}")
        col4.metric("Tumor expression score", f"{tumor_expression_score}/5")
        col5.metric("Selectivity score", f"{selectivity_score}/5")

        st.dataframe(expr_df)

        fig, ax = plt.subplots()

        if expr_df["Group"].nunique() > 1:
            plot_df = expr_df.groupby("Group")["Target_Expression"].mean().reset_index()
            ax.bar(plot_df["Group"], plot_df["Target_Expression"])
            ax.set_ylabel("Mean log2 expression")
        else:
            ax.bar(expr_df["Patient"], expr_df["Target_Expression"])
            ax.set_ylabel("log2 expression")
            plt.xticks(rotation=90)

        ax.set_title(f"{target} expression in {project_id}")
        st.pyplot(fig)

    with tabs[2]:
        st.subheader("2. TCGA Survival")

        if survival_df.empty:
            st.warning("Expression 데이터와 clinical survival 데이터가 충분히 매칭되지 않았습니다.")
        else:
            st.dataframe(survival_df)

            km_high = km_curve(survival_df, "High")
            km_low = km_curve(survival_df, "Low")

            fig, ax = plt.subplots()

            if not km_high.empty:
                ax.step(km_high["Time"], km_high["Survival"], where="post", label="High")
            if not km_low.empty:
                ax.step(km_low["Time"], km_low["Survival"], where="post", label="Low")

            ax.set_xlabel("Days")
            ax.set_ylabel("Overall survival probability")
            ax.set_title(f"{target} High vs Low Survival")
            ax.legend()
            st.pyplot(fig)

            st.metric("Survival score", f"{survival_score_value}/5")
            st.write(survival_text)

    with tabs[3]:
        st.subheader("3. Immune Association")

        if immune_df.empty:
            st.warning("Immune marker correlation을 계산할 수 없습니다.")
        else:
            st.dataframe(immune_df)

            fig, ax = plt.subplots()
            ax.bar(immune_df["Immune_marker"], immune_df["Correlation_with_target"])
            ax.set_ylabel("Correlation with target")
            ax.set_title(f"{target} immune marker correlation")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.metric("Immune score", f"{immune_score}/3")

    with tabs[4]:
        st.subheader("4. GEO Metadata")

        st.write(
            "GEO는 recurrence, metastasis, treatment response 등 특정 clinical question 검증용으로 사용합니다."
        )

        if st.button("Load GEO Metadata"):
            try:
                with st.spinner("GEO metadata를 불러오는 중입니다."):
                    geo_df = get_geo_info(gse_id)

                st.success(f"{gse_id} loaded")
                st.dataframe(geo_df)

            except Exception as e:
                st.error(f"GEO 데이터를 불러오지 못했습니다: {e}")

    with tabs[5]:
        st.subheader("5. Score Breakdown")

        score_df = pd.DataFrame({
            "Domain": [
                "Tumor expression",
                "Tumor selectivity",
                "Survival relevance",
                "Immune association",
                "Recurrence evidence",
                "Metastasis evidence",
                "Cancer dependency",
                "Cell surface localization",
                "Internalization",
                "Payload compatibility",
                "Biomarker feasibility",
                "Normal tissue safety",
                "Competitive white space",
                "Marketability",
            ],
            "Score": [
                tumor_expression_score,
                selectivity_score,
                survival_score_value,
                immune_score,
                recurrence_score,
                metastasis_score,
                dependency_score,
                localization_score,
                internalization_score,
                payload_score,
                biomarker_score,
                safety_score,
                competition_score,
                market_score,
            ],
            "Max": [
                5,
                5,
                5,
                3,
                3,
                3,
                5,
                5,
                5,
                5,
                5,
                20,
                10,
                20,
            ],
        })

        st.dataframe(score_df)

    with tabs[6]:
        st.subheader("6. Scoring Criteria")

        st.markdown("""
### Final Target Intelligence Score / 100

| Category | Max score | Meaning |
|---|---:|---|
| Biology | 30 | Expression, selectivity, survival, immune, recurrence, metastasis, dependency |
| Modality suitability | 20 | Surface localization, internalization, payload compatibility, biomarker feasibility |
| Safety | 20 | Normal tissue safety and on-target/off-tumor risk |
| Competition | 10 | Competitive white space |
| Marketability | 20 | Market size, unmet need, licensing potential |

### Grade

| Score | Grade | Recommendation |
|---:|---|---|
| 85–100 | A | Strongly proceed |
| 75–84 | B+ | Proceed to validation |
| 65–74 | B | Selective validation |
| 55–64 | C+ | Validate only with strong rationale |
| 45–54 | C | Hold or deprioritize |
| <45 | D | Not recommended |

### Important interpretation

This score is not a regulatory conclusion.  
It is a structured expert-decision framework for target prioritization.
""")

    with tabs[7]:
        st.subheader("7. Generated Report")

        report = make_report(
            target=target,
            cancer=cancer,
            project_id=project_id,
            tumor_mean=tumor_mean,
            normal_mean=normal_mean,
            fc=fc,
            biology_score=biology_score,
            modality_score=modality_score,
            safety_score=safety_score,
            competition_score=competition_score,
            market_score=market_score,
            final_score=final_score,
            grade=grade,
            recommendation=recommendation,
            survival_text=survival_text,
        )

        st.text_area("Report", report, height=750)

        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"{target}_{project_id}_Target_Intelligence_Report_v05.txt",
            mime="text/plain",
        )

else:
    st.info("왼쪽에서 target gene과 암종을 선택한 뒤 Generate Advanced Report를 누르세요.")
