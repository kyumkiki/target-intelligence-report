import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import io
import GEOparse


st.set_page_config(
    page_title="Target Intelligence Report",
    layout="wide"
)

GDC_API = "https://api.gdc.cancer.gov"


TCGA_PROJECTS = {
    "Ovarian Cancer": "TCGA-OV",
    "Breast Cancer": "TCGA-BRCA",
    "Pancreatic Cancer": "TCGA-PAAD",
    "Lung Adenocarcinoma": "TCGA-LUAD",
    "Lung Squamous Cell Carcinoma": "TCGA-LUSC",
    "Colorectal Cancer": "TCGA-COAD",
    "Stomach Cancer": "TCGA-STAD",
    "Liver Cancer": "TCGA-LIHC",
    "Glioblastoma": "TCGA-GBM",
    "Melanoma": "TCGA-SKCM",
}


def gdc_filter(project_id, sample_types):
    return {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [project_id]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_type",
                    "value": ["Gene Expression Quantification"]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "files.analysis.workflow_type",
                    "value": ["STAR - Counts"]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "cases.samples.sample_type",
                    "value": sample_types
                }
            }
        ]
    }


@st.cache_data(show_spinner=False)
def get_gdc_files(project_id, sample_types, max_files=30):
    fields = [
        "file_id",
        "file_name",
        "cases.case_id",
        "cases.submitter_id",
        "cases.samples.sample_type"
    ]

    params = {
        "filters": gdc_filter(project_id, sample_types),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": str(max_files)
    }

    r = requests.get(f"{GDC_API}/files", params=params, timeout=60)
    r.raise_for_status()

    hits = r.json()["data"]["hits"]
    rows = []

    for h in hits:
        case = h.get("cases", [{}])[0]
        sample = case.get("samples", [{}])[0]

        rows.append({
            "file_id": h["file_id"],
            "file_name": h["file_name"],
            "case_id": case.get("case_id"),
            "patient_id": case.get("submitter_id"),
            "sample_type": sample.get("sample_type")
        })

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def download_gdc_file(file_id):
    url = f"{GDC_API}/data/{file_id}"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.text


def extract_gene_expression(file_text, target_gene):
    df = pd.read_csv(io.StringIO(file_text), sep="\t", comment="#")

    possible_gene_cols = ["gene_name", "gene_id"]
    possible_expr_cols = [
        "tpm_unstranded",
        "fpkm_unstranded",
        "unstranded"
    ]

    gene_col = None
    for col in possible_gene_cols:
        if col in df.columns:
            gene_col = col
            break

    expr_col = None
    for col in possible_expr_cols:
        if col in df.columns:
            expr_col = col
            break

    if gene_col is None or expr_col is None:
        return None

    gene_row = df[df[gene_col].astype(str).str.upper() == target_gene.upper()]

    if gene_row.empty:
        return None

    value = gene_row.iloc[0][expr_col]

    try:
        return float(value)
    except:
        return None


@st.cache_data(show_spinner=False)
def get_tcga_expression(project_id, target_gene, tumor_n=30, normal_n=10):
    tumor_files = get_gdc_files(
        project_id,
        ["Primary Tumor"],
        max_files=tumor_n
    )

    normal_files = get_gdc_files(
        project_id,
        ["Solid Tissue Normal"],
        max_files=normal_n
    )

    all_files = pd.concat([tumor_files, normal_files], ignore_index=True)

    rows = []

    for _, row in all_files.iterrows():
        try:
            text = download_gdc_file(row["file_id"])
            expr = extract_gene_expression(text, target_gene)

            if expr is not None:
                rows.append({
                    "Patient": row["patient_id"],
                    "Group": "Tumor" if row["sample_type"] == "Primary Tumor" else "Normal",
                    "Sample_Type": row["sample_type"],
                    "Expression": expr
                })

        except Exception:
            continue

    return pd.DataFrame(rows)


def expression_summary(df):
    normal_df = df[df["Group"] == "Normal"]
    tumor_df = df[df["Group"] == "Tumor"]

    normal_mean = normal_df["Expression"].mean() if len(normal_df) > 0 else np.nan
    tumor_mean = tumor_df["Expression"].mean() if len(tumor_df) > 0 else np.nan

    if pd.notna(normal_mean) and normal_mean != 0:
        fc = tumor_mean / normal_mean
    else:
        fc = np.nan

    if pd.isna(fc):
        score = 1
    elif fc >= 3:
        score = 2
    elif fc >= 1.5:
        score = 1
    else:
        score = 0

    return normal_mean, tumor_mean, fc, score


@st.cache_data(show_spinner=False)
def get_geo_info(gse_id):
    gse = GEOparse.get_GEO(geo=gse_id, destdir="./geo_cache", silent=True)
    samples = []

    for gsm_name, gsm in gse.gsms.items():
        samples.append({
            "GSM": gsm_name,
            "Title": gsm.metadata.get("title", [""])[0],
            "Source": gsm.metadata.get("source_name_ch1", [""])[0],
            "Characteristics": "; ".join(gsm.metadata.get("characteristics_ch1", []))
        })

    return pd.DataFrame(samples)


def calculate_adc_score(expression_score, membrane, internalization, clinical, safety):
    total = expression_score + membrane + internalization + clinical + safety

    if total >= 8:
        level = "High ADC/ApDC suitability"
    elif total >= 5:
        level = "Moderate ADC/ApDC suitability"
    else:
        level = "Low ADC/ApDC suitability"

    return total, level


def calculate_market_score(unmet, patient, competition, biomarker, licensing):
    total = unmet + patient + competition + biomarker + licensing

    if total >= 8:
        level = "High marketability"
    elif total >= 5:
        level = "Moderate marketability"
    else:
        level = "Low marketability"

    return total, level


def make_report(target, cancer, project_id, normal_mean, tumor_mean, fc, adc_score, adc_level, market_score, market_level):
    return f"""
Target Intelligence Report

Target: {target}
Cancer: {cancer}
TCGA Project: {project_id}

1. Expression
- TCGA/GDC RNA-seq STAR Counts data was queried.
- Normal mean expression: {normal_mean:.3f}
- Tumor mean expression: {tumor_mean:.3f}
- Tumor/Normal fold-change: {fc:.3f}

2. Survival
- Survival module will be connected to TCGA clinical data in the next version.

3. Recurrence
- Recurrence module will be connected using GEO datasets with recurrence annotations.

4. Metastasis
- Metastasis module will be connected using GEO datasets with primary/metastasis annotations.

5. Immune
- Immune correlation module will be added using TCGA immune signature genes.

6. ADC/ApDC Suitability
- Score: {adc_score}/10
- Interpretation: {adc_level}

7. Competitors
- Competitor landscape should be manually curated first.
- Later version can connect ClinicalTrials.gov, PubMed, and company pipelines.

8. Marketability
- Score: {market_score}/10
- Interpretation: {market_level}

Summary
This report is generated from live TCGA/GDC data query.
GEO is included as an external validation data source.
"""


st.title("Target Intelligence Report")
st.write("실제 TCGA/GDC 데이터를 불러와 타겟-암종 발현 분석을 수행하는 MVP입니다.")

st.sidebar.header("Input")

target = st.sidebar.text_input("Target gene", "ALCAM")

cancer = st.sidebar.selectbox(
    "Cancer type",
    list(TCGA_PROJECTS.keys())
)

project_id = TCGA_PROJECTS[cancer]

st.sidebar.write(f"TCGA Project: `{project_id}`")

tumor_n = st.sidebar.slider("Tumor sample number", 5, 80, 30)
normal_n = st.sidebar.slider("Normal sample number", 0, 30, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("GEO validation")

gse_id = st.sidebar.text_input("GEO GSE ID", "GSE26712")

st.sidebar.markdown("---")
st.sidebar.subheader("ADC/ApDC Scoring")

membrane = st.sidebar.slider("Membrane localization", 0, 2, 2)
internalization = st.sidebar.slider("Internalization evidence", 0, 2, 1)
clinical = st.sidebar.slider("Clinical relevance", 0, 2, 1)
safety = st.sidebar.slider("Normal tissue safety", 0, 2, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Marketability Scoring")

unmet = st.sidebar.slider("Unmet medical need", 0, 2, 2)
patient = st.sidebar.slider("Patient population", 0, 2, 1)
competition = st.sidebar.slider("Competition gap", 0, 2, 1)
biomarker = st.sidebar.slider("Biomarker strategy", 0, 2, 2)
licensing = st.sidebar.slider("Licensing potential", 0, 2, 1)


if st.sidebar.button("Generate TCGA Report"):

    with st.spinner("TCGA/GDC에서 실제 RNA-seq 데이터를 불러오는 중입니다."):
        expr_df = get_tcga_expression(project_id, target, tumor_n, normal_n)

    if expr_df.empty:
        st.error("데이터를 불러오지 못했습니다. Target gene 이름 또는 암종을 확인하세요.")
        st.stop()

    normal_mean, tumor_mean, fc, expression_score = expression_summary(expr_df)

    adc_score, adc_level = calculate_adc_score(
        expression_score,
        membrane,
        internalization,
        clinical,
        safety
    )

    market_score, market_level = calculate_market_score(
        unmet,
        patient,
        competition,
        biomarker,
        licensing
    )

    st.header(f"{target} in {cancer}")
    st.caption(f"Data source: TCGA/GDC project {project_id}")

    tabs = st.tabs([
        "Expression",
        "GEO",
        "ADC Suitability",
        "Marketability",
        "Report"
    ])

    with tabs[0]:
        st.subheader("1. TCGA Expression")

        col1, col2, col3 = st.columns(3)

        col1.metric("Normal mean", "NA" if pd.isna(normal_mean) else f"{normal_mean:.3f}")
        col2.metric("Tumor mean", "NA" if pd.isna(tumor_mean) else f"{tumor_mean:.3f}")
        col3.metric("Tumor/Normal FC", "NA" if pd.isna(fc) else f"{fc:.3f}")

        st.dataframe(expr_df)

        plot_df = expr_df.groupby("Group")["Expression"].mean().reset_index()

        fig, ax = plt.subplots()
        ax.bar(plot_df["Group"], plot_df["Expression"])
        ax.set_ylabel("Expression")
        ax.set_title(f"{target} expression in {project_id}")
        st.pyplot(fig)

    with tabs[1]:
        st.subheader("2. GEO Validation Metadata")

        if st.button("Load GEO info"):
            try:
                with st.spinner("GEO 데이터를 불러오는 중입니다."):
                    geo_df = get_geo_info(gse_id)

                st.success(f"{gse_id} loaded")
                st.dataframe(geo_df)

                st.write("""
GEO는 dataset마다 recurrence, metastasis, treatment response annotation 형식이 다릅니다.
따라서 다음 단계에서는 이 metadata를 보고 Recurrence / Non-recurrence 또는 Primary / Metastasis 그룹을 지정하는 기능을 추가합니다.
""")

            except Exception as e:
                st.error(f"GEO 데이터를 불러오지 못했습니다: {e}")

    with tabs[2]:
        st.subheader("3. ADC/ApDC Suitability")

        st.metric("ADC/ApDC Suitability Score", f"{adc_score}/10")
        st.write(adc_level)

        st.table(pd.DataFrame({
            "Component": [
                "Tumor-enriched expression",
                "Membrane localization",
                "Internalization evidence",
                "Clinical relevance",
                "Normal tissue safety"
            ],
            "Score": [
                expression_score,
                membrane,
                internalization,
                clinical,
                safety
            ]
        }))

    with tabs[3]:
        st.subheader("4. Marketability")

        st.metric("Marketability Score", f"{market_score}/10")
        st.write(market_level)

        st.table(pd.DataFrame({
            "Component": [
                "Unmet medical need",
                "Patient population",
                "Competition gap",
                "Biomarker strategy",
                "Licensing potential"
            ],
            "Score": [
                unmet,
                patient,
                competition,
                biomarker,
                licensing
            ]
        }))

    with tabs[4]:
        st.subheader("5. Generated Report")

        report = make_report(
            target,
            cancer,
            project_id,
            normal_mean,
            tumor_mean,
            fc,
            adc_score,
            adc_level,
            market_score,
            market_level
        )

        st.text_area("Report", report, height=600)

        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"{target}_{project_id}_Target_Intelligence_Report.txt",
            mime="text/plain"
        )

else:
    st.info("왼쪽에서 Target gene과 Cancer type을 선택한 뒤 Generate TCGA Report를 누르세요.")
