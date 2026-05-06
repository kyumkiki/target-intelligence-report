import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import io
import json
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
    "Melanoma": "TCGA-SKCM",
}

IMMUNE_MARKERS = ["CD8A", "CD4", "FOXP3", "PDCD1", "CD274", "LAG3", "TIGIT"]


# =====================================================
# GDC / TCGA DATA FUNCTIONS
# =====================================================

def make_gdc_file_filter(project_id, sample_types):
    return {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "in", "content": {"field": "files.data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "files.analysis.workflow_type", "value": ["STAR - Counts"]}},
            {"op": "in", "content": {"field": "cases.samples.sample_type", "value": sample_types}},
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
            "patient_id": case.get("submitter_id"),
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

    gene_col = "gene_name" if "gene_name" in df.columns else None

    expr_col = None
    for col in ["tpm_unstranded", "fpkm_unstranded", "unstranded"]:
        if col in df.columns:
            expr_col = col
            break

    if gene_col is None or expr_col is None:
        return {}

    result = {}

    for gene in genes:
        row = df[df[gene_col].astype(str).str.upper() == gene.upper()]
        if not row.empty:
            try:
                result[gene.upper()] = float(row.iloc[0][expr_col])
            except Exception:
                result[gene.upper()] = np.nan

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

    for _, row in all_files.iterrows():
        text = download_gdc_file(row["file_id"])
        gene_values = extract_gene_values(text, genes_to_extract)

        target_value = gene_values.get(target_gene.upper())

        if target_value is not None and not pd.isna(target_value):
            data_row = {
                "Patient": row["patient_id"],
                "Group": "Tumor" if row["sample_type"] == "Primary Tumor" else "Normal",
                "Sample_Type": row["sample_type"],
                "Expression": target_value,
            }

            for marker in IMMUNE_MARKERS:
                data_row[marker] = gene_values.get(marker.upper(), np.nan)

            rows.append(data_row)

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
def get_tcga_clinical(project_id, max_cases=500):
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
            time = days_to_death
        else:
            time = days_to_last_follow_up

        if time is None:
            continue

        event = 1 if str(vital_status).lower() == "dead" else 0

        rows.append({
            "Patient": h.get("submitter_id"),
            "OS_time": float(time),
            "OS_event": event,
            "Vital_Status": vital_status,
            "Tumor_Stage": tumor_stage,
        })

    return pd.DataFrame(rows)


# =====================================================
# ANALYSIS FUNCTIONS
# =====================================================

def expression_summary(df):
    tumor_df = df[df["Group"] == "Tumor"]
    normal_df = df[df["Group"] == "Normal"]

    tumor_mean = tumor_df["Expression"].mean() if len(tumor_df) > 0 else np.nan
    normal_mean = normal_df["Expression"].mean() if len(normal_df) > 0 else np.nan

    if pd.notna(normal_mean) and normal_mean != 0:
        fc = tumor_mean / normal_mean
    else:
        fc = np.nan

    if pd.isna(fc):
        expression_score = 1
    elif fc >= 3:
        expression_score = 2
    elif fc >= 1.5:
        expression_score = 1
    else:
        expression_score = 0

    return normal_mean, tumor_mean, fc, expression_score


def immune_correlation(df):
    tumor_df = df[df["Group"] == "Tumor"].copy()
    rows = []

    for marker in IMMUNE_MARKERS:
        if marker in tumor_df.columns:
            temp = tumor_df[["Expression", marker]].dropna()
            if len(temp) >= 3:
                corr = temp["Expression"].corr(temp[marker])
            else:
                corr = np.nan
            rows.append({"Immune_marker": marker, "Correlation_with_target": corr})

    return pd.DataFrame(rows)


def make_survival_groups(expr_df, clinical_df):
    merged = pd.merge(expr_df, clinical_df, on="Patient", how="inner")
    merged = merged[merged["Group"] == "Tumor"].copy()

    if merged.empty:
        return pd.DataFrame()

    median_expr = merged["Expression"].median()
    merged["Target_Group"] = np.where(merged["Expression"] >= median_expr, "High", "Low")

    return merged


def km_curve(df, group_name):
    temp = df[df["Target_Group"] == group_name].copy()
    temp = temp.sort_values("OS_time")

    if temp.empty:
        return pd.DataFrame()

    times = []
    survivals = []

    survival = 1.0
    event_times = sorted(temp["OS_time"].unique())

    for t in event_times:
        at_risk = len(temp[temp["OS_time"] >= t])
        events = len(temp[(temp["OS_time"] == t) & (temp["OS_event"] == 1)])

        if at_risk > 0:
            survival *= (1 - events / at_risk)

        times.append(t)
        survivals.append(survival)

    return pd.DataFrame({
        "Time": times,
        "Survival": survivals,
        "Group": group_name,
    })


def survival_score(survival_df):
    if survival_df.empty:
        return 0, "Not available"

    high = survival_df[survival_df["Target_Group"] == "High"]
    low = survival_df[survival_df["Target_Group"] == "Low"]

    if high.empty or low.empty:
        return 0, "Not available"

    high_event_rate = high["OS_event"].mean()
    low_event_rate = low["OS_event"].mean()

    if high_event_rate > low_event_rate * 1.3:
        return 2, "High target expression may be associated with poorer survival"
    elif high_event_rate > low_event_rate:
        return 1, "High target expression shows a weak poor-survival trend"
    else:
        return 0, "No clear poor-survival trend"


def calculate_adc_score(expression_score, membrane, internalization, clinical_relevance, safety, competition_modifier):
    raw_total = expression_score + membrane + internalization + clinical_relevance + safety + competition_modifier
    total = max(0, min(10, raw_total))

    if total >= 8:
        level = "High ADC/ApDC suitability"
    elif total >= 5:
        level = "Moderate ADC/ApDC suitability"
    else:
        level = "Low ADC/ApDC suitability"

    return total, level


def calculate_market_score(unmet, patient, competition_gap, biomarker, licensing):
    total = unmet + patient + competition_gap + biomarker + licensing

    if total >= 8:
        level = "High marketability"
    elif total >= 5:
        level = "Moderate marketability"
    else:
        level = "Low marketability"

    return total, level


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
    survival_text,
    adc_score,
    adc_level,
    market_score,
    market_level,
):
    normal_text = "Not available" if pd.isna(normal_mean) else f"{normal_mean:.3f}"
    fc_text = "Not available" if pd.isna(fc) else f"{fc:.3f}"

    return f"""
Target Intelligence Report

Target: {target}
Cancer: {cancer}
TCGA Project: {project_id}

1. Expression
- Data source: TCGA/GDC RNA-seq STAR Counts
- Tumor mean expression: {tumor_mean:.3f}
- Normal mean expression: {normal_text}
- Tumor/Normal fold-change: {fc_text}

Meaning:
Expression is the first evidence layer.
A good ADC/ApDC target should show sufficient tumor expression and, ideally, tumor-selective expression compared with normal tissue.

2. Survival
- Result: {survival_text}

Meaning:
Survival association is used as a disease relevance indicator.
If high target expression is associated with poor survival, the target may be biologically linked to aggressive disease.

3. Recurrence
- GEO recurrence module is not fully automated yet.
- GEO metadata can be checked in the GEO tab.

Meaning:
Recurrence association supports clinical relevance and patient stratification potential.

4. Metastasis
- GEO metastasis module is not fully automated yet.
- Primary/metastasis labels should be curated from GEO metadata.

Meaning:
Metastasis association supports target relevance for advanced or recurrent cancer.

5. Immune
- Immune marker correlation is calculated using target expression and immune-related marker genes.

Meaning:
Immune association helps understand whether the target is linked to the tumor microenvironment.

6. ADC/ApDC Suitability
- Score: {adc_score}/10
- Interpretation: {adc_level}

Meaning:
This score reflects tumor expression, membrane localization, internalization evidence, clinical relevance, safety, and competition.

7. Competitors
- Competitor landscape is currently expert-curated.
- Future versions may connect PubMed, ClinicalTrials.gov, and company pipeline data.

Meaning:
A target with high biological value but excessive competition may have lower business attractiveness.

8. Marketability
- Score: {market_score}/10
- Interpretation: {market_level}

Meaning:
Marketability reflects unmet need, patient population, competition gap, biomarker strategy, and licensing potential.

Overall Interpretation:
This report combines public cancer data with expert-defined decision criteria.
The key value is not only data visualization, but the conversion of biological and business evidence into a structured Target Intelligence Score.
"""


# =====================================================
# UI
# =====================================================

st.title("Target Intelligence Report")
st.write("TCGA/GEO 기반 타겟-암종 분석 및 전문가 기준 점수화 리포트")

st.sidebar.header("Input")

target = st.sidebar.text_input("Target gene", "ALCAM")
cancer = st.sidebar.selectbox("Cancer type", list(TCGA_PROJECTS.keys()))
project_id = TCGA_PROJECTS[cancer]

st.sidebar.write(f"TCGA Project: `{project_id}`")

tumor_n = st.sidebar.slider("Tumor sample number", 1, 30, 10)
normal_n = st.sidebar.slider("Normal sample number", 0, 20, 0)

st.sidebar.markdown("---")
st.sidebar.subheader("GEO Validation")
gse_id = st.sidebar.text_input("GEO GSE ID", "GSE26712")

st.sidebar.markdown("---")
st.sidebar.subheader("ADC/ApDC Expert Scoring")

membrane = st.sidebar.slider("Membrane localization", 0, 2, 2)
internalization = st.sidebar.slider("Internalization evidence", 0, 2, 1)
manual_clinical = st.sidebar.slider("Additional clinical relevance", 0, 2, 1)
safety = st.sidebar.slider("Normal tissue safety", 0, 2, 1)
competition_modifier = st.sidebar.slider("Competition modifier", -1, 1, 0)

st.sidebar.markdown("---")
st.sidebar.subheader("Marketability Scoring")

unmet = st.sidebar.slider("Unmet medical need", 0, 2, 2)
patient = st.sidebar.slider("Patient population", 0, 2, 1)
competition_gap = st.sidebar.slider("Competition gap", 0, 2, 1)
biomarker = st.sidebar.slider("Biomarker strategy", 0, 2, 2)
licensing = st.sidebar.slider("Licensing potential", 0, 2, 1)


if st.sidebar.button("Generate Full Report"):

    with st.spinner("TCGA/GDC expression 데이터를 불러오는 중입니다."):
        expr_df = get_tcga_expression(project_id, target, tumor_n, normal_n)

    if expr_df.empty:
        st.error("TCGA expression 데이터를 불러오지 못했습니다. Target gene, 암종, sample 수를 확인하세요.")
        st.stop()

    normal_mean, tumor_mean, fc, expression_score = expression_summary(expr_df)

    with st.spinner("TCGA clinical survival 데이터를 불러오는 중입니다."):
        clinical_df = get_tcga_clinical(project_id)

    survival_df = make_survival_groups(expr_df, clinical_df)
    survival_relevance_score, survival_text = survival_score(survival_df)

    clinical_relevance = max(manual_clinical, survival_relevance_score)

    immune_df = immune_correlation(expr_df)

    adc_score, adc_level = calculate_adc_score(
        expression_score,
        membrane,
        internalization,
        clinical_relevance,
        safety,
        competition_modifier,
    )

    market_score, market_level = calculate_market_score(
        unmet,
        patient,
        competition_gap,
        biomarker,
        licensing,
    )

    st.header(f"{target} in {cancer}")
    st.caption(f"Data source: TCGA/GDC project {project_id}")

    tabs = st.tabs([
        "Expression",
        "Survival",
        "Immune",
        "GEO",
        "ADC/ApDC Score",
        "Marketability",
        "Scoring Criteria",
        "Report",
    ])

    with tabs[0]:
        st.subheader("1. TCGA Expression")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Tumor mean", f"{tumor_mean:.3f}")
        col2.metric("Normal mean", "NA" if pd.isna(normal_mean) else f"{normal_mean:.3f}")
        col3.metric("Tumor/Normal FC", "NA" if pd.isna(fc) else f"{fc:.3f}")
        col4.metric("Expression Score", f"{expression_score}/2")

        st.dataframe(expr_df)

        fig, ax = plt.subplots()

        if expr_df["Group"].nunique() > 1:
            plot_df = expr_df.groupby("Group")["Expression"].mean().reset_index()
            ax.bar(plot_df["Group"], plot_df["Expression"])
            ax.set_ylabel("Mean Expression")
        else:
            ax.bar(expr_df["Patient"], expr_df["Expression"])
            ax.set_ylabel("Expression")
            plt.xticks(rotation=90)

        ax.set_title(f"{target} expression in {project_id}")
        st.pyplot(fig)

    with tabs[1]:
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

            st.metric("Survival Relevance Score", f"{survival_relevance_score}/2")
            st.write(survival_text)

    with tabs[2]:
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

    with tabs[3]:
        st.subheader("4. GEO Metadata")

        st.write("GEO는 데이터셋마다 annotation 구조가 달라 metadata 확인 후 recurrence/metastasis 그룹을 지정해야 합니다.")

        if st.button("Load GEO Metadata"):
            try:
                with st.spinner("GEO metadata를 불러오는 중입니다."):
                    geo_df = get_geo_info(gse_id)
                st.success(f"{gse_id} loaded")
                st.dataframe(geo_df)
            except Exception as e:
                st.error(f"GEO 데이터를 불러오지 못했습니다: {e}")

    with tabs[4]:
        st.subheader("5. ADC/ApDC Suitability Score")

        st.metric("ADC/ApDC Suitability Score", f"{adc_score}/10")
        st.write(adc_level)

        score_df = pd.DataFrame({
            "Component": [
                "Tumor expression evidence",
                "Membrane localization",
                "Internalization evidence",
                "Clinical relevance",
                "Normal tissue safety",
                "Competition modifier",
            ],
            "Score": [
                expression_score,
                membrane,
                internalization,
                clinical_relevance,
                safety,
                competition_modifier,
            ],
            "Meaning": [
                "TCGA tumor expression and tumor/normal selectivity",
                "Whether the target is located on the cell surface",
                "Whether target binding can lead to receptor-mediated internalization",
                "Survival, recurrence, metastasis, or disease relevance",
                "Lower normal tissue risk gives a higher score",
                "High competition can reduce attractiveness; white space can increase it",
            ],
        })

        st.dataframe(score_df)

    with tabs[5]:
        st.subheader("6. Marketability Score")

        st.metric("Marketability Score", f"{market_score}/10")
        st.write(market_level)

        market_df = pd.DataFrame({
            "Component": [
                "Unmet medical need",
                "Patient population",
                "Competition gap",
                "Biomarker strategy",
                "Licensing potential",
            ],
            "Score": [
                unmet,
                patient,
                competition_gap,
                biomarker,
                licensing,
            ],
            "Meaning": [
                "Treatment limitations and medical need",
                "Market size and target patient population",
                "Differentiation space compared with existing drugs",
                "Feasibility of target-positive patient selection",
                "Potential attractiveness for pharma licensing",
            ],
        })

        st.dataframe(market_df)

    with tabs[6]:
        st.subheader("7. Scoring Criteria")

        st.markdown("""
### Expression Score

| 기준 | 점수 | 의미 |
|---|---:|---|
| Tumor/Normal FC ≥ 3 | 2 | 종양 선택성이 높음 |
| Tumor/Normal FC 1.5–3 | 1 | 종양 증가 경향 |
| Tumor/Normal FC < 1.5 | 0 | 종양 선택성 낮음 |
| Normal sample 없음 | 1 | Tumor expression만 인정, 선택성 판단 보류 |

### Clinical Relevance Score

| 기준 | 점수 | 의미 |
|---|---:|---|
| High expression group의 event rate가 뚜렷하게 높음 | 2 | 질병 악성도와 관련 가능성 높음 |
| High expression group의 event rate가 약간 높음 | 1 | 약한 임상 관련성 |
| 차이 없음 | 0 | 임상 관련성 불명확 |

### ADC/ApDC Suitability Score

| 항목 | 점수 |
|---|---:|
| Tumor expression evidence | 0–2 |
| Membrane localization | 0–2 |
| Internalization evidence | 0–2 |
| Clinical relevance | 0–2 |
| Normal tissue safety | 0–2 |
| Competition modifier | -1–1 |

판정 기준:

| 총점 | 판정 |
|---:|---|
| 8–10 | High suitability |
| 5–7 | Moderate suitability |
| 0–4 | Low suitability |

### Marketability Score

| 항목 | 점수 |
|---|---:|
| Unmet medical need | 0–2 |
| Patient population | 0–2 |
| Competition gap | 0–2 |
| Biomarker strategy | 0–2 |
| Licensing potential | 0–2 |

판정 기준:

| 총점 | 판정 |
|---:|---|
| 8–10 | High marketability |
| 5–7 | Moderate marketability |
| 0–4 | Low marketability |
""")

    with tabs[7]:
        st.subheader("8. Generated Report")

        report = make_report(
            target,
            cancer,
            project_id,
            tumor_mean,
            normal_mean,
            fc,
            survival_text,
            adc_score,
            adc_level,
            market_score,
            market_level,
        )

        st.text_area("Report", report, height=700)

        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"{target}_{project_id}_Target_Intelligence_Report.txt",
            mime="text/plain",
        )

else:
    st.info("왼쪽에서 Target gene과 Cancer type을 선택한 뒤 Generate Full Report를 누르세요.")
