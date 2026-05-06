import streamlit as st

st.set_page_config(
    page_title="Target Intelligence Report",
    layout="wide"
)

st.title("Target Intelligence Report")

st.write("""
This app generates a target-cancer intelligence report using public datasets such as TCGA and GEO.
""")

st.sidebar.header("Input")

target = st.sidebar.text_input("Target gene", "ALCAM")
cancer = st.sidebar.selectbox(
    "Cancer type",
    ["Ovarian Cancer", "Breast Cancer", "Pancreatic Cancer", "Lung Cancer", "Colorectal Cancer"]
)

if st.sidebar.button("Generate Report"):
    st.header(f"{target} in {cancer}")

    st.subheader("1. Expression")
    st.write("Tumor vs normal expression analysis will be shown here.")

    st.subheader("2. Survival")
    st.write("Overall survival and disease-free survival analysis will be shown here.")

    st.subheader("3. Recurrence")
    st.write("Recurrence-associated evidence from TCGA/GEO will be shown here.")

    st.subheader("4. Metastasis")
    st.write("Metastasis-associated evidence will be shown here.")

    st.subheader("5. Immune")
    st.write("Immune microenvironment association will be shown here.")

    st.subheader("6. ADC Suitability")
    st.write("ADC/ApDC target suitability score will be shown here.")

    st.subheader("7. Competitors")
    st.write("Known drugs, antibodies, ADCs, and competing programs will be shown here.")

    st.subheader("8. Marketability")
    st.write("Market size, unmet need, and licensing potential will be shown here.")

    st.success("First prototype report generated.")
else:
    st.info("Enter a target and cancer type, then click Generate Report.")
