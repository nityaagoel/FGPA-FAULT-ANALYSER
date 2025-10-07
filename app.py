import streamlit as st
import pandas as pd
from analyzer.fault_parser import parse_outputs
from analyzer.fault_metrics import compute_metrics
from analyzer.fault_visualizer import plot_bit_errors, plot_fault_distribution
from analyzer.fault_classifier import classify_faults

st.set_page_config(page_title="FPGA Fault Analyzer Dashboard", layout="wide")

st.title("🔍 FPGA Fault Analyzer Dashboard")
st.write("Upload your **golden** and **faulty** simulation outputs to analyze fault coverage and mismatches.")

col1, col2 = st.columns(2)
golden_file = col1.file_uploader("Upload Golden Output (CSV)", type=["csv"])
faulty_file = col2.file_uploader("Upload Faulty Output (CSV)", type=["csv"])

if golden_file and faulty_file:
    st.divider()
    st.subheader("📊 Analysis Report")

    df = parse_outputs(golden_file, faulty_file)
    metrics = compute_metrics(df)

    # Display metrics
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Total Test Cases", metrics["Total Test Cases"])
    colB.metric("Mismatches", metrics["Mismatches Detected"])
    colC.metric("Fault Coverage (%)", metrics["Fault Coverage (%)"])
    colD.metric("Avg Bit Error", metrics["Average Bit Error"])

    st.subheader("🔬 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📈 Visualizations")
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_bit_errors(df))
    with c2:
        st.pyplot(plot_fault_distribution(df))

    # Optional ML report
    with st.expander("🤖 Fault Severity Classification Report"):
        report = classify_faults(df)
        st.json(report)
else:
    st.info("Please upload both files to start analysis.")
