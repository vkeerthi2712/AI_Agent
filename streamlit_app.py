# streamlit_main.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import requests
import json
import plotly.express as px
from scipy.optimize import curve_fit
from dotenv import load_dotenv
import time

load_dotenv()
FASTAPI_BACKEND_URL = os.environ.get("FASTAPI_BACKEND_URL", "http://127.0.0.1:8000")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

st.set_page_config(page_title="AI Data Analyst (HITL)", layout="centered")
st.title("AI Data Analyst — Human-in-the-Loop")

# Session initialization
for k in ["combined_df", "uploaded_file_name", "uploaded_file_type", "review_id",
          "draft_insights", "draft_chart_spec", "chart_bytes", "approved_flag",
          "exec_summary", "conclusion", "generated_report"]:
    if k not in st.session_state:
        st.session_state[k] = None

def generate_chart_bytes_dynamic(df, chart_plan):
    try:
        chart_type = chart_plan.get("chart_type", "bar").lower().replace(" ", "_")
        x = chart_plan.get("x_axis") or chart_plan.get("x") or (df.columns[0] if len(df.columns) > 0 else None)
        y = chart_plan.get("y_axis") or chart_plan.get("y") or (df.columns[1] if len(df.columns) > 1 else None)

        if not x or not y or x not in df.columns or y not in df.columns:
            if len(df.columns) >= 2:
                x, y = df.columns[0], df.columns[1]
            else:
                return None

        if chart_type.startswith("bar"):
            fig = px.bar(df, x=x, y=y)
        elif chart_type.startswith("line"):
            fig = px.line(df, x=x, y=y)
        elif chart_type.startswith("scatter"):
            fig = px.scatter(df, x=x, y=y)
        else:
            fig = px.scatter(df, x=x, y=y)

        return fig.to_image(format="png", width=800, height=600, scale=2)
    except Exception as e:
        st.error(f"Chart generation error: {e}")
        return None

# ==============================
# 📂 File Upload (Main Body, not Sidebar)
# ==============================
st.markdown("## Upload Your Data")
uploaded_files = st.file_uploader("Upload CSV, JSON, or XLSX files (multiple allowed)", 
                                   type=["csv", "json", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    combined = pd.DataFrame()
    for uf in uploaded_files:
        ext = os.path.splitext(uf.name)[1].lower()
        uf.seek(0)
        if ext == ".csv":
            df = pd.read_csv(uf)
        elif ext == ".json":
            df = pd.read_json(uf)
        else:
            df = pd.read_excel(uf)
        combined = pd.concat([combined, df], ignore_index=True)

    st.session_state.combined_df = combined
    st.session_state.uploaded_file_name = uploaded_files[0].name
    st.session_state.uploaded_file_type = uploaded_files[0].type

    st.success("✅ Files loaded successfully")
    st.subheader("📊 Preview of Data")
    st.dataframe(combined.head())
    st.write(f"Rows: {len(combined)} | Columns: {', '.join(list(combined.columns))}")

# ==============================
# 🔎 Question Input
# ==============================
question = st.text_input("Ask a question or request a chart:", 
                         value="What can you show me about this data?")

if st.button("Analyze with AI (draft)"):
    if st.session_state.combined_df is None:
        st.error("Upload data first.")
    else:
        with st.spinner("Requesting draft from backend..."):
            csv_bytes = st.session_state.combined_df.to_csv(index=False).encode("utf-8")
            files = {"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")}
            data = {"user_query": question, "model_id": "gemini-1.5-flash"}

            try:
                resp = requests.post(f"{FASTAPI_BACKEND_URL}/get_insights", files=files, data=data, timeout=60)
                resp.raise_for_status()
                payload = resp.json()

                st.session_state.review_id = payload.get("review_id")
                draft = payload.get("draft", {})
                st.session_state.draft_insights = draft.get("insights") or ""
                st.session_state.draft_chart_spec = draft.get("chart_spec") or {}

                st.success(f"Draft created — Review ID: {st.session_state.review_id}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error calling backend: {e}")
                st.session_state.review_id = None

# ==============================
# 🧑‍⚖️ HITL Section
# ==============================
if st.session_state.review_id:
    st.markdown("---")
    st.header("Human-in-the-Loop Review")
    st.write(f"**Review ID:** {st.session_state.review_id}")

    st.subheader("AI Draft — Insights (editable)")
    edited_insight = st.text_area("Edit the AI-generated one-paragraph insight:",
                                  value=st.session_state.draft_insights or "", height=160)

    st.subheader("AI Draft — Chart Spec (editable JSON)")
    initial_chart = st.session_state.draft_chart_spec or {}
    chart_json_str = json.dumps(initial_chart, indent=2)
    edited_chart_json = st.text_area("Edit chart specification JSON:", value=chart_json_str, height=180)

    try:
        edited_chart_spec = json.loads(edited_chart_json) if edited_chart_json.strip() else {}
    except Exception as e:
        st.error(f"Invalid JSON for chart spec: {e}")
        edited_chart_spec = initial_chart

    st.subheader("Preview chart")
    chart_bytes = generate_chart_bytes_dynamic(st.session_state.combined_df, edited_chart_spec)
    if chart_bytes:
        st.image(chart_bytes, use_column_width=True)
        st.session_state.chart_bytes = chart_bytes
    else:
        st.info("Chart preview unavailable.")

    cols = st.columns(3)
    with cols[0]:
        if st.button("Approve & Submit Edits"):
            review_payload = {
                "review_id": st.session_state.review_id,
                "approved": True,
                "reviewer_name": "reviewer1",
                "reviewer_edits": {"summary": edited_insight, "conclusion": "", "chart_spec": edited_chart_spec}
            }
            try:
                resp = requests.post(f"{FASTAPI_BACKEND_URL}/review_insights", json=review_payload, timeout=30)
                resp.raise_for_status()
                st.success("Review submitted and approved ✅")
                st.session_state.approved_flag = True
                st.session_state.exec_summary = edited_insight
                st.session_state.conclusion = ""
            except Exception as e:
                st.error(f"Failed to submit review: {e}")

    with cols[1]:
        if st.button("Reject Draft"):
            review_payload = {
                "review_id": st.session_state.review_id,
                "approved": False,
                "reviewer_name": "reviewer1",
                "reviewer_edits": {"reason": "Rejected by reviewer"}
            }
            try:
                resp = requests.post(f"{FASTAPI_BACKEND_URL}/review_insights", json=review_payload, timeout=30)
                resp.raise_for_status()
                st.warning("Draft rejected. Re-run Analyze to generate a new draft.")
                st.session_state.approved_flag = False
            except Exception as e:
                st.error(f"Failed to reject: {e}")

    with cols[2]:
        if st.button("Save Edits Locally"):
            st.session_state.draft_insights = edited_insight
            st.session_state.draft_chart_spec = edited_chart_spec
            st.success("Edits saved locally.")

# ==============================
# 📑 Report Generation
# ==============================
st.markdown("---")
st.header("Generate Final Report")

with st.form("report_form"):
    report_format = st.selectbox("Report format", ("docx", "pdf"))
    exec_summary = st.text_area("Executive summary (editable)",
                                value=st.session_state.exec_summary or st.session_state.draft_insights or "",
                                height=150)
    conclusion = st.text_area("Conclusion (editable)", value=st.session_state.conclusion or "", height=100)
    attach_chart = st.checkbox("Attach generated chart preview", value=True)
    submit = st.form_submit_button("Generate Report")

if submit:
    if st.session_state.combined_df is None:
        st.error("Upload data first.")
    else:
        try:
            csv_bytes = st.session_state.combined_df.to_csv(index=False).encode("utf-8")
            files = {"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")}
            data = {
                "user_query": question or "",
                "report_format": report_format,
                "model_id": "gemini-1.5-flash",
                "review_id": st.session_state.review_id or ""
            }

            if attach_chart and st.session_state.chart_bytes:
                files["chart_image"] = ("chart.png", io.BytesIO(st.session_state.chart_bytes), "image/png")

            if st.session_state.review_id and not st.session_state.approved_flag:
                review_payload = {
                    "review_id": st.session_state.review_id,
                    "approved": True,
                    "reviewer_name": "streamlit_user",
                    "reviewer_edits": {"summary": exec_summary, "conclusion": conclusion,
                                       "chart_spec": st.session_state.draft_chart_spec or {}}
                }
                try:
                    resp = requests.post(f"{FASTAPI_BACKEND_URL}/review_insights", json=review_payload, timeout=30)
                    resp.raise_for_status()
                    st.success("Reviewer edits submitted ✅")
                    st.session_state.approved_flag = True
                except Exception as e:
                    st.error(f"Could not submit reviewer edits: {e}")

            with st.spinner("Generating report..."):
                resp = requests.post(f"{FASTAPI_BACKEND_URL}/generate_report", files=files, data=data, timeout=120)
                resp.raise_for_status()
                content = resp.content
                content_type = resp.headers.get("Content-Type", "application/octet-stream")
                disposition = resp.headers.get("Content-Disposition", "")
                filename = "generated_report.docx"
                if "filename=" in disposition:
                    filename = disposition.split("filename=")[-1].strip('"')

                st.session_state.generated_report = {
                    "content": content,
                    "filename": filename,
                    "content_type": content_type
                }
                st.success("Report generated 🎉")
        except Exception as e:
            st.error(f"Failed to generate report: {e}")

# ✅ Download button OUTSIDE the form
if st.session_state.generated_report:
    st.download_button(
        "📥 Download Report",
        data=st.session_state.generated_report["content"],
        file_name=st.session_state.generated_report["filename"],
        mime=st.session_state.generated_report["content_type"]
    )

st.markdown("⚠️ If you used HITL, ensure you clicked **Approve & Submit Edits** before generating the report.")
