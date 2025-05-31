import streamlit as st, json, pandas as pd, plotly.express as px
import document_processor as dp
import career_tools as ct
import feedback as fb
from agent import JobCoachAgent

st.set_page_config(page_title="JobCoach AI", layout="wide", page_icon="💼")
job_coach = JobCoachAgent()

# -- session state defaults (same as before) --------------------
if "resume_text" not in st.session_state: ...
# (keep the same state init block)

# ---------- Sidebar: upload & process documents ---------------
with st.sidebar.expander("📄 Input Documents", expanded=True):
    pdf = st.file_uploader("Upload Resume (PDF)")
    jd  = st.text_area("Paste Job Description")
    role = st.text_input("Current Role")
    if st.button("📥 Process Documents"):
        if pdf:
            text = dp.pdf_to_text(pdf)
            chunks = dp.split_text(text)
            dp.build_faiss_index(chunks)
            st.session_state.resume_text = text
            st.session_state.current_skills = ct.extract_skills(text)
            st.success("Resume processed!")
        # ... (same jd / role logic)

# ---------- Main UI -------------------------------------------
st.title("💼 JobCoach AI – Your Career Mentor")
query = st.text_area("Ask a question:")
if st.button("🚀 Execute Query"):
    resp, _ = job_coach.answer(query, st.session_state.job_description)
    st.session_state.result = resp
    fb.store(query, str(resp), None)

# -- render result + feedback & roadmap viz (reuse old code) ----
