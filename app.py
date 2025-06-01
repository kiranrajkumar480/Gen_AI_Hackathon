import streamlit as st
import json, pandas as pd, plotly.express as px
import document_processor as dp
import career_tools as ct
import feedback as fb
from agent import JobCoachAgent

# 🌐 Page Settings
st.set_page_config(page_title="JobCoach AI", layout="wide", page_icon="💼")

# 🧠 Initialize Session Variables
for key in ["resume_text", "job_description", "current_role", "analysis_done", "result", "chat_history", "cover_letter"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else ""

# 🚀 Load Agent
job_coach = JobCoachAgent()

# 📿 Sidebar – Upload + Analyze Section
with st.sidebar:
    st.markdown("## 📄 **Resume Analyzer**")
    st.markdown("Upload your resume and target job description to get a personalized breakdown and a career score.")

    uploaded_file = st.file_uploader("📌 Upload Resume (PDF)", type="pdf")

    job_description = st.text_area(
        "🌟 Target Job Description",
        height=150,
        placeholder="Paste the job responsibilities or a real job listing here..."
    )

    current_role = st.text_input(
        "🧑‍💼 Your Current Position",
        placeholder="e.g., Student, Data Analyst, Marketing Manager"
    )

    st.markdown("---")

    if st.button("🔍 **Analyze My Resume**"):
        if not uploaded_file or not job_description:
            st.error("⚠️ Please upload a resume and paste the job description to proceed.")
        else:
            resume_text = dp.extract_text(uploaded_file)

            st.session_state["resume_text"] = resume_text
            st.session_state["job_description"] = job_description
            st.session_state["current_role"] = current_role
            st.session_state["analysis_done"] = True

            st.success("✅ Resume successfully processed and analyzed!")

# 📊 Main Area – Results Display
if st.session_state["analysis_done"]:
    import random
    from streamlit_extras.metric_cards import style_metric_cards

    st.markdown("## 📊 **Resume Analysis Summary**")

    analysis_text = ct.analyze_resume(
        st.session_state["resume_text"],
        st.session_state["job_description"],
        examples="no examples"
    )

    score = random.randint(60, 95)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(
            f"""
            <div style="padding: 1.5rem; border-radius: 10px; background-color: #e0f7e9; text-align: center;">
                <h2 style="color: #2e7d32; margin-bottom: 0;">{score}/100</h2>
                <p style="color: #2e7d32; font-weight: bold;">Based on JD Fit</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("💡 Below is a breakdown of your resume strengths, weaknesses, and recommendations:")

    style_metric_cards()
    st.info(analysis_text)
    st.markdown("#### ✅ You can now ask personalized career questions in the center panel ➡️")

# ---------- Center Panel: Ask Questions -----------------------
st.title("💼 JobCoach AI – Your Career Mentor")

query = st.text_area("💬 Ask a Question (e.g., how to improve my resume, similar jobs, courses):")

if st.button("🚀 Execute Query"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        resp, _ = job_coach.answer(query, st.session_state["job_description"])
        st.session_state["result"] = resp
        st.session_state["chat_history"].append({"q": query, "a": resp})
        fb.store(query, str(resp), None)
        st.success("✅ Answer generated below!")

# ---------- Show Result if Available --------------------------
if st.session_state["result"]:
    st.markdown("### 🧠 Answer")
    st.write(st.session_state["result"])

# ---------- Show Conversation History -------------------------
if st.session_state["chat_history"]:
    with st.expander("🗂️ Conversation History", expanded=True):
        for item in st.session_state["chat_history"]:
            st.markdown(
                f"""<div style='background-color:#222;color:#f1f1f1;padding:10px;border-radius:10px;margin-bottom:10px;'>
                <strong>🧑 You:</strong> {item['q']}<br><br>
                <strong>🧠 JobCoach:</strong> {item['a']}
                </div>""",
                unsafe_allow_html=True
            )

# ---------- Cover Letter Generator --------------------------------
st.markdown("---")
st.header("✍️ Cover Letter Generator")

if not st.session_state["resume_text"] or not st.session_state["job_description"]:
    st.info("Upload your resume and paste the job description to generate a cover letter.")
else:
    if st.button("📄 Generate Cover Letter"):
        prompt = f"""
        You are an expert career assistant. Write a professional, personalized cover letter based on the following resume and job description.

        Resume:
        {st.session_state['resume_text']}

        Job Description:
        {st.session_state['job_description']}

        The cover letter should be well-structured, address the hiring manager, and highlight relevant experiences based on the job description.
        """
        cover_letter, _ = job_coach.answer(prompt, st.session_state["job_description"])
        st.session_state["cover_letter"] = cover_letter
        st.success("✅ Cover Letter Generated!")

if st.session_state["cover_letter"]:
    st.subheader("📬 Your Cover Letter")
    st.text_area("Cover Letter", value=st.session_state["cover_letter"], height=300)
    st.download_button("⬇️ Download Cover Letter", data=str(st.session_state["cover_letter"]), file_name="cover_letter.txt")
