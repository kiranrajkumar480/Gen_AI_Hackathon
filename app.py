import streamlit as st
import json, pandas as pd, plotly.express as px
import document_processor as dp
import career_tools as ct
import feedback as fb
import visual  # Import the new visualization module
from agent import JobCoachAgent
import requests
from typing import List, Dict

# 🌐 Page Settings
st.set_page_config(page_title="JobCoach AI", layout="wide", page_icon="💼")

# 🧠 Initialize Session Variables
for key in ["resume_text", "job_description", "current_role", "analysis_done", "result", "chat_history", "cover_letter", "roadmap", "similar_jobs"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["chat_history", "similar_jobs"] else ""

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

    # Extract skills from resume and job description
    resume_skills = ct.extract_skills(st.session_state["resume_text"])
    jd_skills = ct.extract_job_skills(st.session_state["job_description"])
    gap_data = ct.get_skill_gap(resume_skills, jd_skills)

    # Perform resume analysis
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

    # Add Skill Gap Visualization (Using Plotly)
    st.markdown("#### 📊 **Skill Gap Analysis**")
    st.markdown("This chart compares the skills on your resume with those required in the job description:")

    # Prepare data for the chart
    skills = list(gap_data["Resume"].keys())
    resume_values = [gap_data["Resume"][skill] for skill in skills]
    jd_values = [gap_data["Job Description"][skill] for skill in skills]

    # Debug: Display the data being used for the chart
    st.write("**Debug - Skills Extracted:**", skills)
    st.write("**Debug - Resume Values:**", resume_values)
    st.write("**Debug - Job Description Values:**", jd_values)

    # If no skills are extracted, display a warning and skip chart rendering
    if not skills:
        st.warning("⚠️ No skills were extracted from your resume or job description. Please ensure your documents contain relevant skills.")
    else:
        # Create and display the Plotly chart
        st.markdown("**Skill Comparison Chart**")
        fig = visual.create_skill_gap_chart(gap_data)
        st.plotly_chart(fig, use_container_width=True)

    # Add Career Roadmap Generator
    st.markdown("---")
    st.header("🗺️ Career Roadmap Generator")
    st.markdown("Enter your target role to generate a personalized career roadmap:")

    target_role = st.text_input("🎯 Target Role", placeholder="e.g., Product Manager, Data Scientist")

    if st.button("📈 Generate Career Roadmap"):
        if not target_role.strip():
            st.warning("Please enter a target role first.")
        elif not st.session_state["resume_text"] or not st.session_state["job_description"]:
            st.info("Upload your resume and paste the job description to generate a career roadmap.")
        else:
            roadmap = job_coach.agent.run(f"Generate a career roadmap to transition to {target_role}")
            st.session_state["roadmap"] = roadmap
            st.success("✅ Career Roadmap Generated!")

    if st.session_state["roadmap"]:
        st.subheader("🛤️ Your Career Roadmap")
        st.markdown(st.session_state["roadmap"])
        st.download_button("⬇️ Download Career Roadmap", data=str(st.session_state["roadmap"]), file_name="career_roadmap.txt")

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

# ---------- Similar Jobs Finder --------------------------------
st.markdown("---")
st.header("🔎 Similar Jobs Finder")
st.markdown("Find job listings similar to your target role in real-time:")

# Function to fetch similar jobs using a job search API (e.g., JSearch via RapidAPI)
def fetch_similar_jobs(role: str, location: str = "USA", num_results: int = 3) -> List[Dict]:
    # Note: This requires a RapidAPI key for JSearch or a similar job search API
    # For demonstration purposes, replace 'YOUR_RAPIDAPI_KEY' with your actual key
    api_key = "YOUR_RAPIDAPI_KEY"  # Replace with your RapidAPI key
    url = "https://jsearch.p.rapidapi.com/search"
    querystring = {"query": f"{role} in {location}", "num_results": str(num_results)}
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()
        jobs = response.json().get("data", [])
        return [
            {
                "title": job.get("job_title", "N/A"),
                "company": job.get("employer_name", "N/A"),
                "location": job.get("job_city", "N/A") + ", " + job.get("job_country", "N/A"),
                "link": job.get("job_apply_link", "#")
            }
            for job in jobs
        ]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching jobs: {e}")
        return []

# UI for Similar Jobs Finder
similar_role = st.text_input("🔍 Search for Similar Roles", placeholder="e.g., Data Scientist, Product Manager", value=st.session_state["current_role"])
location = st.text_input("📍 Location", placeholder="e.g., USA, London", value="USA")

if st.button("🔎 Find Similar Jobs"):
    if not similar_role.strip():
        st.warning("Please enter a role to search for similar jobs.")
    else:
        similar_jobs = fetch_similar_jobs(similar_role, location)
        st.session_state["similar_jobs"] = similar_jobs
        st.success("✅ Similar Jobs Found!")

# Display Similar Jobs
if st.session_state["similar_jobs"]:
    st.subheader("🌟 Similar Job Listings")
    for job in st.session_state["similar_jobs"]:
        st.markdown(
            f"""
            <div style='background-color:#f5f5f5;padding:15px;border-radius:10px;margin-bottom:10px;'>
                <strong>📌 {job['title']}</strong><br>
                <strong>🏢 Company:</strong> {job['company']}<br>
                <strong>📍 Location:</strong> {job['location']}<br>
                <a href='{job['link']}' target='_blank'>Apply Here</a>
            </div>
            """,
            unsafe_allow_html=True
        )