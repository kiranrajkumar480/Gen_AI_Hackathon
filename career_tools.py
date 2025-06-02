# career_tools.py
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from config import NLP, LLM
from typing import List, Dict

_SKILL_KEYWORDS = {
    "Python", "SQL", "Machine Learning", "Data Analysis",
    "Project Management", "JavaScript", "AWS", "Marketing"
}

def extract_skills(resume_text: str) -> List[str]:
    doc = NLP(resume_text)
    return list({tok.text for tok in doc if tok.text in _SKILL_KEYWORDS
                                   or tok.pos_ in {"NOUN", "PROPN"}})

def extract_job_skills(jd: str) -> List[str]:
    doc = NLP(jd)
    return list({tok.text for tok in doc if tok.text in _SKILL_KEYWORDS or tok.pos_ in {"NOUN", "PROPN"}})

def get_skill_gap(resume_skills: List[str], jd_skills: List[str]) -> Dict[str, Dict[str, int]]:
    all_skills = set(resume_skills).union(set(jd_skills))
    gap_data = {"Resume": {}, "Job Description": {}}
    
    for skill in all_skills:
        gap_data["Resume"][skill] = 1 if skill in resume_skills else 0
        gap_data["Job Description"][skill] = 1 if skill in jd_skills else 0
    
    return gap_data

def _simple_chain(template: str) -> LLMChain:
    return LLMChain(
        llm=LLM,
        prompt=PromptTemplate(template=template, input_variables=["data"])
    )

def analyze_resume(resume: str, jd: str, examples: str) -> str:
    prompt = """As a career coach, compare the resume with the job description...
    {data}"""
    chain = _simple_chain(prompt)
    return chain.run({"data": json.dumps(
        {"resume": resume, "job_description": jd, "examples": examples})})

def generate_career_roadmap(current_role: str, target_role: str, resume_skills: List[str], jd_skills: List[str]) -> str:
    """
    Generate a personalized career roadmap to transition from the current role to the target role.
    
    Args:
        current_role (str): The user's current role (e.g., "Marketing Manager").
        target_role (str): The user's target role (e.g., "Product Manager").
        resume_skills (List[str]): Skills extracted from the user's resume.
        jd_skills (List[str]): Skills required for the target role, extracted from the job description.
    
    Returns:
        str: A step-by-step career roadmap as a formatted string.
    """
    # Identify missing skills
    missing_skills = [skill for skill in jd_skills if skill not in resume_skills]

    # Prepare data for the LLM
    data = {
        "current_role": current_role,
        "target_role": target_role,
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "missing_skills": missing_skills
    }

    # Define the prompt for generating a career roadmap
    prompt = """
    As a career coach, create a detailed, step-by-step career roadmap for a user transitioning from their current role to a target role. Provide actionable steps including skill development, certifications, experience-building activities, and networking tips. Format the roadmap as a numbered list.

    Current Role: {data[current_role]}
    Target Role: {data[target_role]}
    Current Skills: {data[resume_skills]}
    Required Skills for Target Role: {data[jd_skills]}
    Skills to Develop: {data[missing_skills]}

    Career Roadmap:
    """
    chain = _simple_chain(prompt)
    roadmap = chain.run({"data": data})

    return roadmap