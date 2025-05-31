# career_tools.py
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from config import NLP, LLM
from typing import List, Dict

_SKILL_KEYWORDS = {
    "Python","SQL","Machine Learning","Data Analysis",
    "Project Management","JavaScript","AWS","Marketing"
}

def extract_skills(resume_text: str) -> List[str]:
    doc = NLP(resume_text)
    return list({tok.text for tok in doc if tok.text in _SKILL_KEYWORDS
                                   or tok.pos_ in {"NOUN","PROPN"}})

def _simple_chain(template: str) -> LLMChain:
    return LLMChain(
        llm=LLM,
        prompt=PromptTemplate(template=template, input_variables=["data"])
    )

# --- each high-level feature gets its own helper ----------------
def analyze_resume(resume: str, jd: str, examples: str) -> str:
    prompt = """As a career coach, compare the resume with the job description...
    {data}"""
    chain = _simple_chain(prompt)
    return chain.run({"data": json.dumps(
        {"resume": resume,"job_description": jd,"examples": examples})})

# (skill_gap_analysis, generate_interview_questions, generate_career_roadmap,
#  fetch_job_description)  -- copy the logic from your monolith, unchanged
