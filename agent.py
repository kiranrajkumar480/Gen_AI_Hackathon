import json
import streamlit as st
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from typing import Tuple, Dict
from config import LLM
import career_tools as ct
import feedback as fb
import document_processor as dp
import comparator as cmp
from pathlib import Path

class JobCoachAgent:
    def __init__(self):
        self.memory = ConversationBufferMemory()
        self._build_tools()
        self.agent = initialize_agent(
            tools=self.tools,
            llm=LLM,
            agent_type="zero-shot-react-description",
            memory=self.memory,
            verbose=True,
        )

    def _build_tools(self):
        self.tools = [
            Tool(
                name="ResumeAnalyzer",
                func=lambda jd: ct.analyze_resume(
                    st.session_state.get("resume_text",""),
                    jd, fb.positive_examples("resume analysis")
                ),
                description="Analyze resume against a job description."
            ),
            # (SkillGapAnalyzer etc.) ...
        ]

    # --- direct calls for free-form queries -------------------
    def answer(self, query:str, jd:str)->Tuple[str|Dict,dict|None]:
        if not Path("faiss_index").exists():
            return "Please upload and process the resume first.", None
        docs = dp.load_faiss_index().similarity_search(query)
        return cmp.run(query, docs,
                       st.session_state.get("resume_text",""),
                       jd, fb.positive_examples(query))
