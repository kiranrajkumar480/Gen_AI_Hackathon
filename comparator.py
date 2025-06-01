from typing import Dict, Tuple, List
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from config import LLM

_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

_PROMPT = PromptTemplate(
    template="""Provide a detailed, professional answer...
    Context:{context} Resume:{resume} JD:{job_description}
    Q:{question}""",
    input_variables=["context","resume","job_description","question"]
)

def run(query:str, docs:List, resume:str, jd:str, examples:str
       ) -> Tuple[Dict,str]:
    chain = load_qa_chain(LLM, chain_type="stuff", prompt=_PROMPT)
    resp = chain(
        {"input_documents": docs,"resume":resume,
         "job_description":jd,"question":query},return_only_outputs=True)
    return {"Gemini": resp["output_text"]}, None  # metrics optional
