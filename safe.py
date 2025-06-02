import streamlit as st
import spacy
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import json
import uuid
import time
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("Missing GOOGLE_API_KEY. Please configure it in your .env file.")
    st.stop()

# Configure Google Generative AI
from google.generativeai import configure
configure(api_key=google_api_key)

# Initialize Google Gemini AI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Initialize ChromaDB client for feedback storage
chroma_client = chromadb.PersistentClient(path="./feedback_db")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
feedback_collection = chroma_client.get_or_create_collection(
    name="career_feedback",
    embedding_function=embedding_function
)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    st.error(f"Error loading spaCy model: {e}. Please run 'python -m spacy download en_core_web_sm'.")
    st.stop()

def block_inappropriate_query(user_query):
    """Block queries with toxic or inappropriate content."""
    toxic_keywords = ["fuck", "sex", "sexual", "shit", "ass", "violence", "drugs", "hate", "political"]
    if any(word in user_query.lower() for word in toxic_keywords):
        return "I'm sorry, but I cannot process inappropriate or offensive queries."
    return None

def block_irrelevant_query(user_query):
    """Block queries irrelevant to career guidance."""
    relevant_keywords = ["resume", "skill", "interview", "career", "job", "role", "position", "qualification", "experience"]
    if not any(keyword in user_query.lower() for keyword in relevant_keywords):
        return "This query appears irrelevant to career guidance. Please ask something related to jobs or careers."
    return None

def store_feedback(user_query, response, feedback):
    """Store feedback in ChromaDB for continuous improvement."""
    try:
        doc_id = str(uuid.uuid4())
        metadata = {
            "user_query": user_query,
            "response": response,
            "feedback": feedback if feedback else "None",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        feedback_collection.add(
            documents=[f"{user_query} -> {response}"],
            metadatas=[metadata],
            ids=[doc_id]
        )
        return True
    except Exception as e:
        st.error(f"Feedback storage error: {e}")
        return False

def get_relevant_feedback(user_query):
    """Retrieve relevant positive feedback examples from ChromaDB."""
    try:
        results = feedback_collection.query(
            query_texts=[user_query],
            where={"feedback": "positive"},
            n_results=3
        )
        examples = []
        for i in range(len(results["ids"][0])):
            example = (
                f"Example {i+1}:\n"
                f"Query: {results['metadatas'][0][i]['user_query']}\n"
                f"Response: {results['metadatas'][0][i]['response']}\n"
            )
            examples.append(example)
        return "\n".join(examples) if examples else "No relevant positive feedback examples found."
    except Exception as e:
        st.warning(f"Error retrieving feedback examples: {e}")
        return "No relevant positive feedback examples found."

class DocumentProcessor:
    @staticmethod
    def get_pdf_text(pdf_docs):
        text = ""
        try:
            reader = PdfReader(pdf_docs)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            return ""

    @staticmethod
    def get_text_chunks(text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return splitter.split_text(text)

    @staticmethod
    def get_vector_store(text_chunks):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
            return True
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return False

class CareerTools:
    @staticmethod
    def extract_skills(resume_text):
        doc = nlp(resume_text)
        skills = []
        skill_keywords = ["Python", "SQL", "Machine Learning", "Data Analysis", "Project Management", "JavaScript", "AWS", "Marketing"]
        for token in doc:
            if token.text in skill_keywords or token.pos_ in ["NOUN", "PROPN"]:
                skills.append(token.text)
        return list(set(skills))

    @staticmethod
    def analyze_resume(resume_text, job_description, feedback_examples):
        prompt = PromptTemplate(
            template="""
            As a career coach, compare the resume with the job description. Provide feedback on:
            - Matching skills and experience
            - Missing skills or qualifications
            - Suggestions for optimization (e.g., keywords, formatting)
            
            Relevant Positive Feedback Examples:
            {feedback_examples}
            
            Resume: {resume}
            Job Description: {job_description}
            Feedback:
            """,
            input_variables=["resume", "job_description", "feedback_examples"]
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        try:
            response = chain.run({"resume": resume_text, "job_description": job_description, "feedback_examples": feedback_examples})
            return response
        except Exception as e:
            return f"Error analyzing resume: {e}"

    @staticmethod
    def skill_gap_analysis(current_skills, target_role, feedback_examples):
        prompt = PromptTemplate(
            template="""
            As a career coach, identify skill gaps for transitioning to the target role based on current skills.
            Provide:
            - Missing skills
            - Recommended learning paths (e.g., online courses)
            
            Relevant Positive Feedback Examples:
            {feedback_examples}
            
            Current Skills: {skills}
            Target Role: {role}
            Response:
            """,
            input_variables=["skills", "role", "feedback_examples"]
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        try:
            role_skills = {
                "Data Scientist": ["Python", "SQL", "Machine Learning", "Statistics", "Data Visualization"],
                "Product Manager": ["Project Management", "Agile", "UX Design", "Market Research", "Stakeholder Management"],
                "Senior Developer": ["Python", "JavaScript", "System Design", "APIs", "Database Management"]
            }
            required_skills = role_skills.get(target_role, ["Python", "SQL"])
            missing_skills = [skill for skill in required_skills if skill.lower() not in [s.lower() for s in current_skills]]
            learning_paths = {
                "Python": "Coursera: Python for Everybody",
                "SQL": "edX: SQL for Data Science",
                "Machine Learning": "Coursera: Machine Learning by Andrew Ng",
                "Statistics": "Khan Academy: Statistics and Probability",
                "Data Visualization": "Coursera: Data Visualization with Tableau",
                "Project Management": "Coursera: Google Project Management Certificate",
                "Agile": "Udemy: Agile Fundamentals",
                "UX Design": "Coursera: UX Design Fundamentals",
                "Market Research": "LinkedIn Learning: Market Research Foundations",
                "Stakeholder Management": "Coursera: Stakeholder Management",
                "JavaScript": "freeCodeCamp: JavaScript Certification",
                "System Design": "Grokking the System Design Interview",
                "APIs": "Postman API Fundamentals",
                "Database Management": "Udemy: Database Design and Management"
            }
            response = chain.run({
                "skills": ", ".join(current_skills),
                "role": target_role,
                "feedback_examples": feedback_examples
            })
            return {"missing_skills": missing_skills, "learning_paths": [learning_paths.get(skill, "Explore online courses") for skill in missing_skills], "details": response}
        except Exception as e:
            return {"error": f"Error in skill gap analysis: {e}"}

    @staticmethod
    def generate_interview_questions(job_description, role_type, feedback_examples):
        prompt = PromptTemplate(
            template="""
            As a career coach, generate 5 {role_type} interview questions based on the job description, with STAR-method sample answers.
            
            Relevant Positive Feedback Examples:
            {feedback_examples}
            
            Job Description: {job_description}
            Output format:
            - Question 1: [Question]
              Sample Answer: [STAR-based answer]
            ...
            """,
            input_variables=["job_description", "role_type", "feedback_examples"]
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        try:
            response = chain.run({"job_description": job_description, "role_type": role_type, "feedback_examples": feedback_examples})
            return response
        except Exception as e:
            return f"Error generating interview questions: {e}"

    @staticmethod
    def generate_career_roadmap(current_role, target_role, missing_skills, learning_paths, feedback_examples):
        prompt = PromptTemplate(
            template="""
            As a career coach, create a step-by-step career roadmap from the current role to the target role.
            
            Relevant Positive Feedback Examples:
            {feedback_examples}
            
            Current Role: {current_role}
            Target Role: {target_role}
            Missing Skills: {missing_skills}
            Learning Paths: {learning_paths}
            Roadmap:
            """,
            input_variables=["current_role", "target_role", "missing_skills", "learning_paths", "feedback_examples"]
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        try:
            response = chain.run({
                "current_role": current_role,
                "target_role": target_role,
                "missing_skills": ", ".join(missing_skills),
                "learning_paths": ", ".join(learning_paths),
                "feedback_examples": feedback_examples
            })
            steps = [f"Learn {skill}: {path}" for skill, path in zip(missing_skills, learning_paths)]
            steps.extend([
                f"Update resume to highlight {target_role} skills",
                f"Apply for {target_role} roles",
                f"Network with {target_role} professionals on LinkedIn"
            ])
            return {"steps": steps, "details": response}
        except Exception as e:
            return {"error": f"Error generating roadmap: {e}"}

    @staticmethod
    def fetch_job_description(role):
        mock_descriptions = {
            "Data Scientist": "Seeking a Data Scientist proficient in Python, SQL, and Machine Learning to build predictive models.",
            "Product Manager": "Looking for a Product Manager with skills in Agile, UX Design, and Market Research to lead product development.",
            "Senior Developer": "Hiring a Senior Developer experienced in Python, JavaScript, and System Design for scalable applications."
        }
        return mock_descriptions.get(role, "No job description available for this role.")

class MultiLLMComparator:
    @staticmethod
    def process_query(query, context, resume_text="", job_description="", feedback_examples=""):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        results = {}
        metrics = {}
        prompt_template = PromptTemplate(
            template="""
            Provide a detailed, professional answer based on the context. Tailor to career guidance.
            
            Relevant Positive Feedback Examples:
            {feedback_examples}
            
            Context: {context}
            Resume: {resume}
            Job Description: {job_description}
            Question: {question}
            Answer:
            """,
            input_variables=["context", "resume", "job_description", "question", "feedback_examples"]
        )
        try:
            chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)
            response = chain(
                {
                    "input_documents": context,
                    "resume": resume_text,
                    "job_description": job_description,
                    "question": query,
                    "feedback_examples": feedback_examples
                },
                return_only_outputs=True
            )
            results["Gemini"] = response["output_text"]
            metrics["Gemini"] = {
                "length": len(response["output_text"].split()),
                "similarity_to_gemini": 1.0  # Same model, so similarity is 1
            }
        except Exception as e:
            results["Gemini"] = f"Error: {str(e)}"
            metrics["Gemini"] = {"length": 0, "similarity_to_gemini": 0.0}
        
        return results, metrics

class JobCoachAgent:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.career_tools = CareerTools()
        self.comparator = MultiLLMComparator()
        self.memory = ConversationBufferMemory()
        self.setup_agent()

    def setup_agent(self):
        tools = [
            Tool(
                name="ResumeAnalyzer",
                func=lambda x: self.career_tools.analyze_resume(
                    st.session_state.get("resume_text", ""), x, get_relevant_feedback("resume analysis")
                ),
                description="Analyzes a resume against a job description."
            ),
            Tool(
                name="SkillGapAnalyzer",
                func=lambda x: json.dumps(self.career_tools.skill_gap_analysis(
                    st.session_state.get("current_skills", []), x, get_relevant_feedback("skill gap analysis")
                )),
                description="Identifies skill gaps for a target role."
            ),
            Tool(
                name="InterviewQuestionGenerator",
                func=lambda x: self.career_tools.generate_interview_questions(
                    x, st.session_state.get("interview_type", "behavioral"), get_relevant_feedback("interview questions")
                ),
                description="Generates interview questions."
            ),
            Tool(
                name="CareerRoadmapGenerator",
                func=lambda x: json.dumps(self.career_tools.generate_career_roadmap(
                    st.session_state.get("current_role", ""), x,
                    st.session_state.get("missing_skills", []),
                    st.session_state.get("learning_paths", []),
                    get_relevant_feedback("career roadmap")
                )),
                description="Generates a career roadmap."
            ),
            Tool(
                name="JobDescriptionFetcher",
                func=lambda x: self.career_tools.fetch_job_description(x),
                description="Fetches a job description for a given role."
            )
        ]
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent_type="zero-shot-react-description",
            memory=self.memory,
            verbose=True
        )

    def process_query(self, user_question, job_description=""):
        try:
            feedback_examples = get_relevant_feedback(user_question)
            if "resume" in user_question.lower():
                response = self.agent.run(f"Analyze resume for job description: {job_description}")
                return response, None
            elif "skill" in user_question.lower():
                response = self.agent.run(f"Perform skill gap analysis for role: {user_question}")
                return response, None
            elif "interview" in user_question.lower():
                response = self.agent.run(f"Generate interview questions for job description: {job_description}")
                return response, None
            elif "roadmap" in user_question.lower() or "career plan" in user_question.lower():
                response = self.agent.run(f"Generate career roadmap for role: {user_question}")
                return response, None
            elif "job description" in user_question.lower():
                response = self.agent.run(f"Fetch job description for role: {user_question}")
                return response, None
            else:
                if not Path("faiss_index").exists():
                    return "Please upload and process the resume first.", None
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                docs = db.similarity_search(user_question)
                response, metrics = self.comparator.process_query(
                    user_question, docs, st.session_state.get("resume_text", ""), job_description, feedback_examples
                )
                return response, metrics
        except Exception as e:
            return f"Error processing query: {str(e)}", None

# Streamlit UI Configuration
st.set_page_config(page_title="JobCoach AI", layout="wide", page_icon="💼")

# Initialize session state
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "current_skills" not in st.session_state:
    st.session_state.current_skills = []
if "job_description" not in st.session_state:
    st.session_state.job_description = ""
if "current_role" not in st.session_state:
    st.session_state.current_role = ""
if "missing_skills" not in st.session_state:
    st.session_state.missing_skills = []
if "learning_paths" not in st.session_state:
    st.session_state.learning_paths = []
if "interview_type" not in st.session_state:
    st.session_state.interview_type = "behavioral"
if "result" not in st.session_state:
    st.session_state.result = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = None
if "corrected_response" not in st.session_state:
    st.session_state.corrected_response = None
if "execute_query" not in st.session_state:
    st.session_state.execute_query = False

# Initialize JobCoach agent
job_coach = JobCoachAgent()

# Sidebar for inputs
with st.sidebar.expander("📄 Input Documents", expanded=True):
    pdf_docs = st.file_uploader("Upload Resume (PDF)", accept_multiple_files=False)
    job_description = st.text_area("Paste Job Description (or leave blank to fetch)")
    current_role = st.text_input("Current Role (e.g., Marketing Manager)")
    if st.button("📥 Process Documents", use_container_width=True):
        if pdf_docs:
            with st.spinner("Processing resume..."):
                resume_text = job_coach.doc_processor.get_pdf_text(pdf_docs)
                if resume_text:
                    text_chunks = job_coach.doc_processor.get_text_chunks(resume_text)
                    if job_coach.doc_processor.get_vector_store(text_chunks):
                        st.session_state.resume_text = resume_text
                        st.session_state.current_skills = job_coach.career_tools.extract_skills(resume_text)
                        st.success(f"Resume processed successfully! Extracted skills: {', '.join(st.session_state.current_skills)}")
        if job_description:
            st.session_state.job_description = job_description
            st.success("Job description saved!")
        elif current_role:
            st.session_state.job_description = job_coach.career_tools.fetch_job_description(current_role)
            st.success("Fetched job description!")
        if current_role:
            st.session_state.current_role = current_role
            st.success("Current role saved!")
        if not (pdf_docs or job_description or current_role):
            st.error("Please provide a resume, job description, or current role.")

# Main Interface
st.title("💼 JobCoach AI: Your Career Mentor")
st.write("Get personalized career guidance through natural language queries")

# Query Section
with st.expander("🔍 Start Your Career Journey", expanded=True):
    user_input = st.text_area(
        "Enter your question:",
        "Review my resume for a Data Scientist position",
        height=100,
        label_visibility="collapsed",
        key="query_input"
    )

    if st.button("🚀 Execute Query", use_container_width=True) or st.session_state.execute_query:
        with st.spinner("Processing..."):
            try:
                inappropriate_result = block_inappropriate_query(user_input)
                if inappropriate_result:
                    st.session_state.result = inappropriate_result
                    st.session_state.feedback = None
                    st.session_state.corrected_response = None
                else:
                    irrelevant_result = block_irrelevant_query(user_input)
                    if irrelevant_result:
                        st.session_state.result = irrelevant_result
                        st.session_state.feedback = None
                        st.session_state.corrected_response = None
                    else:
                        start_time = time.time()
                        response, metrics = job_coach.process_query(user_input, st.session_state.job_description)
                        execution_time = time.time() - start_time
                        st.session_state.result = response
                        st.session_state.feedback = None
                        st.session_state.corrected_response = None
                        store_feedback(user_input, str(response), None)
            except Exception as e:
                st.error(f"Error: {str(e)}")
        st.session_state.execute_query = False

# Handle Enter key press
def on_enter():
    st.session_state.execute_query = True

st.text_input("Press Enter to Execute", key="enter_input", on_change=on_enter, label_visibility="collapsed")

# Results Display
if st.session_state.result:
    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📄 Query Results")
        if isinstance(st.session_state.result, dict):
            for llm, result in st.session_state.result.items():
                st.write(f"**{llm} Response**:")
                st.write(result)
                st.write("---")
        else:
            st.write(st.session_state.result)

        # Feedback section
        st.divider()
        st.subheader("💬 Provide Feedback")
        if st.session_state.feedback is None:
            st.write("Was this response helpful?")
            col_fb1, col_fb2 = st.columns(2)
            with col_fb1:
                if st.button("👍 Yes"):
                    st.session_state.feedback = "positive"
                    store_feedback(
                        user_input if not st.session_state.corrected_response else st.session_state.corrected_response,
                        str(st.session_state.result),
                        "positive"
                    )
                    st.success("Thank you! Feedback recorded.")
            with col_fb2:
                if st.button("👎 No"):
                    st.session_state.feedback = "negative"
                    store_feedback(
                        user_input if not st.session_state.corrected_response else st.session_state.corrected_response,
                        str(st.session_state.result),
                        "negative"
                    )
                    with st.spinner("Analyzing and correcting your query..."):
                        prompt = PromptTemplate(
                            template="""
                            The user provided negative feedback for the following query and response:
                            
                            Query: {query}
                            Response: {response}
                            
                            Relevant Positive Feedback Examples:
                            {feedback_examples}
                            
                            Suggest a corrected response that aligns with career guidance goals.
                            Response:
                            """,
                            input_variables=["query", "response", "feedback_examples"]
                        )
                        chain = LLMChain(llm=llm, prompt=prompt)
                        corrected_response = chain.run({
                            "query": user_input,
                            "response": str(st.session_state.result),
                            "feedback_examples": get_relevant_feedback(user_input)
                        })
                        st.session_state.corrected_response = corrected_response
                        st.session_state.result = corrected_response
                        st.session_state.feedback = None
                        store_feedback(user_input, corrected_response, None)
                        st.info(f"Corrected Response: {corrected_response}")
        else:
            st.success("Thank you for your feedback! It's helping us improve.")
            if st.session_state.corrected_response:
                st.info(f"Corrected Response: {st.session_state.corrected_response}")

    with col2:
        st.subheader("📈 Career Visualization")
        if st.session_state.result and "roadmap" in user_input.lower():
            try:
                roadmap_data = json.loads(st.session_state.result)
                steps = roadmap_data.get("steps", [])
                if steps:
                    df = pd.DataFrame({"Step": list(range(1, len(steps) + 1)), "Task": steps})
                    fig = px.timeline(df, x_start="Step", x_end="Step", y="Task", title="Career Roadmap")
                    st.plotly_chart(fig, use_container_width=True)
            except json.JSONDecodeError:
                st.warning("Could not visualize roadmap. Ensure the response is a valid roadmap.")
        else:
            st.write("Run a career roadmap query to visualize your path.")

if __name__ == "__main__":
    st.write("JobCoach AI is running!")