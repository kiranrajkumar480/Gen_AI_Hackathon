import uuid, time
from typing import Optional
from config import FEEDBACK_COL

def store(query:str, response:str, fb:str|None) -> None:
    FEEDBACK_COL.add(
        documents=[f"{query} -> {response}"],
        ids=[str(uuid.uuid4())],
        metadatas=[{
            "user_query": query,
            "response": response,
            "feedback": fb or "None",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }]
    )

def positive_examples(query:str, n:int=3) -> str:
    res = FEEDBACK_COL.query(query_texts=[query], where={"feedback":"positive"}, n_results=n)
    if not res["ids"]: return "No relevant positive feedback examples found."
    out=[]
    for i in range(len(res["ids"][0])):
        meta=res["metadatas"][0][i]
        out.append(f"Example {i+1}:\nQuery: {meta['user_query']}\nResponse: {meta['response']}\n")
    return "\n".join(out)
