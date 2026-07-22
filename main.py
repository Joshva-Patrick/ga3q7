from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Read Groq API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise Exception("GROQ_API_KEY not found. Create a .env file.")

# Groq Client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

app = FastAPI()


class Problem(BaseModel):
    problem_id: str
    problem: str


@app.post("/")
def solve(data: Problem):
    prompt = f"""
Solve the following word problem carefully.
Return ONLY the final numerical answer.
Do not include any explanation.

Problem:
{data.problem}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an expert mathematical reasoning assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content.strip()

    return {
        "problem_id": data.problem_id,
        "answer": answer
    }