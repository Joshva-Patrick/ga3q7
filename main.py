from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# Load .env
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

app = FastAPI()


class Problem(BaseModel):
    problem_id: str
    problem: str


@app.post("/")
def solve(data: Problem):

    prompt = f"""
Solve the following word problem carefully.

Return ONLY a valid JSON object.

The JSON must contain EXACTLY these two keys:

{{
    "reasoning": "Brief step-by-step reasoning",
    "answer": number
}}

Rules:
- No markdown
- No ```json
- No extra keys
- Answer must be numeric whenever possible

Problem:
{data.problem}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are an expert mathematical reasoning assistant. Always return valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    result = json.loads(response.choices[0].message.content)

    return {
        "reasoning": result["reasoning"],
        "answer": result["answer"]
    }
