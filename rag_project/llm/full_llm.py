from groq import Groq
import time

# Initialize client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"  

def run_full_llm(prompt):
    start = time.time()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    latency = time.time() - start

    return response.choices[0].message.content, latency