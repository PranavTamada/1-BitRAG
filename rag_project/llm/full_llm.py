import google.generativeai as genai
import time

genai.configure(api_key="YOUR_GEMINI_API_KEY")

model = genai.GenerativeModel("gemini-2.5-flash-lite")

def run_full_llm(prompt):
    start = time.time()

    response = model.generate_content(prompt)

    latency = time.time() - start

    return response.text, latency
