import google.generativeai as genai
import time

<<<<<<< HEAD
genai.configure(api_key="API_KEY")
=======
genai.configure(api_key="YOUR_GEMINI_API_KEY")
>>>>>>> 84feb76983a36598137e54192204b9242e91b9d7

model = genai.GenerativeModel("gemini-2.5-flash-lite")

def run_full_llm(prompt):
    start = time.time()

    response = model.generate_content(prompt)

    latency = time.time() - start

    return response.text, latency
