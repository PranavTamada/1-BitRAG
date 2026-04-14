import ollama
import time
def run_cheap_llm(prompt):
    start = time.time()

    response = ollama.chat(
        model='tinyllama',
        messages=[{"role": "user", "content": prompt}]
    )

    latency = time.time() - start
    return response['message']['content'], latency