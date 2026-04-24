import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv('.env')

print(f"Key loaded: {bool(os.getenv('SPARK_API_KEY'))}")

client = OpenAI(
    api_key=os.getenv("SPARK_API_KEY"),
    base_url="https://api.cerebras.ai/v1"
)

try:
    response = client.chat.completions.create(
        model="llama3.3-70b",
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=10
    )
    print("Success:", response.choices[0].message.content)
except Exception as e:
    import traceback
    traceback.print_exc()
