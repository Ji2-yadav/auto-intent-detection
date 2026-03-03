import os
from google import genai
from dotenv import load_dotenv

from dotenv import load_dotenv

load_dotenv(override=True)
import os
print("API KEY PRESENT:", bool(os.environ.get("GEMINI_API_KEY")))
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-3-flash-preview"]

for model in models:
    try:
        print(f"Testing {model}...")
        resp = client.models.generate_content(model=model, contents="Hello!")
        print(f"Success! Response: {resp.text.strip()}")
    except Exception as e:
        print(f"Error ({type(e).__name__}): {e}")
