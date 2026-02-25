import json
from google import genai

GEMINI_API_KEY = "AIzaSyDuxrBFwZt5el0WYuMtSQbO5dGvTx-zq8E"
client = genai.Client(api_key=GEMINI_API_KEY)

queries = []
with open("broad-labeled-intents.jsonl", "r", encoding="utf-8-sig") as f:
    for line in f:
        data = json.loads(line)
        if data["broad_intent"] == "WEB_SERVICE_INQUIRY":
            queries.append(data["query"])

prompt = "Translate all of the following Japanese queries into natural English. Provide just a numbered list of the English translations, maintaining the order.\n\n"
for i, q in enumerate(queries, 1):
    prompt += f"{i}. {q}\n"

try:
    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=prompt
    )
    print(response.text)
except Exception as e:
    print(f"Error: {e}")
