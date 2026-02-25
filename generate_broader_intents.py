import json
import os
from google import genai

# Retrieve API key from environment variable
GEMINI_API_KEY = "AIzaSyDuxrBFwZt5el0WYuMtSQbO5dGvTx-zq8E"

def main():
    if not os.path.exists("train-main.jsonl"):
        print("Error: train-main.jsonl not found.")
        return

    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        return

    client = genai.Client(api_key=GEMINI_API_KEY)
    
    utterances = []
    with open("train-main.jsonl", "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip(): 
                continue
            try:
                rec = json.loads(line)
                # extract query
                text = rec.get("en_query", "").strip()
                if text:
                    utterances.append(text)
            except Exception as e:
                pass
                
    # Deduplicate to save tokens
    unique_utterances = list(set(utterances))
    print(f"Loaded {len(unique_utterances)} unique utterances from train-main.jsonl")
    
    prompt = f"""
You are an expert NLP data analyst and conversational AI designer for an insurance company virtual assistant.
Below are {len(unique_utterances)} user queries in Japanese related to insurance policies, claims, documents, online services, etc.

YOUR TASK:
Analyze the provided user queries and propose a concise list of exactly 9 to 11 BROADER, high-level intent categories that holistically cover all these scenarios.
These should represent high-level conceptual groupings (e.g., instead of 5 different specific intents for updating an address, viewing a policy, or checking a balance, you might group them under a single broad intent like MANAGE_POLICY_DETAILS).

FORMAT REQUIREMENTS:
1. Provide exactly 9 to 11 broad intents.
2. Name each category in UPPER_SNAKE_CASE (e.g., SUBMIT_CLAIM, INQUIRE_POLICY, MANAGE_ACCOUNT).
3. Provide a brief 1-2 sentence description for each category explaining what types of queries fall under it.
4. Provide a few example queries from the data that would belong to that category.

Here are the user queries:
""" + "\n".join(unique_utterances)

    print("Asking Gemini to generate 9-11 broader intents... This may take a minute.")
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt
        )
        
        print("\n================== BROADER INTENTS ==================\n")
        print(response.text)
        print("\n======================================================\n")
        
        out_file = "broad_intents_output.txt"
        with open(out_file, "w", encoding="utf-8-sig") as f:
            f.write(response.text)
        print(f"Successfully saved these broader intents to {out_file}.")
        
    except Exception as e:
        print(f"Error calling Gemini: {e}")

if __name__ == "__main__":
    main()
