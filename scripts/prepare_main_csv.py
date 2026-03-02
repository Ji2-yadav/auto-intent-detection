import pandas as pd
import json
import csv

def main():
    try:
        df = pd.read_csv('data/main.csv', sep='\t')
    except Exception as e:
        df = pd.read_csv('data/main.csv')
    
    if 'question_name' not in df.columns:
        print("Error: 'question_name' column not found in main.csv. Columns found:", df.columns)
        return
        
    queries = df['question_name'].dropna().unique()
    
    with open('data/train-main.jsonl', 'w', encoding='utf-8') as f:
        for q in queries:
            q_str = str(q).strip()
            if q_str:
                f.write(json.dumps({"en_query": q_str}, ensure_ascii=False) + '\n')
    
    print(f"Successfully wrote {len(queries)} queries to data/train-main.jsonl")

if __name__ == '__main__':
    main()
