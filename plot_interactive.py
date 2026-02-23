import numpy as np
import json
import os
import pandas as pd
import plotly.express as px
import umap

def main():
    embeddings_path = "data/embeddings.npy"
    if not os.path.exists(embeddings_path):
        print(f"Error: {embeddings_path} not found.")
        return
        
    embeddings = np.load(embeddings_path)
    
    # Load utterances to show on hover
    utterances = []
    with open("data/utterances.jsonl", "r") as f:
        for line in f:
            u = json.loads(line)
            utterances.append(u.get("text", "Unknown"))
            
    # Load labels
    annotations_path = "data/annotations.jsonl"
    annotations = {}
    if os.path.exists(annotations_path):
        with open(annotations_path, "r") as f:
            for line in f:
                a = json.loads(line)
                intent_list = a.get("predicted_intents", a.get("labels", ["UNCATEGORIZED"]))
                u_id = a.get("utterance_id", a.get("id"))
                if u_id is not None:
                    annotations[u_id] = intent_list[0] if len(intent_list) > 0 else "UNCATEGORIZED"
                    
    y = []
    for i in range(len(embeddings)):
        y.append(annotations.get(i, "UNCATEGORIZED"))
        
    print("Running UMAP dimensionality reduction (creates tighter clusters than t-SNE)...")
    # UMAP preserves global structure and local density much better for sentence embeddings
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "Intent": y,
        "Query": utterances[:len(embeddings_2d)]
    })
    
    print("Generating interactive HTML plot with Plotly...")
    fig = px.scatter(
        df, 
        x="x", 
        y="y", 
        color="Intent", 
        hover_data=["Query"],
        title="Interactive UMAP Visualization of Intent Training Data",
        width=1600, 
        height=1000
    )
    
    fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=0.5, color='white')))
    
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
            itemwidth=30
        ),
        margin=dict(r=400) # give generous space for legend on the right
    )
    
    out_file = "clusters_interactive.html"
    fig.write_html(out_file)
    print(f"Saved interactive plot to {out_file} (double-click to expand in browser!)")

if __name__ == '__main__':
    main()
