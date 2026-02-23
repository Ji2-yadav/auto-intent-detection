import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import argparse
import os

def main():
    # Load embeddings
    embeddings_path = "data/embeddings.npy"
    if not os.path.exists(embeddings_path):
        print(f"Error: {embeddings_path} not found.")
        return
        
    embeddings = np.load(embeddings_path)
    
    # Load annotations to get the cluster labels
    labels = []
    annotations_path = "data/annotations.jsonl"
    
    annotations = {}
    if os.path.exists(annotations_path):
        with open(annotations_path, "r") as f:
            for line in f:
                a = json.loads(line)
                # handle if it's new-style export or old-style active-learning
                if "predicted_intents" in a:
                    intent_list = a["predicted_intents"]
                elif "labels" in a:
                    intent_list = a["labels"]
                else:
                    intent_list = ["UNCATEGORIZED"]
                    
                u_id = a.get("utterance_id", a.get("id"))
                
                # If we don't have an ID in the JSON (e.g., exported without ID), 
                # we might just fall back to standard list ordering if the length matches.
                if u_id is not None:
                    annotations[u_id] = intent_list
    
    # Utterances IDs should correspond to the indices of embeddings
    y = []
    for i in range(len(embeddings)):
        if i in annotations and len(annotations[i]) > 0:
            y.append(annotations[i][0])
        else:
            y.append("UNCATEGORIZED")
            
    print(f"Loaded {len(embeddings)} embeddings and {len(y)} labels")
    
    # Run tSNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plotting
    print("Generating plot...")
    plt.figure(figsize=(16, 12))
    
    # Create a scatter plot colored by label
    unique_labels = sorted(list(set(y)))
    palette = sns.color_palette("husl", len(unique_labels))
    
    sns.scatterplot(
        x=embeddings_2d[:, 0], 
        y=embeddings_2d[:, 1],
        hue=y,
        palette=palette,
        hue_order=unique_labels,
        legend="full",
        alpha=0.7,
        s=50
    )
    
    # Format legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=9, ncol=2)
    plt.title("t-SNE Visualization of Intent Training Data", fontsize=18)
    plt.tight_layout()
    plt.savefig("tsne_clusters.png", dpi=300)
    print("Saved plot to tsne_clusters.png")

if __name__ == "__main__":
    main()
