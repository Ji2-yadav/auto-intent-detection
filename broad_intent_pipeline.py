import json, sys, os, re, time, hashlib
import numpy as np
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from google import genai
import warnings

warnings.filterwarnings('ignore')

GEMINI_API_KEY = "AIzaSyDuxrBFwZt5el0WYuMtSQbO5dGvTx-zq8E"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
N_CLUSTERS = 10
ACTIVE_ITERATIONS = 4
LOW_CONF_SAMPLE = 25

def gemini_call(client, prompt, retries=3):
    for i in range(retries):
        try:
            resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            return resp.text.strip()
        except Exception as e:
            if "429" in str(e) or "503" in str(e):
                time.sleep(5)
            else:
                print(f"Gemini error: {e}")
    return None

def main():
    print("🚀 Starting ML-Driven Broad Intent DiscoveryPipeline 🚀\n")
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # 1. Load Data
    print("Step 1: Loading unique utterances from train-main.jsonl...")
    utterances = []
    seen = set()
    with open("train-main.jsonl", "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip(): continue
            try:
                rec = json.loads(line)
                text = rec.get("en_query", "").strip()
                if not text: continue
                key = hashlib.md5(text.lower().encode()).hexdigest()
                if key not in seen:
                    seen.add(key)
                    utterances.append(text)
            except: pass
    print(f"Loaded {len(utterances)} unique utterances.\n")

    # 2. Embeddings
    print(f"Step 2: Computing embeddings with {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(utterances, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")
    
    # 3. UMAP & Clustering (forcing ~10 clusters)
    print("\nStep 3: Dimensionality reduction and Clustering...")
    reducer = umap.UMAP(n_components=15, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42)
    emb_reduced = reducer.fit_transform(embeddings)
    
    clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS, metric='euclidean', linkage='ward')
    cluster_labels = clustering.fit_predict(emb_reduced)
    
    # Calculate Silhoutte Score
    sil_score = silhouette_score(emb_reduced, cluster_labels)
    print(f"Formed {N_CLUSTERS} clusters. Silhouette Score: {sil_score:.3f}\n")
    
    # 4. Extract samples and ask LLM to name the clusters
    print("Step 4: Naming clusters using Gemini...")
    broad_intents = {}
    cluster_names = {}
    
    for cid in range(N_CLUSTERS):
        mask = np.where(cluster_labels == cid)[0]
        centroid = embeddings[mask].mean(axis=0)
        dists = np.linalg.norm(embeddings[mask] - centroid, axis=1)
        nearest_idx = dists.argsort()[:10]
        samples = [utterances[mask[i]] for i in nearest_idx]
        
        # TF-IDF for keywords
        cluster_texts = [utterances[idx] for idx in mask]
        try:
            vec = TfidfVectorizer(max_features=10, ngram_range=(1,2))
            vec.fit(cluster_texts)
            terms = vec.get_feature_names_out().tolist()
        except:
            terms = []
            
        prompt = f"""
You are defining high-level, BROAD intent categories for an insurance virtual assistant.
Here is a cluster of user queries representing a single broad topic.
Keywords: {', '.join(terms)}
Samples:
{chr(10).join(['- ' + s for s in samples[:8]])}

Based on these, propose exactly ONE broad, high-level intent name in UPPER_SNAKE_CASE (e.g., SUBMIT_CLAIM, MANAGE_POLICY, INQUIRE_STATUS). 
DO NOT output anything else except the intent name. Keep it broad and highly representative.
"""
        name = gemini_call(client, prompt)
        if name:
            name = re.sub(r'[^A-Z_]', '', name.upper().strip())
        if not name or len(name) < 3:
            name = f"BROAD_INTENT_{cid}"
            
        cluster_names[cid] = name
        print(f"  Cluster {cid:2d} ({len(mask):3d} items) -> {name}")
        
    # Assign initial labels
    labels = np.array([cluster_names[c] for c in cluster_labels])
    
    # 5. Active Learning Loop to refine boundaries
    print("\nStep 5: Active Learning / Reclassification Loop...")
    
    for iteration in range(ACTIVE_ITERATIONS):
        print(f"\n--- Iteration {iteration+1}/{ACTIVE_ITERATIONS} ---")
        
        # Train classifier
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
        clf.fit(embeddings, labels)
        
        preds = clf.predict(embeddings)
        probs = clf.predict_proba(embeddings)
        confidences = np.max(probs, axis=1)
        
        # Find low confidence items that are NOT already manually reviewed in this session
        # (We will just take the lowest conf items in each iteration)
        low_conf_indices = np.argsort(confidences)[:LOW_CONF_SAMPLE]
        
        avg_conf = np.mean(confidences)
        print(f"  Classifier trained. Average confidence: {avg_conf:.4f}")
        print(f"  Extracting top {LOW_CONF_SAMPLE} lowest confidence queries for LLM review...")
        
        curr_classes = list(set(labels))
        classes_str = ", ".join(curr_classes)
        
        changed_count = 0
        for idx in low_conf_indices:
            text = utterances[idx]
            current_label = labels[idx]
            conf = confidences[idx]
            
            prompt = f"""
Here are the current {len(curr_classes)} broad intent categories: {classes_str}.
Classify this user query into the BEST matching broad category from the list above. 
If it absolutely does not fit ANY of them, you may propose a NEW broad UPPER_SNAKE_CASE category.

Query: "{text}"

Output ONLY the category name. No explanations.
"""
            new_label = gemini_call(client, prompt)
            if new_label:
                new_label = re.sub(r'[^A-Z_]', '', new_label.upper().strip())
                if new_label and new_label != current_label:
                    labels[idx] = new_label
                    changed_count += 1
                    
        print(f"  LLM reclassified {changed_count}/{LOW_CONF_SAMPLE} queries to fix ambiguous boundaries.")
        print(f"  Current Intent Categories ({len(curr_classes)}): {classes_str}")
        
    # Final Training & Metrics
    print("\nStep 6: Finalizing Model & Metrics...")
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(embeddings, labels)
    final_probs = clf.predict_proba(embeddings)
    final_confidences = np.max(final_probs, axis=1)
    
    print(f"  Final Average Confidence: {np.mean(final_confidences):.4f}")
    final_unique = set(labels)
    print(f"  Final Broad Intents ({len(final_unique)}):")
    for b_intent in sorted(list(final_unique)):
        count = sum(labels == b_intent)
        print(f"    - {b_intent} ({count} items)")
        
    # Save output
    out_file = "broad-labeled-intents.jsonl"
    with open(out_file, "w", encoding="utf-8-sig") as f:
        for i, text in enumerate(utterances):
            rec = {
                "query": text,
                "broad_intent": labels[i],
                "confidence": float(final_confidences[i])
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n✅ Saved {len(utterances)} records to {out_file}")
    
    # 7. t-SNE Plot
    print("\nStep 7: Generating t-SNE plot...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(16, 12))
    unique_labels = sorted(list(set(labels)))
    palette = sns.color_palette("husl", len(unique_labels))
    
    sns.scatterplot(
        x=emb_2d[:, 0], 
        y=emb_2d[:, 1],
        hue=labels,
        palette=palette,
        hue_order=unique_labels,
        alpha=0.8,
        s=60,
        edgecolor='w',
        linewidth=0.5
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.title("t-SNE Visualization of Broad Intent Clusters", fontsize=18)
    plt.tight_layout()
    plot_file = "broad_tsne_clusters.png"
    plt.savefig(plot_file, dpi=300)
    print(f"✅ Saved plot to {plot_file}")

if __name__ == "__main__":
    main()
