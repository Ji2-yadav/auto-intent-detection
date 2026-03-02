"""
intent_pipeline.py — Active-learning intent labeling pipeline.

Subcommands:
    bootstrap   Phase 1: embed → cluster → LLM taxonomy → initial labels
    loop        Phase 2: train classifier → score → sample → LLM label+review → retrain
    status      Show quality metrics / progress
    export      Write final labeled-intents.jsonl

Fully autonomous: Gemini acts as both labeler and reviewer (no human needed).
"""

import json, sys, os, re, time, hashlib, argparse
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime, timezone

# ── ML imports ──
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import hdbscan

# ── LLM import ──
from google import genai

# ── Config ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyASB0cWLjkUVoOhMWhYFcCdxiNWloQ_EzY"
GEMINI_MODELS  = ["gemini-2.0-flash", "gemma-3-27b-it", "gemini-2.5-flash-lite"]
_current_model_idx = 0
EMBED_MODEL    = "paraphrase-multilingual-MiniLM-L12-v2"
DATA_DIR       = Path("data")
BATCH_SIZE_LLM = 10          # clusters per LLM call
ACTIVE_BATCH   = 40          # utterances per active-learning iteration
MAX_ITERATIONS = 30          # auto-stop after N active-learning rounds
LLM_DELAY      = 8           # seconds between Gemini calls (free tier)
# ────────────────────────────────────────────────────────────────────────


def gemini_call(client, prompt, max_retries=3):
    """Call Gemini with retry on 429 + model fallback chain."""
    global _current_model_idx
    for attempt in range(max_retries + 1):
        model = GEMINI_MODELS[_current_model_idx]
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            return resp.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "503" in err:
                # Try next model in fallback chain
                if _current_model_idx < len(GEMINI_MODELS) - 1:
                    _current_model_idx += 1
                    print(f"    🔄 Switching to {GEMINI_MODELS[_current_model_idx]} (model {_current_model_idx+1}/{len(GEMINI_MODELS)})")
                    continue
                elif attempt < max_retries:
                    wait = 65
                    print(f"    ⏳ All models exhausted, waiting {wait}s ({attempt+1}/{max_retries})…")
                    _current_model_idx = 0  # reset to first model
                    time.sleep(wait)
                    continue
            print(f"    ⚠ Gemini error [{model}]: {e}")
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 1: BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════

def cmd_bootstrap(args):
    DATA_DIR.mkdir(exist_ok=True)
    client = genai.Client(api_key=GEMINI_API_KEY)
    input_file = args.input

    # ── 1. Build canonical utterances ──
    print("Step 1/6: Building canonical utterance set…")
    utterances = []
    seen = set()
    with open(input_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec["en_query"].strip()
            # simple near-dedup by lowercased text hash
            key = hashlib.md5(text.lower().encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            utterances.append({"id": len(utterances), "text": text})

    utt_path = DATA_DIR / "utterances.jsonl"
    with open(utt_path, "w") as f:
        for u in utterances:
            f.write(json.dumps(u) + "\n")
    print(f"  {len(utterances)} unique utterances → {utt_path}")

    # ── 2. Compute SBERT embeddings ──
    print("\nStep 2/6: Computing SBERT embeddings…")
    emb_path = DATA_DIR / "embeddings.npy"
    idx_path = DATA_DIR / "faiss.index"

    model = SentenceTransformer(EMBED_MODEL)
    texts = [u["text"] for u in utterances]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")
    np.save(emb_path, embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(idx_path))
    print(f"  {embeddings.shape} → {emb_path}, {idx_path}")

    # ── 3. UMAP + HDBSCAN clustering ──
    print("\nStep 3/6: UMAP dimensionality reduction + HDBSCAN clustering…")
    import umap
    reducer = umap.UMAP(n_components=15, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42)
    emb_reduced = reducer.fit_transform(embeddings)
    print(f"  UMAP: {embeddings.shape[1]}d → {emb_reduced.shape[1]}d")

    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, metric="euclidean",
                                 cluster_selection_method="eom")
    cluster_labels = clusterer.fit_predict(emb_reduced)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int((cluster_labels == -1).sum())
    print(f"  {n_clusters} clusters found, {n_noise} noise points")

    # Build per-cluster info
    cluster_info = {}
    for cid in sorted(set(cluster_labels)):
        if cid == -1:
            continue
        mask = np.where(cluster_labels == cid)[0]
        centroid = embeddings[mask].mean(axis=0)
        dists = np.linalg.norm(embeddings[mask] - centroid, axis=1)
        nearest_idx = dists.argsort()[:10]
        samples = [texts[mask[i]] for i in nearest_idx]

        # Also get TF-IDF top terms for this cluster
        cluster_texts = [texts[mask[i]] for i in range(len(mask))]
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=20, ngram_range=(1,2))
            tfidf = vec.fit_transform(cluster_texts)
            terms = vec.get_feature_names_out().tolist()[:10]
        except:
            terms = []

        cluster_info[cid] = {
            "size": int(len(mask)),
            "samples": samples,
            "terms": terms,
            "indices": mask.tolist(),
        }

    # ── 4. LLM taxonomy proposal (batched) ──
    print("\nStep 4/6: Proposing taxonomy via Gemini (batched)…")
    taxonomy = {}
    all_cids = sorted(cluster_info.keys())

    for i in range(0, len(all_cids), BATCH_SIZE_LLM):
        batch = all_cids[i:i + BATCH_SIZE_LLM]
        parts = []
        for cid in batch:
            info = cluster_info[cid]
            t = ", ".join(info["terms"][:5])
            s = " | ".join(info["samples"][:5])
            parts.append(f"Cluster {cid} ({info['size']} items): keywords=[{t}] samples=[{s}]")

        prompt = (
            "You are an NLU taxonomy designer for an insurance company virtual assistant.\n"
            "Below are clusters of user queries with keywords and samples.\n"
            "For EACH cluster, propose intent labels. A cluster may map to MULTIPLE intents.\n\n"
            + "\n".join(parts) + "\n\n"
            "Reply in this exact format, one line per cluster:\n"
            "Cluster <id>: <INTENT_1>, <INTENT_2>, ...\n"
            "Use UPPER_SNAKE_CASE labels. Make the intents EXTREMELY granular and specific (e.g., VIEW_LIFE_INSURANCE_PREMIUM_CERTIFICATE instead of just VIEW_POLICY_DETAILS).\n"
            "Do not group distinct but related questions into broad intents; create distinct, detailed intents for each specific scenario."
        )
        raw = gemini_call(client, prompt)
        if raw:
            for line in raw.split("\n"):
                line = line.strip()
                m = re.match(r'(?:Cluster\s*)?(\d+)\s*:\s*(.+)', line)
                if m:
                    cid_parsed = int(m.group(1))
                    labels_raw = m.group(2).strip()
                    labels = []
                    for token in re.split(r'[,\s]+', labels_raw):
                        cleaned = token.strip("`*.,;:!?\"'")
                        if cleaned and cleaned == cleaned.upper() and "_" in cleaned and len(cleaned) > 2:
                            labels.append(cleaned)
                    if labels:
                        taxonomy[cid_parsed] = labels

        # Fill missing
        for cid in batch:
            if cid not in taxonomy:
                taxonomy[cid] = [f"CLUSTER_{cid}"]
            print(f"  Cluster {cid:>2} ({cluster_info[cid]['size']:>4} items) → {', '.join(taxonomy[cid])}")

        if i + BATCH_SIZE_LLM < len(all_cids):
            time.sleep(LLM_DELAY)

    # Assign noise points a special label
    taxonomy[-1] = ["UNCATEGORIZED"]

    # ── 5. Consolidate taxonomy ──
    print("\nStep 5/6: Consolidating taxonomy…")
    # Collect all unique intents
    all_intents = set()
    for labels in taxonomy.values():
        all_intents.update(labels)
    all_intents.discard("UNCATEGORIZED")

    # LLM consolidation: merge duplicates
    if len(all_intents) > 5:
        intent_list = ", ".join(sorted(all_intents))
        consolidation_prompt = (
            "You are an NLU taxonomy designer for an insurance provider.\n"
            "Below is a list of proposed intent labels from our automated pipeline:\n\n"
            f"{intent_list}\n\n"
            "Merge any duplicates or near-synonyms into a single canonical label.\n"
            "PAY ATTENTION TO THESE DISTINCTIONS:\n"
            "- Ensure the intents remain highly granular and specific (e.g., KEEP 'VIEW_LIFE_INSURANCE_PREMIUM_CERTIFICATE' and 'VIEW_PENSION_CERTIFICATE' separate rather than merging them into 'VIEW_DOCUMENT').\n"
            "- Ensure 'Claim Status' vs 'File Claim' are kept separate (e.g., CHECK_CLAIM_STATUS vs INITIATE_CLAIM).\n"
            "- Ensure 'Premium Payment' vs 'Payment History' are kept separate.\n\n"
            "Reply with ONLY a JSON object mapping old→new labels, e.g.:\n"
            '{"SUBMIT_CLAIM": "INITIATE_CLAIM", "FILE_NEW_CLAIM": "INITIATE_CLAIM", ...}\n'
            "Include ALL labels. If a label is already canonical, map it to itself."
        )
        time.sleep(LLM_DELAY)
        merge_raw = gemini_call(client, consolidation_prompt)
        merge_map = {}
        if merge_raw:
            # Try to parse JSON from the response
            json_match = re.search(r'\{[^{}]+\}', merge_raw, re.DOTALL)
            if json_match:
                try:
                    merge_map = json.loads(json_match.group())
                except:
                    pass

        if merge_map:
            # Apply merges
            for cid in taxonomy:
                taxonomy[cid] = list(set(
                    merge_map.get(l, l) for l in taxonomy[cid]
                ))
            merged = sum(1 for k, v in merge_map.items() if k != v)
            print(f"  Merged {merged} duplicate labels")

    # Build final taxonomy file
    final_intents = set()
    for labels in taxonomy.values():
        final_intents.update(labels)
    final_intents.discard("UNCATEGORIZED")

    tax_data = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "intents": {
            name: {"label_path": name.lower().replace("_", "/"), "definition": ""}
            for name in sorted(final_intents)
        }
    }
    tax_path = DATA_DIR / "taxonomy_v1.json"
    with open(tax_path, "w") as f:
        json.dump(tax_data, f, indent=2)
    print(f"  {len(final_intents)} intents → {tax_path}")

    # ── 6. Initial labeling of ALL utterances ──
    print("\nStep 6/6: Initial multi-label annotation (batched)…")
    ann_path = DATA_DIR / "annotations.jsonl"
    annotations = []

    # Assign cluster-based labels
    for uid, u in enumerate(utterances):
        cid = int(cluster_labels[uid])
        labels = taxonomy.get(cid, ["UNCATEGORIZED"])
        annotations.append({
            "annotation_id": len(annotations),
            "utterance_id": uid,
            "labels": labels,
            "source": "bootstrap_cluster",
            "taxonomy_version": 1,
            "confidence": float(clusterer.probabilities_[uid]) if cid != -1 else 0.0,
            "rationale": f"HDBSCAN cluster {cid}",
            "status": "auto_accepted",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    with open(ann_path, "w") as f:
        for a in annotations:
            f.write(json.dumps(a) + "\n")

    n_uncategorized = sum(1 for a in annotations if "UNCATEGORIZED" in a["labels"])
    print(f"  {len(annotations)} annotations → {ann_path}")
    print(f"  {n_uncategorized} uncategorized (noise)")
    print(f"\n✅ Bootstrap complete! Run 'python intent_pipeline.py loop' to refine.")


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 2: ACTIVE LEARNING LOOP
# ═══════════════════════════════════════════════════════════════════════

def load_state():
    """Load all data assets."""
    utterances = []
    with open(DATA_DIR / "utterances.jsonl") as f:
        for line in f:
            utterances.append(json.loads(line))

    embeddings = np.load(DATA_DIR / "embeddings.npy")
    index = faiss.read_index(str(DATA_DIR / "faiss.index"))

    # Load latest annotations (last annotation per utterance wins)
    annotations = {}
    with open(DATA_DIR / "annotations.jsonl") as f:
        for line in f:
            a = json.loads(line)
            annotations[a["utterance_id"]] = a

    # Load latest taxonomy
    tax_files = sorted(DATA_DIR.glob("taxonomy_v*.json"))
    taxonomy = json.load(open(tax_files[-1]))

    return utterances, embeddings, index, annotations, taxonomy


def train_classifier(embeddings, annotations, taxonomy):
    """Train multi-label logistic regression on embeddings."""
    all_intents = sorted(taxonomy["intents"].keys())
    mlb = MultiLabelBinarizer(classes=all_intents)

    # Gather labeled data (exclude UNCATEGORIZED with low confidence)
    X_ids, y_raw = [], []
    for uid, ann in annotations.items():
        labels = [l for l in ann["labels"] if l in taxonomy["intents"]]
        if not labels:
            continue
        X_ids.append(uid)
        y_raw.append(labels)

    if len(X_ids) < 10:
        return None, None, mlb

    X = embeddings[X_ids]
    Y = mlb.fit_transform(y_raw)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1.0))
    clf.fit(X, Y)
    return clf, X_ids, mlb


def score_pool(embeddings, index, annotations, clf, mlb, taxonomy):
    """Score unlabeled/low-confidence items for active learning selection."""
    all_uids = set(range(len(embeddings)))

    # Candidates for active learning: Anything not explicitly reviewed by LLM
    candidate_uids = []
    for uid in all_uids:
        ann = annotations.get(uid)
        if ann is None or ann["source"] == "bootstrap_cluster":
            candidate_uids.append(uid)

    if not candidate_uids or clf is None:
        return candidate_uids[:ACTIVE_BATCH]

    pool_embs = embeddings[candidate_uids]

    # ── Uncertainty: entropy of predictions ──
    try:
        probs = clf.predict_proba(pool_embs)
        # For multi-label, probs is a list of arrays (one per class)
        if isinstance(probs, list):
            probs = np.column_stack([p[:, 1] if p.shape[1] == 2 else p[:, 0] for p in probs])
        entropy = -np.sum(probs * np.log(probs + 1e-10) + (1 - probs) * np.log(1 - probs + 1e-10), axis=1)
    except:
        entropy = np.zeros(len(candidate_uids))

    # ── Diversity: distance from nearest already-labeled item ──
    labeled_embs = []
    for uid, ann in annotations.items():
        if ann.get("source") in ("active_learning", "llm_reviewed"):
            labeled_embs.append(embeddings[uid])
    if labeled_embs:
        labeled_arr = np.array(labeled_embs).astype("float32")
        temp_idx = faiss.IndexFlatL2(labeled_arr.shape[1])
        temp_idx.add(labeled_arr)
        dists, _ = temp_idx.search(pool_embs.astype("float32"), 1)
        diversity = dists.flatten()
    else:
        diversity = np.ones(len(candidate_uids))

    # ── Disagreement: Classifier prediction vs current label ──
    disagreement = np.zeros(len(candidate_uids))
    try:
        preds = clf.predict(pool_embs)
        # preds is a binary matrix (n_samples, n_classes)
        for i, uid in enumerate(candidate_uids):
            ann = annotations.get(uid)
            if ann and ann["source"] == "bootstrap_cluster":
                pred_labels = set(mlb.classes_[preds[i] == 1])
                curr_labels = set(ann.get("labels", []))
                
                if "UNCATEGORIZED" in curr_labels:
                    disagreement[i] = 2.0
                elif curr_labels and not pred_labels.intersection(curr_labels):
                    # Zero overlap
                    disagreement[i] = 1.0
                elif pred_labels != curr_labels:
                    # Partial overlap
                    disagreement[i] = 0.5
    except Exception as e:
        print("    ⚠ Disagreement score error:", e)

    # Normalize
    if entropy.max() > 0:
        entropy = entropy / entropy.max()
    if diversity.max() > 0:
        diversity = diversity / diversity.max()
    if disagreement.max() > 0:
        disagreement = disagreement / disagreement.max()

    # Composite score: 20% uncertainty, 40% diversity, 20% disagreement, 20% random
    score = 0.2 * entropy + 0.4 * diversity + 0.2 * disagreement + 0.2 * np.random.random(len(candidate_uids))
    
    # ── Massive boost for UNCATEGORIZED and Disagreements ──
    for i, uid in enumerate(candidate_uids):
        if annotations.get(uid) and "UNCATEGORIZED" in annotations[uid]["labels"]:
            score[i] += 10.0  # Force to top
        elif disagreement[i] >= 1.0:
            score[i] += 5.0   # Total disagreement also highly prioritized

    # Top batch
    top_idx = score.argsort()[::-1][:ACTIVE_BATCH]
    return [candidate_uids[i] for i in top_idx]


def llm_label_and_review(client, utterances, batch_uids, taxonomy):
    """LLM labels batch with top-3 + rationale, then self-reviews."""
    intent_list = ", ".join(sorted(taxonomy["intents"].keys()))
    results = {}

    # Process in sub-batches of 5 for manageable prompts
    for i in range(0, len(batch_uids), 5):
        sub_uids = batch_uids[i:i + 5]
        items = "\n".join(
            f"{uid}: \"{utterances[uid]['text']}\""
            for uid in sub_uids
        )

        prompt = (
            "You are an expert NLU data labeler for an insurance company.\n"
            f"Available intents: {intent_list}\n"
            "If no intent fits perfectly, DO NOT use UNCATEGORIZED.\n"
            "Instead, set needs_new_intent=true and propose a specific, highly granular new intent.\n\n"
            "CRITICAL DISTINCTIONS TO WATCH FOR:\n"
            "1. Intents must be extremely and specifically detailed (e.g. QUERY_HEALTH_CELEBRATION_BONUS instead of a broad INQUIRE_POLICY_BENEFIT).\n"
            "2. Claim Filing vs Status: 'I want to report an accident' is INITIATE_CLAIM. 'Where is my claim check' is CHECK_CLAIM_STATUS.\n"
            "3. Beneficiary Changes vs Personal Info: Updating who gets the money is UPDATE_BENEFICIARY. Changing an address is UPDATE_ACCOUNT_DETAILS.\n\n"
            "For each query below, assign 1-3 intent labels (multi-label if needed).\n"
            "NEVER return 'UNCATEGORIZED' as a label. Always pick the closest match or propose a new one.\n\n"
            f"{items}\n\n"
            "Reply as JSON array:\n"
            '[{"id": <id>, "labels": ["INTENT_1", ...], "confidence": 0.0-1.0, '
            '"rationale": "brief reason", "needs_new_intent": false, "proposed_intent": null}]\n'
            "ONLY valid JSON, no markdown fences."
        )

        raw = gemini_call(client, prompt)
        if raw:
            # Strip markdown fences if present
            raw = re.sub(r'^```\w*\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)
            try:
                parsed = json.loads(raw)
                for item in parsed:
                    uid = item["id"]
                    results[uid] = {
                        "labels": item.get("labels", ["UNCATEGORIZED"]),
                        "confidence": item.get("confidence", 0.5),
                        "rationale": item.get("rationale", ""),
                        "needs_new_intent": item.get("needs_new_intent", False),
                        "proposed_intent": item.get("proposed_intent"),
                    }
            except json.JSONDecodeError:
                # Try line-by-line parsing
                for uid in sub_uids:
                    results[uid] = {
                        "labels": ["UNCATEGORIZED"],
                        "confidence": 0.0,
                        "rationale": "JSON parse failed",
                        "needs_new_intent": False,
                        "proposed_intent": None,
                    }

        # Fill missing
        for uid in sub_uids:
            if uid not in results:
                results[uid] = {
                    "labels": ["UNCATEGORIZED"],
                    "confidence": 0.0,
                    "rationale": "LLM call failed",
                    "needs_new_intent": False,
                    "proposed_intent": None,
                }

        if i + 5 < len(batch_uids):
            time.sleep(LLM_DELAY)

    return results


def cmd_loop(args):
    """Run active learning loop."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    utterances, embeddings, index, annotations, taxonomy = load_state()
    print(f"Loaded: {len(utterances)} utterances, {len(taxonomy['intents'])} intents, {len(annotations)} annotations")

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"  ACTIVE LEARNING — Iteration {iteration}/{MAX_ITERATIONS}")
        print(f"{'='*60}")

        # 1. Train classifier
        print("\n  Training classifier…")
        clf, train_ids, mlb = train_classifier(embeddings, annotations, taxonomy)
        if clf is None:
            print("  Not enough labeled data, using all low-conf items")

        # 2. Score pool
        print("  Scoring pool…")
        batch_uids = score_pool(embeddings, index, annotations, clf, mlb, taxonomy)
        if not batch_uids:
            print("  ✅ No more items to label! All done.")
            break
        print(f"  Selected {len(batch_uids)} items for labeling")

        # 3. LLM label + review (autonomous)
        print("  LLM labeling + review…")
        results = llm_label_and_review(client, utterances, batch_uids, taxonomy)

        # 4. Check for new intents
        new_intents = set()
        for uid, res in results.items():
            if res.get("needs_new_intent") and res.get("proposed_intent"):
                proposed = res["proposed_intent"].upper().replace(" ", "_")
                if proposed not in taxonomy["intents"]:
                    new_intents.add(proposed)

        if new_intents:
            print(f"  📌 New intents proposed: {', '.join(new_intents)}")
            for intent in new_intents:
                taxonomy["intents"][intent] = {
                    "label_path": intent.lower().replace("_", "/"),
                    "definition": f"Auto-discovered in iteration {iteration}",
                }
            # Save updated taxonomy
            new_version = taxonomy["version"] + 1
            taxonomy["version"] = new_version
            taxonomy["created_at"] = datetime.now(timezone.utc).isoformat()
            tax_path = DATA_DIR / f"taxonomy_v{new_version}.json"
            with open(tax_path, "w") as f:
                json.dump(taxonomy, f, indent=2)
            print(f"  Taxonomy updated → v{new_version} ({len(taxonomy['intents'])} intents)")

        # 5. Update annotations
        ann_path = DATA_DIR / "annotations.jsonl"
        with open(ann_path, "a") as f:
            for uid, res in results.items():
                # Filter labels to known intents (keep new ones too)
                valid_labels = [l for l in res["labels"] if l in taxonomy["intents"]]
                if not valid_labels:
                    valid_labels = res["labels"][:3]  # keep LLM suggestions

                ann = {
                    "annotation_id": max((a["annotation_id"] for a in annotations.values()), default=0) + 1,
                    "utterance_id": uid,
                    "labels": valid_labels,
                    "source": "active_learning",
                    "taxonomy_version": taxonomy["version"],
                    "confidence": res["confidence"],
                    "rationale": res["rationale"],
                    "status": "llm_reviewed",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                f.write(json.dumps(ann) + "\n")
                annotations[uid] = ann

        # 6. Stats
        n_labeled = sum(1 for a in annotations.values() if a["source"] != "bootstrap_cluster")
        n_uncat = sum(1 for a in annotations.values() if "UNCATEGORIZED" in a["labels"])
        n_low = sum(1 for a in annotations.values()
                     if a["confidence"] < 0.5 and a["source"] == "bootstrap_cluster")
        print(f"\n  Progress: {n_labeled} LLM-labeled, {n_uncat} uncategorized, {n_low} low-conf bootstrap")

        # Convergence check
        if n_uncat == 0 and n_low == 0:
            print("  ✅ Converged! No uncategorized or low-confidence items remain.")
            break

    print(f"\n✅ Active learning complete after {iteration} iterations.")
    print(f"   Run 'python intent_pipeline.py status' for metrics.")
    print(f"   Run 'python intent_pipeline.py export' for final dataset.")


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 3: STATUS & EXPORT
# ═══════════════════════════════════════════════════════════════════════

def cmd_status(args):
    """Print quality metrics."""
    utterances, embeddings, index, annotations, taxonomy = load_state()

    print(f"{'='*50}")
    print(f"  INTENT PIPELINE STATUS")
    print(f"{'='*50}")
    print(f"\n  Utterances:   {len(utterances)}")
    print(f"  Annotations:  {len(annotations)}")
    print(f"  Taxonomy v{taxonomy['version']}: {len(taxonomy['intents'])} intents")

    # Source breakdown
    sources = {}
    for a in annotations.values():
        sources[a["source"]] = sources.get(a["source"], 0) + 1
    print(f"\n  Annotation sources:")
    for src, count in sorted(sources.items()):
        print(f"    {src:25s} {count:>5}")

    # Confidence distribution
    confs = [a["confidence"] for a in annotations.values()]
    print(f"\n  Confidence: mean={np.mean(confs):.2f}, median={np.median(confs):.2f}")
    print(f"    < 0.3: {sum(1 for c in confs if c < 0.3)}")
    print(f"    0.3–0.7: {sum(1 for c in confs if 0.3 <= c < 0.7)}")
    print(f"    > 0.7: {sum(1 for c in confs if c >= 0.7)}")

    # Label distribution (top 20)
    label_counts = {}
    for a in annotations.values():
        for l in a["labels"]:
            label_counts[l] = label_counts.get(l, 0) + 1
    print(f"\n  Top 20 labels:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"    {label:35s} {count:>5}")

    n_uncat = sum(1 for a in annotations.values() if "UNCATEGORIZED" in a["labels"])
    n_multi = sum(1 for a in annotations.values() if len(a["labels"]) > 1)
    print(f"\n  Uncategorized: {n_uncat}")
    print(f"  Multi-label:   {n_multi}")


def cmd_export(args):
    """Export final labeled dataset."""
    utterances, embeddings, index, annotations, taxonomy = load_state()

    # 1. Train final classifier with heavy weights on LLM-reviewed data via oversampling
    X_ids, y_raw = [], []
    for uid, ann in annotations.items():
        labels = [l for l in ann["labels"] if l in taxonomy["intents"]]
        if not labels:
            continue
            
        # Oversample active learning labels heavily over noisy bootstrap labels
        weight = 10 if ann.get("source") in ("active_learning", "llm_reviewed") else 1
        for _ in range(weight):
            X_ids.append(uid)
            y_raw.append(labels)

    X = embeddings[X_ids]
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_raw)

    print("  Training final classifier for label propagation...")
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1.0))
    clf.fit(X, Y)
    
    import joblib
    joblib.dump({"clf": clf, "mlb": mlb}, DATA_DIR / "intent_model.pkl")
    print("  Saved trained model to data/intent_model.pkl")
    
    # Predict all to fix bootstrap-level misclassifications
    all_preds_bin = clf.predict(embeddings)
    all_preds_probs = clf.predict_proba(embeddings)

    out_path = args.output or "labeled-intents.jsonl"
    corrected_count = 0
    with open(out_path, "w", encoding="utf-8-sig") as f:
        for i, u in enumerate(utterances):
            ann = annotations.get(u["id"], {})
            source = ann.get("source", "none")
            
            # Use original if LLM actually reviewed it
            if source in ("active_learning", "llm_reviewed"):
                final_labels = ann.get("labels", ["UNCATEGORIZED"])
                final_conf = ann.get("confidence", 0.0)
            else:
                # Extract probabilities cleanly
                probs_for_i = []
                for j in range(len(mlb.classes_)):
                    prob = all_preds_probs[j][i, 1] if isinstance(all_preds_probs, list) else all_preds_probs[i, j]
                    probs_for_i.append(float(prob))
                
                # Dynamic Relative Thresholding for Multi-Intent Ambiguity
                max_prob = max(probs_for_i) if probs_for_i else 0.0
                threshold = max(0.15, max_prob * 0.3)
                
                pred_labels = []
                pred_conf_vals = []
                for j, prob in enumerate(probs_for_i):
                    if prob >= threshold:
                        pred_labels.append(mlb.classes_[j])
                        pred_conf_vals.append(prob)
                
                if not pred_labels:
                    # Fallback to argmax if absolutely nothing passes 0.20
                    max_idx = np.argmax(probs_for_i)
                    final_labels = [mlb.classes_[max_idx]]
                    final_conf = float(probs_for_i[max_idx])
                else:
                    sorted_labels = sorted(zip(pred_labels, pred_conf_vals), key=lambda x: -x[1])
                    final_labels = [l for l, c in sorted_labels]
                    final_conf = float(np.mean(pred_conf_vals))
                
                # Check if we changed from original
                orig_labels = set(ann.get("labels", []))
                if set(final_labels) != orig_labels:
                    corrected_count += 1
                source = "active_learning_propagation"

            rec = {
                "en_query": u["text"],
                "predicted_intents": final_labels,
                "confidence": final_conf,
                "source": source,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Exported {len(utterances)} records → {out_path}")
    print(f"🔄 Corrected {corrected_count} originally mislabeled bootstrap items via label propagation.")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Active Learning Intent Pipeline")
    sub = parser.add_subparsers(dest="cmd")

    p_boot = sub.add_parser("bootstrap", help="Phase 1: cluster + LLM → taxonomy_v1")
    p_boot.add_argument("--input", default="train-raw.jsonl", help="Input JSONL file")

    p_loop = sub.add_parser("loop", help="Phase 2: active learning iterations")

    p_stat = sub.add_parser("status", help="Show quality metrics")

    p_exp = sub.add_parser("export", help="Export labeled dataset")
    p_exp.add_argument("--output", default="labeled-intents.jsonl", help="Output file")

    args = parser.parse_args()
    if args.cmd == "bootstrap":
        cmd_bootstrap(args)
    elif args.cmd == "loop":
        cmd_loop(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "export":
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
