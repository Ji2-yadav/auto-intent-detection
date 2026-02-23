# Autonomous Active Learning Intent Engine

This project is a fully autonomous pipeline that discovers, labels, and refines a robust taxonomy of user intents from raw, unlabeled query data. It uses Large Language Models (LLMs) paired with clustering and machine learning to categorize thousands of utterances with zero human-in-the-loop intervention.

## Quick Start

The system runs entirely via the `intent_pipeline.py` script. The process is divided into 4 sequential steps.

### Prerequisites

1. Ensure you have your `GEMINI_API_KEY` set in your environment variables.
2. Place your raw input data at `train-raw.jsonl` (format: `{"en_query": "utterance text"}`).

### Commands

**1. Bootstrap Phase**

```bash
python intent_pipeline.py bootstrap
```

_What it does:_ Loads raw data, generates SBERT embeddings, runs UMAP dimensionality reduction, and clusters the data with HDBSCAN. The LLM evaluates each cluster to propose the initial Intent Taxonomy (`data/taxonomy_v1.json`) and assigns an initial confident label to the center of each cluster.

**2. Active Learning Loop**

```bash
python intent_pipeline.py loop
```

_What it does:_ The core of the autonomous refinement. Runs an iterative loop (default 30 iterations). In each loop, a Logistic Regression classifier evaluates the dataset and flags 40 queries that are highly uncertain, highly diverse (using FAISS), or where the machine learning model completely disagrees with the initial cluster label. The LLM reviews these challenging queries individually, fixes their labels, or proposes brand-new intents to the taxonomy if there are gaps.

**3. Status & Metrics**

```bash
python intent_pipeline.py status
```

_What it does:_ Prints the current taxonomy size, label distribution, and the ratio of raw cluster labels vs. LLM-verified labels.

**4. Final Export**

```bash
python intent_pipeline.py export --output labeled-intents.jsonl
```

_What it does:_ Trains the multi-label classifier one final time, applying a 10x oversample weight to the expert LLM-reviewed data. It serializes the trained weights to `data/intent_model.pkl`, and applies **Label Propagation** mathematically correcting the labels for every single raw, unreviewed bootstrap item in the dataset based on the learned boundaries.

**5. Live Inference**

```bash
python infer.py
# Or for a single query: python infer.py "is there traffic today and do I need an umbrella"
```

_What it does:_ Instead of training on-the-fly, this script instantly loads the pre-trained `intent_model.pkl` weights into memory. It embeds your custom queries, Normalizes them, and outputs live classification probabilities.

---

## How it Works (Under the Hood)

The engine relies on four mathematical pillars rather than hardcoded prompt rules:

1. **Anti-Mega-Cluster Bootstrapping:**
   Instead of clustering the raw 384d SBERT embeddings directly (which often collapses into just 2 or 3 giant, meaningless clusters), the pipeline first applies a 15-dimensional UMAP reduction. HDBSCAN then groups these, resulting in highly precise, fine-grained semantic clusters.
2. **Disagreement & Diversity Sampling:**
   The active learning loop doesn't just ask the LLM to randomly label data. It uses "Core-Set" sampling (FAISS) to ensure the LLM reviews points spread evenly across the entire semantic space. Furthermore, it explicitly searches for **Disagreements**—moments where the logistic regression boundary wildly contradicts the original HDBSCAN cluster. These are guaranteed to be anomalies or misclassifications, and the LLM is forced to review them.
3. **Targeted Label Propagation (Model Export):**
   It is too expensive to have the LLM review all 3,000 queries. Instead, the final export trains a lightweight model that heavily trusts the small subset of queries the LLM _did_ review. It saves those boundaries to `intent_model.pkl` to be used for instant inference, and mechanically corrects the unreviewed export dataset.
4. **Dynamic Relative Thresholding (Multi-Label Ambiguity):**
   If a user asks a compound question (e.g. "what should I wear today, is it raining?"), the classifier does not strictly fall back to a 0.5 binary limit. It uses a dynamic relative threshold (`max_prob * 0.7`) to mathematically detect when two different intents are competing tightly, allowing the pipeline to seamlessly return multiple intents for ambiguous sentences without arbitrarily cutting them off.

---

## Hidden Flaws & Potential Improvements

Because the system is fully autonomous, there are structural trade-offs:

1. **Taxonomy Churn:**
   During the active learning loop, if the LLM encounters a query that doesn't fit exactly, it has the power to dynamically invent a new intent (e.g. creating `GET_HOURLY_FORECAST` instead of using `GET_CURRENT_WEATHER`). While this fixes ambiguity, it can lead to taxonomy bloat if the LLM invents extremely niche intents that only fit 1 or 2 queries.
   - _Improvement:_ Add an intermediate "Taxonomy Consolidation" phase triggered every 5 active learning iterations to merge overly specific intents back together using hierarchical constraints.
2. **"UNCATEGORIZED" Fallback:**
   Early in the process, highly disjoint noise points are labeled `UNCATEGORIZED`. While label propagation usually fixes this, if an utterance is truly bizarre or out-of-domain (e.g., gibberish typed by a user), label propagation might forcibly map it to the closest intent rather than correctly rejecting it as out-of-domain.
   - _Improvement:_ Train an explicit Out-Of-Domain (OOD) decision boundary based on a cosine distance threshold from the active learning cluster centroids.
3. **Rate Limits & API Constraints:**
   The pipeline implements an intelligent Model Fallback Chain (switching from `gemini-2.0-flash` to `gemma-3-27b-it` to `gemini-2.5-flash-lite`) if an API rate-limit (429) is hit. However, free-tier limits mean the loop takes several minutes to process deep ambiguity.
   - _Improvement:_ Transition to a local vLLM endpoint (e.g. running a quantized Llama 3 8B locally) to allow the active learning loop to process thousands of queries instantly without any API delays.

---

## Testing Multi-Label Inference

You can drop into the interactive inference shell (`python infer.py`) and test boundary cases. Because of the **Dynamic Relative Threshold**, try queries that inherently blend two concepts.

**Examples to try:**

- `"how do i file a claim for my car accident and what is my policy number"`
  _(Should trigger competing probabilities allowing both `INITIATE_CLAIM` and `VIEW_POLICY_DETAILS` to surface)_
- `"is my house covered for flood damage and how much is my premium"`
- `"can I add my daughter to my auto insurance and update my billing address"`
- `"check my claim status and tell me when my next payment is due"`

> **Note:** SBERT naturally maps large compound sentences to the "stronger" structural action. If an intent is swallowed by another (e.g., returning only `CREATE_ALARM`), it means the `intent_pipeline.py loop` simply needs to run for more active learning iterations so the LLM can encounter these boundaries and mathematically separate them for the Logistic Regression model!
