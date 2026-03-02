import argparse
import json
from pathlib import Path
import numpy as np
import joblib

# We can import internal data loading logic from intent_pipeline
from intent_pipeline import EMBED_MODEL

# We need sentence-transformers to embed the unseen utterance
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

class IntentInferencer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.embed_model = None
        self.clf = None
        self.mlb = None
        
        self._initialize_model()

    def _initialize_model(self):
        """Load trained label propagation classifier from disk."""
        print("Initializing inference engine...")
        
        model_path = self.data_dir / "intent_model.pkl"
        if not model_path.exists():
            print(f"Error: {model_path} not found. Please run 'python intent_pipeline.py export' to train and save the model.")
            return

        print("  Loading trained model (mlb and clf)...")
        model_data = joblib.load(model_path)
        self.mlb = model_data["mlb"]
        self.clf = model_data["clf"]
        
        print("  Loading SBERT embedding model (this may take a few seconds)...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        print("✅ Ready for inference.\n")

    def predict(self, query):
        """Embed a new string and predict its intents."""
        if self.clf is None or self.mlb is None or self.embed_model is None:
            return None, 0.0

        # 1. Embed the raw string to 384d vector
        emb = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")

        # 2. Predict probabilities using the trained OneVsRest classifier
        preds_probs = self.clf.predict_proba(emb)
        
        # Extract probabilities cleanly
        probs_for_i = []
        for j in range(len(self.mlb.classes_)):
            prob = preds_probs[j][0, 1] if isinstance(preds_probs, list) else preds_probs[0, j]
            probs_for_i.append(float(prob))
            
        # 3. Dynamic Relative Thresholding for Multi-Intent Ambiguity
        # Include any intent that is within 30% of the top intent's probability (minimum floor of 0.15)
        max_prob = max(probs_for_i) if probs_for_i else 0.0
        threshold = max(0.15, max_prob * 0.3)
        
        pred_labels = []
        pred_conf_vals = []
        for j, prob in enumerate(probs_for_i):
            if prob >= threshold:
                pred_labels.append(self.mlb.classes_[j])
                pred_conf_vals.append(prob)

        # 4. Fallback if absolutely no intent crossed 0.15 floor
        if not pred_labels:
            max_idx = np.argmax(probs_for_i)
            pred_labels = [self.mlb.classes_[max_idx]]
            pred_conf = float(probs_for_i[max_idx])
        else:
            # Sort highest confidence first
            sorted_labels = sorted(zip(pred_labels, pred_conf_vals), key=lambda x: -x[1])
            pred_labels = [l for l, c in sorted_labels]
            pred_conf = float(np.mean(pred_conf_vals))

        return pred_labels, pred_conf

def main():
    parser = argparse.ArgumentParser(description="Infer intents for new phrases using the trained active learning model.")
    parser.add_argument("query", nargs="?", help="Text to classify. If omitted, starts interactive mode.")
    args = parser.parse_args()

    inferencer = IntentInferencer()

    if args.query:
        labels, conf = inferencer.predict(args.query)
        print(f"Query:      \"{args.query}\"")
        print(f"Intents:    {labels}")
        print(f"Confidence: {conf:.4f}")
    else:
        print("--- Interactive Mode --- (Type 'exit' to quit)")
        while True:
            try:
                q = input("\nEnter query: ")
                if not q.strip(): continue
                if q.lower() in ("exit", "quit"): break
                
                labels, conf = inferencer.predict(q)
                print(f"Intents:    {labels}")
                print(f"Confidence: {conf:.4f}")
            except KeyboardInterrupt:
                break
            except EOFError:
                break

if __name__ == "__main__":
    main()
