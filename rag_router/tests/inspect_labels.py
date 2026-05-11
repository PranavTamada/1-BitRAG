"""Quick script to inspect the labeled routing data."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

path = "data/labeled_routing_data.jsonl"
lines = open(path, encoding="utf-8").readlines()
print(f"=== Labeled Data: {len(lines)} samples ===\n")

# Label distribution
labels = [json.loads(l)["cheap_succeeds"] for l in lines]
print(f"Cheap succeeds: {sum(labels)}/{len(labels)} ({sum(labels)/len(labels):.0%})")

# Sample record
rec = json.loads(lines[0])
print(f"\n--- Sample Record ---")
print(f"Query:           {rec['query'][:80]}...")
print(f"Cheap BERTScore: {rec['cheap_bertscore']:.4f}")
print(f"Full BERTScore:  {rec['full_bertscore']:.4f}")
print(f"Cheap Succeeds:  {rec['cheap_succeeds']}")
print(f"Retrieval feats: {list(rec['retrieval_features'].keys())}")
print(f"Query feats:     {list(rec['query_features'].keys())}")

# Score distribution
cheap_scores = [json.loads(l)["cheap_bertscore"] for l in lines]
full_scores = [json.loads(l)["full_bertscore"] for l in lines]
print(f"\n--- Score Summary ---")
print(f"Cheap BERTScore: mean={sum(cheap_scores)/len(cheap_scores):.4f}, "
      f"min={min(cheap_scores):.4f}, max={max(cheap_scores):.4f}")
print(f"Full BERTScore:  mean={sum(full_scores)/len(full_scores):.4f}, "
      f"min={min(full_scores):.4f}, max={max(full_scores):.4f}")
