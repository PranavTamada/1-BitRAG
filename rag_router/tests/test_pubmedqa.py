"""Quick test: verify PubMedQA loader works."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders import load_pubmedqa
d = load_pubmedqa(max_samples=3)
print(f"Loaded {len(d)} samples")
for i, s in enumerate(d):
    print(f"\n  Sample {i+1}:")
    print(f"  Q: {s['query'][:100]}...")
    print(f"  A: {s['answer'][:100]}...")
    print(f"  Dataset: {s['dataset']}")
print("\n[OK] PubMedQA loader works!")
