import json
import numpy as np

def convert(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def log_results(results, file_path="logs.json"):
    with open(file_path, "a") as f:
        for r in results:
            r_clean = json.loads(json.dumps(r, default=convert))
            f.write(json.dumps(r_clean) + "\n")