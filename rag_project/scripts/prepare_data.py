from datasets import load_dataset
import json

def build_hotpotqa_dataset(n=200):
    dataset = load_dataset("hotpot_qa", "fullwiki", split=f"train[:{n}]")

    data = []

    for item in dataset:
        query = item["question"]

        # Combine all context sentences
        context_list = item["context"]["sentences"]
        context = " ".join([" ".join(sent) for sent in context_list])

        data.append({
            "query": query,
            "context": context[:300]  # truncate for speed
        })

    return data


def save_dataset(data, path="../data/dataset.json"):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    data = build_hotpotqa_dataset(200)
    save_dataset(data)
    print("HotpotQA dataset created with 200 samples")