from datasets import load_dataset
import json

def build_hotpotqa_dataset(n=200, max_docs_per_sample=5):
    dataset = load_dataset("hotpot_qa", "fullwiki", split=f"train[:{n}]")

    data = []

    for item in dataset:
        query = item["question"]
        answer = item["answer"]

        # context format: (titles, sentences)
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]

        docs = []

        for title, sent_list in zip(titles, sentences):
            text = " ".join(sent_list).strip()

            if len(text) > 50:  # filter useless tiny docs
                docs.append(text)

            if len(docs) >= max_docs_per_sample:
                break

        data.append({
            "query": query,
            "answer": answer,
            "docs": docs   # IMPORTANT: keep list, not merged text
        })

    return data


def save_dataset(data, path="../data/hotpot_clean.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    data = build_hotpotqa_dataset(200)
    save_dataset(data)
    print("✅ Clean HotpotQA dataset created")