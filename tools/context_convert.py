import json

def bio_to_json(input_file, output_file, dataset_name="wnut17"):
    """
    Convert BIO formatted NER dataset (e.g., WNUT17) into JSON array format
    with ltokens/rtokens filled:
      - ltokens = tokens[i-2] if available else []
      - rtokens = tokens[i+1] if available else []
    """

    sentences = []
    tokens, labels = [], []

    # 1. Read BIO format files
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append((tokens, labels))
                    tokens, labels = [], []
            else:
                parts = line.split()
                if len(parts) == 1:
                    token, label = parts[0], "O"
                else:
                    token, label = parts[0], parts[-1]
                tokens.append(token)
                labels.append(label)

    if tokens:
        sentences.append((tokens, labels))

    # 2. BIO → span
    def extract_entities(tokens, labels):
        entities = []
        start, ent_type = None, None
        for i, label in enumerate(labels):
            if label.startswith("B-"):
                if start is not None:
                    entities.append({"type": ent_type, "start": start, "end": i})
                start = i
                ent_type = label[2:].upper()  # 转大写，和 types.json 对齐
            elif label.startswith("I-") and ent_type == label[2:].upper():
                continue
            else:
                if start is not None:
                    entities.append({"type": ent_type, "start": start, "end": i})
                    start, ent_type = None, None
        if start is not None:
            entities.append({"type": ent_type, "start": start, "end": len(labels)})
        return entities

    # 3. Convert to JSON format (including ltokens/rtokens)
    data = []
    for idx, (tokens, labels) in enumerate(sentences):
        entities = extract_entities(tokens, labels)

        ltokens = sentences[idx-2][0] if idx >= 2 else []
        rtokens = sentences[idx+1][0] if idx+1 < len(sentences) else []

        sample = {
            "tokens": tokens,
            "entities": entities,
            "relations": [],
            "orig_id": f"{dataset_name}_{idx}",
            "ltokens": ltokens,
            "rtokens": rtokens
        }
        data.append(sample)

    # 4. Save as JSON array保存为 JSON 数组
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(data)} sentences → {output_file}")


if __name__ == "__main__":
    bio_to_json("data/datasets/w17/wnut17train.conll", "data/datasets/w17/wnut177_train.json", "wnut17_train")
    bio_to_json("data/datasets/w17/emerging.dev.conll", "data/datasets/w17/wnut177_dev.json", "wnut17_dev")
    bio_to_json("data/datasets/w17/emerging.test.conll", "data/datasets/w17/wnut177_test.json", "wnut17_test")
