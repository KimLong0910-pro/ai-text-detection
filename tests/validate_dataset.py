import json
import os


def validate_jsonl(filepath: str, is_hybrid: bool = False):
    if not os.path.exists(filepath):
        print(f"[{filepath}] Chưa tồn tại file này (bỏ qua).")
        return

    seen_ids = set()
    seen_texts = set()
    model_counts = {}
    errors = 0
    total = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            if item["id"] in seen_ids or item["text"] in seen_texts:
                errors += 1
            seen_ids.add(item["id"])
            seen_texts.add(item["text"])

            words = item["text"].split()
            if len(words) < 50:
                errors += 1

            if is_hybrid:
                b_idx = item["boundary_word_idx"]
                if len(words) != len(item["word_labels"]):
                    errors += 1
                elif not (
                    all(x == 0 for x in item["word_labels"][:b_idx])
                    and all(x == 1 for x in item["word_labels"][b_idx:])
                ):
                    errors += 1
                m = item["generator_model"]
                model_counts[m] = model_counts.get(m, 0) + 1

    print(f"[{filepath}] Tổng dòng: {total} | Số lỗi: {errors}")
    if is_hybrid:
        print("   -> Phân bố model:", model_counts)


if __name__ == "__main__":
    validate_jsonl("data/processed/human_binhvq_10k.jsonl", is_hybrid=False)
    validate_jsonl("data/processed/hybrid_binhvq_test.jsonl", is_hybrid=True)
    validate_jsonl("data/processed/hybrid_binhvq_part1_long.jsonl", is_hybrid=True)
    validate_jsonl("data/processed/hybrid_binhvq_10k.jsonl", is_hybrid=True)