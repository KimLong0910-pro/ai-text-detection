import json
import os
import random
import re
import unicodedata

import requests
from tqdm import tqdm

INPUT_FILE = "data/processed/human_10k.jsonl"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Bật True để test 15 mẫu (5 mẫu/model). Khi chạy thật đổi thành False.
IS_TEST_MODE = True

if IS_TEST_MODE:
    OUTPUT_FILE = "data/processed/hybrid_test.jsonl"
    START_INDEX = 0
    END_INDEX = 18
    MODEL_BATCHES = [
        ("qwen2.5:7b", 0, 6),
        ("gemma2:9b", 6, 12),
        ("llama3.1:8b", 12, 18),
    ]
else:
    OUTPUT_FILE = "data/processed/hybrid_part1_long.jsonl"
    START_INDEX = 0
    END_INDEX = 5000
    MODEL_BATCHES = [
        ("qwen2.5:7b", 0, 1700),
        ("gemma2:9b", 1700, 3400),
        ("llama3.1:8b", 3400, 5000),
    ]


def get_model_for_index(idx: int) -> str:
    for model_name, start, end in MODEL_BATCHES:
        if start <= idx < end:
            return model_name
    return MODEL_BATCHES[0][0]


def find_sentence_cut(words: list, target_ratio: float) -> int:
    target_idx = max(30, int(len(words) * target_ratio))
    best_idx = None
    min_dist = 999
    for i in range(max(25, target_idx - 25), min(len(words) - 20, target_idx + 25)):
        if words[i].endswith((".", "!", "?")):
            dist = abs(i - target_idx)
            if dist < min_dist:
                min_dist = dist
                best_idx = i + 1
    return best_idx if best_idx is not None else target_idx


def call_ollama(model_name: str, prefix_text: str, target_words: int = 120) -> str:
    prompt = (
        "Bạn là một cây bút báo chí tiếng Việt. Hãy viết tiếp đoạn văn sau một cách "
        "tự nhiên, giữ nguyên văn phong, ngữ cảnh và không lặp lại câu đã cho. "
        f"Chỉ trả về phần nội dung viết tiếp bằng tiếng Việt (khoảng {target_words} từ), "
        "không thêm lời dẫn:\n\n"
        f"{prefix_text}"
    )
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "15m",
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 220,
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def clean_ai_output(text: str, prefix_text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    if text.startswith(prefix_text):
        text = text[len(prefix_text):].strip()
    text = re.sub(r"^(Tiếp theo:|Phần viết tiếp:)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_done_ids(filepath: str) -> set:
    if not os.path.exists(filepath):
        return set()
    done = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["parent_human_id"])
            except Exception:
                continue
    return done


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        human_records = [json.loads(line) for line in f]

    done_ids = load_done_ids(OUTPUT_FILE)
    print(f"Chế độ Test: {IS_TEST_MODE} | Đã có: {len(done_ids)} mẫu | Chạy [{START_INDEX}, {END_INDEX})...")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for idx in tqdm(range(START_INDEX, END_INDEX), desc="Sinh Data lai"):
            rec = human_records[idx]
            if rec["id"] in done_ids:
                continue

            rng = random.Random(42 + idx)
            words = rec["text"].split()
            cut_idx = find_sentence_cut(words, rng.uniform(0.3, 0.6))
            human_prefix = " ".join(words[:cut_idx])

            model_name = get_model_for_index(idx)
            try:
                raw_ai = call_ollama(model_name, human_prefix, target_words=120)
                ai_part = clean_ai_output(raw_ai, human_prefix)
            except Exception as e:
                print(f"\n[Lỗi mẫu {rec['id']} với {model_name}]: {e}")
                continue

            ai_words = ai_part.split()
            if len(ai_words) < 30:
                continue

            human_words = human_prefix.split()
            full_words = human_words + ai_words
            word_labels = [0] * len(human_words) + [1] * len(ai_words)

            hybrid_record = {
                "id": f"hybrid_{idx:05d}",
                "parent_human_id": rec["id"],
                "generator_model": model_name,
                "label": 2,
                "boundary_word_idx": len(human_words),
                "human_part": human_prefix,
                "ai_part": ai_part,
                "text": " ".join(full_words),
                "word_labels": word_labels,
            }
            out.write(json.dumps(hybrid_record, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()