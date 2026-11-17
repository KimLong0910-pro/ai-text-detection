import json
import os
import random
import re
import time
import unicodedata

import requests
from tqdm import tqdm

INPUT_FILE = "data/processed/human_binhvq_10k.jsonl"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Bật True để test 15 mẫu (5 mẫu/model). Khi chạy thật 5.000 mẫu đổi thành False.
IS_TEST_MODE = True

if IS_TEST_MODE:
    OUTPUT_FILE = "data/processed/hybrid_binhvq_test.jsonl"
    START_INDEX = 0
    END_INDEX = 15
    MODEL_BATCHES = [
        ("qwen2.5:7b", 0, 5),
        ("gemma2:9b", 5, 10),
        ("llama3.1:8b", 10, 15),
    ]
else:
    OUTPUT_FILE = "data/processed/hybrid_binhvq_part1_long.jsonl"
    START_INDEX = 0
    END_INDEX = 5000
    MODEL_BATCHES = [
        ("qwen2.5:7b", 0, 1700),
        ("gemma2:9b", 1700, 3400),
        ("llama3.1:8b", 3400, 5000),
    ]

REFUSAL_PATTERNS = (
    "tôi không thể",
    "xin lỗi",
    "i cannot",
    "i can't",
    "as an ai",
)


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


def call_ollama(
    model_name: str,
    prefix_text: str,
    target_words: int = 120,
    temperature: float = 0.7,
) -> str:
    prompt = (
        "Bạn là một cây bút báo chí tiếng Việt. Hãy viết tiếp đoạn văn sau ít nhất 2 đến 3 câu "
        "một cách tự nhiên, giữ nguyên văn phong, ngữ cảnh và không lặp lại câu đã cho. "
        f"Chỉ trả về phần nội dung viết tiếp bằng tiếng Việt (khoảng {target_words} từ), "
        "không thêm lời dẫn:\n\n"
        f"{prefix_text}"
    )
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "1m",
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": 260,
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def clean_ai_output(text: str, prefix_text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    if text.startswith(prefix_text):
        text = text[len(prefix_text) :].strip()
    text = re.sub(r"^(Tiếp theo:|Phần viết tiếp:)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    # Cắt bỏ câu bị ngắt lửng ở cuối nếu chạm trần num_predict
    if text and not text.endswith((".", "!", "?", '"', "”")):
        last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last_punct != -1 and last_punct > len(text) * 0.4:
            text = text[: last_punct + 1].strip()

    return text


def is_valid_ai_output(ai_part: str) -> bool:
    if len(ai_part.split()) < 20:
        return False
    lower_head = ai_part[:80].lower()
    if any(p in lower_head for p in REFUSAL_PATTERNS):
        return False
    return True


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
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Chưa tìm thấy {INPUT_FILE}. Bạn cần bấm Play file scripts/build_human_data.py trước!"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        human_records = [json.loads(line) for line in f]

    done_ids = load_done_ids(OUTPUT_FILE)
    print(
        f"Chế độ Test: {IS_TEST_MODE} | Đã có: {len(done_ids)} mẫu | Chạy [{START_INDEX}, {END_INDEX})..."
    )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for idx in tqdm(range(START_INDEX, END_INDEX), desc="Sinh Hybrid (binhvq)"):
            rec = human_records[idx]
            if rec["id"] in done_ids:
                continue

            rng = random.Random(42 + idx)
            words = rec["text"].split()
            cut_idx = find_sentence_cut(words, rng.uniform(0.3, 0.6))
            human_prefix = " ".join(words[:cut_idx])

            primary_model = get_model_for_index(idx)
            # Thử lại tối đa 3 lần trên đúng model đang nằm sẵn trong RAM để tránh tràn 16GB
            attempts = [
                (primary_model, 0.7),
                (primary_model, 0.85),
                (primary_model, 1.0),
            ]

            ai_part = ""
            for model_name, temp in attempts:
                try:
                    raw_ai = call_ollama(
                        model_name, human_prefix, target_words=120, temperature=temp
                    )
                    cleaned = clean_ai_output(raw_ai, human_prefix)
                    if is_valid_ai_output(cleaned):
                        ai_part = cleaned
                        break
                except Exception as e:
                    print(f"\n[Chờ 3s thử lại {rec['id']} với {model_name}]: {e}")
                    time.sleep(3)

            if not ai_part:
                print(f"\n[Bỏ qua mẫu {rec['id']}] Không thể sinh hợp lệ sau 3 lần thử.")
                continue

            ai_words = ai_part.split()
            human_words = human_prefix.split()
            full_words = human_words + ai_words
            word_labels = [0] * len(human_words) + [1] * len(ai_words)

            hybrid_record = {
                "id": f"hybrid_binhvq_{idx:05d}",
                "parent_human_id": rec["id"],
                "generator_model": primary_model,
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