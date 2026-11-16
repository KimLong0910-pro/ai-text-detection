import json
import os
import random
import re
import unicodedata
from tqdm import tqdm

RAW_ROOT = "data/raw/vietnews_repo/data"
OUTPUT_FILE = "data/processed/human_10k.jsonl"
TARGET_COUNT = 10000
RANDOM_SEED = 42


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("_", " ")
    # 1. Xóa tiền tố nguồn báo ở đầu và chú thích ảnh / tin liên quan ở cuối
    text = re.sub(r"^\s*\([^)]+\)\s*[-–—:]\s*", "", text)
    text = re.sub(r"^(Theo|Nguồn)\s+[A-ZÀ-Ỹa-zà-ỹ0-9\s]+[,:-]\s*", "", text)
    text = re.sub(r"\s*[-–—]\s*Ảnh\s*:.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*(Xem thêm|Đọc thêm|Tin liên quan)\s*:.*$", "", text, flags=re.IGNORECASE)
    # 2. Gom khoảng trắng quanh dấu câu, dấu ngoặc và gạch nối (ví dụ: " tăng tốc " -> "tăng tốc")
    text = re.sub(r"\s+([,.;:!?%\]\)])", r"\1", text)
    text = re.sub(r"([\[\(])\s+", r"\1", text)
    text = re.sub(r'"\s+([^"]+?)\s+"', r'"\1"', text)
    text = re.sub(r"(?<=\w)\s+-\s+(?=\w)", "-", text)
    # 3. Gom khoảng trắng thừa và viết hoa chữ cái đầu đoạn
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


def main():
    files = []
    for root, _, filenames in os.walk(RAW_ROOT):
        for fname in filenames:
            if not fname.startswith("."):
                files.append(os.path.join(root, fname))

    random.seed(RANDOM_SEED)
    random.shuffle(files)

    records = []
    seen_texts = set()

    for fpath in tqdm(files, desc="Đang xử lý Data gốc"):
        if len(records) >= TARGET_COUNT:
            break
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            raw_content = f.read()

        paragraphs = [p.strip() for p in raw_content.split("\n\n") if p.strip()]
        body = " ".join(paragraphs[2:]) if len(paragraphs) >= 3 else raw_content

        cleaned = clean_text(body)
        if cleaned in seen_texts:
            continue

        words = cleaned.split()
        if 150 <= len(words) <= 450:
            seen_texts.add(cleaned)
            records.append({
                "id": f"human_{len(records):05d}",
                "source_file": os.path.basename(fpath),
                "source": "vietnews_pre2020",
                "label": 0,
                "word_count": len(words),
                "text": cleaned,
            })

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Hoàn tất! Đã lưu {len(records)} mẫu sạch vào {OUTPUT_FILE}")


if __name__ == "__main__":
    main()