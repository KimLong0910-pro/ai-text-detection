import json
import os
import random
import re
import unicodedata

from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

HF_DATASET_NAME = "vietgpt/binhvq_news_vi"
OUTPUT_FILE = "data/processed/human_binhvq_10k.jsonl"
TARGET_SAMPLES = 10000
MIN_WORDS = 150
MAX_WORDS = 400


def clean_sentence(raw_text: str) -> str:
    text = unicodedata.normalize("NFC", raw_text).strip()
    if re.match(r"^(Ảnh\s*:|Xem thêm|Đọc thêm|Tin liên quan|\(Ảnh)", text, re.I):
        return ""

    text = re.sub(r"\s+([.,!?;:%\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r'"\s*([^"]+?)\s*"', r'"\1"', text)
    text = re.sub(r"\s+-\s+", "-", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def extract_text_from_row(row: dict) -> str:
    for key in ("text", "content", "article", "document"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key]
    str_vals = [v for v in row.values() if isinstance(v, str)]
    return max(str_vals, key=len) if str_vals else ""


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    api = HfApi()
    repo_files = api.list_repo_files(HF_DATASET_NAME, repo_type="dataset")
    parquet_files = sorted([f for f in repo_files if f.endswith(".parquet")])

    # Dùng đúng 5 file đầu tiên đã tải sẵn trong cache (không tốn mạng/dung lượng)
    target_shards = parquet_files[:5]
    print(f"Đang nạp từ cache ({len(target_shards)} shards)...")
    ds = load_dataset(HF_DATASET_NAME, data_files=target_shards, split="train")

    paragraphs = []
    seen_texts = set()
    buffer_sentences = []
    buffer_words = 0

    pbar = tqdm(total=TARGET_SAMPLES, desc="Gom & lọc đoạn văn")
    for row in ds:
        raw = extract_text_from_row(row)
        if not raw:
            continue

        sent = clean_sentence(raw)
        if not sent or not sent.endswith((".", "!", "?", '"', "”")):
            continue

        words_in_sent = len(sent.split())
        if words_in_sent < 5:
            continue

        # Gom các câu liền kề cho đến khi đạt khung 150 - 400 từ
        if buffer_words + words_in_sent <= MAX_WORDS:
            buffer_sentences.append(sent)
            buffer_words += words_in_sent
        else:
            if buffer_words >= MIN_WORDS:
                full_para = " ".join(buffer_sentences)
                if full_para not in seen_texts:
                    seen_texts.add(full_para)
                    paragraphs.append(full_para)
                    pbar.update(1)
                    if len(paragraphs) >= TARGET_SAMPLES:
                        break
            buffer_sentences = [sent]
            buffer_words = words_in_sent

    pbar.close()

    # Xáo trộn ngẫu nhiên các đoạn văn hoàn chỉnh sau khi đã gom xong
    random.Random(42).shuffle(paragraphs)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for idx, text in enumerate(paragraphs):
            record = {
                "id": f"human_binhvq_{idx:05d}",
                "source_file": "binhvq_5shards_grouped",
                "source": "binhvq_news_vi",
                "label": 0,
                "word_count": len(text.split()),
                "text": text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Hoàn tất! Đã lưu {len(paragraphs)} mẫu sạch vào {OUTPUT_FILE}")


if __name__ == "__main__":
    main()