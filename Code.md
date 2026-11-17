## Cấu trúc thư mục
```bash
ai-text-detection/
├── data/
│   └── processed/               # Các file .jsonl đầu ra (ignored trên Git)
│       ├── human_binhvq_10k.jsonl      # 10.000 mẫu văn bản người viết đã làm sạch
│       └── hybrid_binhvq_test.jsonl    # 15 mẫu dữ liệu lai chạy thử nghiệm
├── scripts/
│   ├── build_human_data.py      # Script lọc và làm sạch 10.000 mẫu Human
│   └── build_hybrid_data.py     # Script gọi Ollama sinh dữ liệu Hybrid
├── tests/
│   ├── validate_dataset.py      # Kiểm tra tính toàn vẹn cấu trúc dữ liệu
│   └── inspect_samples.py       # Kiểm tra chi tiết văn phong và gắn cờ mẫu lỗi
├── .gitignore                   # Loại trừ venv, .DS_Store và dữ liệu lớn
├── requirements.txt             # Danh sách thư viện Python của dự án
├── Code.md                      # Tổng hợp lại code
└── README.md                    # Tài liệu hướng dẫn toàn bộ dự án
```

### build_human_data.py
```py
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
    if re.match(r"^(Ảnh\s*:|Xem thêm|Đọc thêm|Tin liên quan|\(Ảnh)", text, re.IGNORECASE):
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
```

### build_hybrid_data.py
```py
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
```

### validate_dataset.py
```py
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
```

### inspect_samples.py
```py
import json
import os
import random
import re
from collections import defaultdict

HUMAN_FILE = "data/processed/human_binhvq_10k.jsonl"
# Đổi thành "data/processed/hybrid_binhvq_part1_long.jsonl" khi kiểm tra bản chạy thật 5.000 mẫu
HYBRID_FILE = "data/processed/hybrid_binhvq_test.jsonl"

MAX_HYBRID_DISPLAY = 15
USE_COLORS = True  # Tô màu phân biệt trên Terminal macOS

# Mã màu ANSI cho Terminal
C_RESET = "\033[0m" if USE_COLORS else ""
C_HUMAN = "\033[96m" if USE_COLORS else ""          # Xanh lơ nhạt: Phần người viết
C_AI = "\033[92m" if USE_COLORS else ""             # Xanh lá nhạt: Phần AI viết tiếp
C_MARKER = "\033[1;93m" if USE_COLORS else ""       # Vàng đậm: Mốc ranh giới chuyển tiếp
C_WARN = "\033[91m" if USE_COLORS else ""           # Đỏ chữ: Cảnh báo lỗi
C_ERR_BG = "\033[1;41;97m" if USE_COLORS else "【"  # Nền đỏ chữ trắng: Bôi đậm đúng chỗ lỗi
C_ERR_END = "\033[0m" if USE_COLORS else "】"
C_NOTE = "\033[95m" if USE_COLORS else ""           # Tím nhạt: Nhận xét phân tích

LLM_CHAT_ARTIFACTS = (
    "dưới đây là",
    "phần viết tiếp",
    "tiếp theo:",
    "tôi không thể",
    "xin lỗi",
    "là một mô hình ngôn ngữ",
    "as an ai",
    "here is",
)

AI_CLICHE_TRANSITIONS = (
    "ngoài ra,",
    "bên cạnh đó,",
    "điều này cho thấy",
    "không chỉ vậy,",
    "tuy nhiên, điều đáng chú ý",
    "nhìn chung,",
    "có thể thấy rằng",
    "trong bối cảnh đó,",
)

VI_STOPWORDS = {
    "và", "của", "là", "các", "những", "được", "trong", "cho", "với", "tại",
    "đã", "đang", "sẽ", "về", "từ", "này", "đó", "khi", "để", "như", "một",
    "có", "không", "người", "năm", "ngày", "theo", "cũng", "nhưng", "vào", "ra",
}


def evaluate_human(item: dict) -> tuple[list, str]:
    """
    Kiểm tra lỗi định dạng trong tập Human.
    Trả về: (danh_sách_lý_do, đoạn_văn_đã_bôi_đỏ_chỗ_lỗi)
    """
    warnings = []
    text = item["text"].strip()
    highlighted = text
    words = text.split()

    # 1. Khung độ dài chuẩn [150, 400] từ
    if not (150 <= len(words) <= 400):
        warnings.append(f"Ngoài khung độ dài 150-400 từ ({len(words)} từ)")

    # 2. Dấu ngoặc kép còn dính khoảng trắng thừa bên trong (vd: " tăng tốc ")
    # Chỉ bắt ngoặc mở theo sau bởi dấu cách, hoặc ngoặc đóng đứng trước bởi dấu cách
    bad_quote_pattern = r'(?:^|[\s(\[])"\s+\S|\S\s+"(?:[\s.,!?;:\)\]]|$)'
    if re.search(bad_quote_pattern, text):
        warnings.append("Dấu ngoặc kép còn khoảng trắng thừa bên trong")
        highlighted = re.sub(
            rf"({bad_quote_pattern})", rf"{C_ERR_BG}\1{C_ERR_END}", highlighted
        )

    # 3. Dấu gạch nối bị tách rời (vd: al - Assad)
    if re.search(r"\S+\s+-\s+\S+", text):
        warnings.append("Còn dấu gạch nối bị tách khoảng trắng (' - ')")
        highlighted = re.sub(
            r"(\S+\s+-\s+\S+)", rf"{C_ERR_BG}\1{C_ERR_END}", highlighted
        )

    # 4. Sót chú thích ảnh, tin liên quan
    junk_pattern = r"(?:Ảnh\s*:|Xem thêm|Đọc thêm|Tin liên quan|\(Ảnh)[^.]*"
    if re.search(junk_pattern, text, re.IGNORECASE):
        warnings.append("Còn sót rác báo chí (Ảnh / Xem thêm)")
        highlighted = re.sub(
            rf"({junk_pattern})",
            rf"{C_ERR_BG}\1{C_ERR_END}",
            highlighted,
            flags=re.IGNORECASE,
        )

    # 5. Chữ cái đầu đoạn viết thường
    if text and text[0].islower():
        warnings.append("Chữ cái đầu đoạn viết thường")
        first_word = words[0] if words else text[0]
        highlighted = f"{C_ERR_BG}{first_word}{C_ERR_END}" + highlighted[len(first_word) :]

    # 6. Cuối bài thiếu dấu kết câu
    if text and not text.endswith((".", "!", "?", '"', "”")):
        warnings.append("Cuối bài thiếu dấu kết câu")
        tail = " ".join(words[-6:])
        if highlighted.endswith(tail):
            highlighted = highlighted[: -len(tail)] + f"{C_ERR_BG}{tail}{C_ERR_END}"

    # 7. Tỷ lệ chữ số quá cao (>15%)
    num_tokens = [w for w in words if any(c.isdigit() for c in w)]
    if words and (len(num_tokens) / len(words)) > 0.15:
        warnings.append(
            f"Quá nhiều con số ({len(num_tokens)}/{len(words)} từ - dễ là tin tỷ số/xổ số)"
        )
        highlighted = re.sub(r"(\b\S*\d\S*\b)", rf"{C_ERR_BG}\1{C_ERR_END}", highlighted)

    # 8. Tỷ lệ từ tiếng Anh / không dấu quá cao (>25%)
    ascii_tokens = [w for w in words if re.fullmatch(r"[A-Za-z]{4,}", w)]
    if words and (len(ascii_tokens) / len(words)) > 0.25:
        warnings.append(
            f"Nhiều từ tiếng Anh/không dấu ({len(ascii_tokens)}/{len(words)} từ)"
        )
        highlighted = re.sub(r"\b([A-Za-z]{4,})\b", rf"{C_ERR_BG}\1{C_ERR_END}", highlighted)

    return warnings, highlighted


def evaluate_hybrid(item: dict) -> list:
    """Bắt các lỗi nối câu, văn phong LLM và dấu hiệu 'học vẹt' trong Hybrid."""
    warnings = []
    human = item["human_part"].strip()
    ai = item["ai_part"].strip()
    h_words = human.lower().split()
    a_words = ai.lower().split()
    total_words = len(h_words) + len(a_words)

    if human.endswith((".", "!", "?")) and ai and ai[0].islower():
        warnings.append("Chữ cái đầu của AI viết thường sau dấu chấm")
    if not ai.endswith((".", "!", "?", '"', "”")):
        warnings.append("Câu cuối AI bị ngắt lửng")

    human_joined = " ".join(h_words)
    for i in range(max(0, len(a_words) - 6)):
        phrase = " ".join(a_words[i : i + 6])
        if phrase in human_joined:
            warnings.append(f"Lặp lại cụm từ của người viết: '{phrase}'")
            break

    if re.search(r"[\u4e00-\u9fff]", ai):
        warnings.append("Lẫn ký tự tiếng Trung (thường gặp ở Qwen)")

    ai_lower = ai.lower()
    for phrase in LLM_CHAT_ARTIFACTS:
        if phrase in ai_lower[:100]:
            warnings.append(f"Dính lời dẫn/từ chối của chatbot: '{phrase}'")
            break

    if "**" in ai or "##" in ai or re.search(r"(^|\n)\s*[-*•]\s+", ai):
        warnings.append("AI tự chèn Markdown (in đậm/gạch đầu dòng) khác văn báo chí")

    if len(a_words) < 45:
        warnings.append(f"Phần AI hơi ngắn ({len(a_words)} từ)")
    if total_words > 420:
        warnings.append(
            f"Tổng độ dài lớn ({total_words} từ -> dễ bị cắt mất đuôi AI nếu train PhoBERT 256 token)"
        )

    return warnings


def analyze_sample_details(item: dict, warnings: list) -> list:
    """Sinh nhận xét chuyên sâu về độ mượt chuyển tiếp, nhịp điệu câu và dấu vết AI."""
    notes = []
    human = item["human_part"].strip()
    ai = item["ai_part"].strip()

    h_sents = [s.strip() for s in re.split(r"[.!?]+", human) if s.strip()]
    a_sents = [s.strip() for s in re.split(r"[.!?]+", ai) if s.strip()]

    h_tail_text = " ".join(h_sents[-2:]).lower()
    a_head_text = " ".join(a_sents[:2]).lower()

    h_keywords = {
        w.strip(".,!?;:\"'()")
        for w in h_tail_text.split()
        if len(w) > 2 and w.strip(".,!?;:\"'()") not in VI_STOPWORDS
    }
    a_keywords = {
        w.strip(".,!?;:\"'()")
        for w in a_head_text.split()
        if len(w) > 2 and w.strip(".,!?;:\"'()") not in VI_STOPWORDS
    }
    overlap = h_keywords.intersection(a_keywords)

    if len(overlap) >= 2:
        sample_kw = ", ".join(list(overlap)[:4])
        notes.append(
            f"Mạch nối mượt: AI giữ được {len(overlap)} từ khóa ngữ cảnh từ phần người viết ({sample_kw})."
        )
    elif len(overlap) == 1:
        notes.append(
            f"Mạch nối trung bình: Chỉ giữ lại 1 từ khóa chung ('{list(overlap)[0]}'), ý đang mở rộng sang hướng mới."
        )
    else:
        notes.append(
            "Mạch nối rời rạc: Không có từ khóa trùng ở câu giao điểm — cần đọc kiểm tra xem AI có bị lạc đề không."
        )

    h_avg_len = sum(len(s.split()) for s in h_sents) / max(1, len(h_sents))
    a_avg_len = sum(len(s.split()) for s in a_sents) / max(1, len(a_sents))
    notes.append(
        f"Nhịp điệu câu: Người viết TB {h_avg_len:.1f} từ/câu ({len(h_sents)} câu) | "
        f"AI viết tiếp TB {a_avg_len:.1f} từ/câu ({len(a_sents)} câu)."
    )

    ai_lower = ai.lower()
    found_cliches = [c for c in AI_CLICHE_TRANSITIONS if c in ai_lower]
    if found_cliches:
        notes.append(
            f"Dấu vết văn phong LLM: AI dùng từ nối khuôn mẫu ({', '.join(found_cliches)})."
        )
    elif not warnings:
        notes.append(
            "Đánh giá chung: Văn phong AI hòa nhập tự nhiên với giọng báo chí, khó phân biệt bằng mắt thường."
        )

    return notes


def inspect_human():
    print("=" * 95)
    print(f"PHẦN 1: KIỂM TRA CHẤT LƯỢNG BỘ DATA GỐC ({os.path.basename(HUMAN_FILE)})")
    print("=" * 95)
    if not os.path.exists(HUMAN_FILE):
        print(f"Không tìm thấy {HUMAN_FILE}. Hãy chạy build_human_data.py trước!\n")
        return

    with open(HUMAN_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    good_samples = []
    bad_samples = []
    reason_stats = defaultdict(int)
    word_counts = []

    for item in samples:
        word_counts.append(item["word_count"])
        w_list, highlighted = evaluate_human(item)
        if not w_list:
            good_samples.append(item)
        else:
            bad_samples.append((item, w_list, highlighted))
            for w in w_list:
                key = w.split("(")[0].strip()
                reason_stats[key] += 1

    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    print(
        f"Tổng số mẫu: {len(samples)} | Trung bình: {avg_words:.1f} từ/mẫu | "
        f"✅ Đạt chuẩn: {len(good_samples)} ({len(good_samples)/max(1,len(samples))*100:.1f}%) | "
        f"⚠️ Bị gắn cờ: {len(bad_samples)}"
    )
    if reason_stats:
        print("Chi tiết các lỗi phát hiện:")
        for k, v in sorted(reason_stats.items(), key=lambda x: -x[1]):
            print(f"   - {k}: {v} mẫu")

    random.seed(42)
    print("\n--- IN THỬ 3 MẪU HUMAN ✅ TỐT (NGẪU NHIÊN) ---")
    for item in random.sample(good_samples, min(3, len(good_samples))):
        print(f"[{item['id']}] ({item['word_count']} từ) | Nguồn: {item['source_file']}")
        print(f"   {C_HUMAN}{item['text']}{C_RESET}\n")

    if bad_samples:
        print("--- IN THỬ TỐI ĐA 5 MẪU HUMAN ⚠️ CẦN XEM LẠI ---")
        for item, w_list, highlighted in bad_samples[:5]:
            print(f"[{item['id']}] ({item['word_count']} từ) | {C_WARN}Lý do: {', '.join(w_list)}{C_RESET}")
            print(f"   📝 Văn bản đánh dấu lỗi:\n      {highlighted}\n")
            print("-" * 95)


def inspect_hybrid():
    print("=" * 95)
    print(f"PHẦN 2: KIỂM TRA & NHẬN XÉT BỘ DATA LAI ({os.path.basename(HYBRID_FILE)})")
    print("=" * 95)
    if not os.path.exists(HYBRID_FILE):
        print(f"Không tìm thấy {HYBRID_FILE}. Hãy chạy build_hybrid_data.py trước!\n")
        return

    with open(HYBRID_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    good_samples = []
    bad_samples = []
    model_stats = defaultdict(lambda: {"total": 0, "good": 0})
    reason_stats = defaultdict(int)

    for item in samples:
        m = item.get("generator_model", "unknown")
        model_stats[m]["total"] += 1
        warnings = evaluate_hybrid(item)
        if not warnings:
            good_samples.append((item, warnings))
            model_stats[m]["good"] += 1
        else:
            bad_samples.append((item, warnings))
            for w in warnings:
                key = w.split("(")[0].strip()
                reason_stats[key] += 1

    print(
        f"TỔNG KẾT HYBRID: ✅ Đạt chuẩn {len(good_samples)}/{len(samples)} mẫu "
        f"({len(good_samples)/max(1,len(samples))*100:.1f}%) | ⚠️ Bị gắn cờ: {len(bad_samples)}"
    )
    print("Thống kê chất lượng theo từng Model AI:")
    for m, st in model_stats.items():
        rate = st["good"] / max(1, st["total"]) * 100
        print(f"   - {m:<15}: {st['good']}/{st['total']} mẫu tốt ({rate:.1f}%)")

    if reason_stats:
        print("Chi tiết các lỗi phát hiện ở tập Hybrid:")
        for k, v in sorted(reason_stats.items(), key=lambda x: -x[1]):
            print(f"   - {k}: {v} mẫu")

    to_display = bad_samples[:10]
    remaining_slots = max(0, MAX_HYBRID_DISPLAY - len(to_display))
    if remaining_slots > 0 and good_samples:
        random.seed(42)
        to_display.extend(
            random.sample(good_samples, min(remaining_slots, len(good_samples)))
        )

    print(
        f"\n--- PHÂN TÍCH CHI TIẾT {len(to_display)} MẪU "
        f"({C_HUMAN}[Người viết]{C_RESET} ➔ {C_AI}[AI viết tiếp]{C_RESET}) ---"
    )
    for item, warnings in to_display:
        status = "✅ TỐT" if not warnings else f"{C_WARN}⚠️ CẦN XEM LẠI{C_RESET}"
        h_words = item["human_part"].split()
        a_words = item["ai_part"].split()
        total_w = len(h_words) + len(a_words)
        h_ratio = len(h_words) / max(1, total_w) * 100
        a_ratio = 100 - h_ratio

        print(
            f"\n[{item['id']}] | Gốc: {item['parent_human_id']} | Model: {item['generator_model']} | {status}"
        )
        print(
            f"   📊 Tỷ lệ đóng góp: Người {len(h_words)} từ ({h_ratio:.0f}%) "
            f"— AI {len(a_words)} từ ({a_ratio:.0f}%) | Tổng: {total_w} từ"
        )

        print("   📝 Toàn văn phân tách ranh giới:")
        print(
            f"      {C_HUMAN}[NGƯỜI VIẾT]: {item['human_part']}{C_RESET} "
            f"{C_MARKER}<<< [RANH GIỚI NGƯỜI -> AI] >>>{C_RESET} "
            f"{C_AI}[AI VIẾT TIẾP]: {item['ai_part']}{C_RESET}"
        )

        if warnings:
            print(f"   {C_WARN}🚨 Cảnh báo kỹ thuật:{C_RESET}")
            for w in warnings:
                print(f"      - {w}")

        notes = analyze_sample_details(item, warnings)
        print(f"   {C_NOTE}💡 Nhận xét mẫu:{C_RESET}")
        for note in notes:
            print(f"      - {note}")
        print("-" * 95)


if __name__ == "__main__":
    inspect_human()
    inspect_hybrid()
```