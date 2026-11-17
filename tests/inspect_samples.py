import json
import os
import random
import re
from collections import defaultdict

HUMAN_FILE = "data/processed/human_binhvq_10k.jsonl"
# Đổi thành "data/processed/hybrid_binhvq_part1_long.jsonl" khi kiểm tra bản chạy thật 5.000 mẫu
HYBRID_FILE = "data/processed/hybrid_binhvq_test.jsonl"

MAX_HUMAN_GOOD_DISPLAY = 3  # Số mẫu Human tốt muốn in thử
MAX_HUMAN_BAD_DISPLAY = 15 # Số mẫu Human lỗi muốn in thử
MAX_HYBRID_DISPLAY = 15
USE_COLORS = True  # Tô màu phân biệt trên Terminal macOS

# Mã màu ANSI cho Terminal
C_RESET = "\033[0m" if USE_COLORS else ""
C_HUMAN = "\033[96m" if USE_COLORS else ""  # Xanh lơ nhạt: Phần người viết
C_AI = "\033[92m" if USE_COLORS else ""  # Xanh lá nhạt: Phần AI viết tiếp
C_MARKER = "\033[1;93m" if USE_COLORS else ""  # Vàng đậm: Mốc ranh giới chuyển tiếp
C_WARN = "\033[91m" if USE_COLORS else ""  # Đỏ chữ: Cảnh báo lỗi
C_ERR_BG = (
    "\033[1;41;97m" if USE_COLORS else "【"
)  # Nền đỏ chữ trắng: Bôi đậm đúng chỗ lỗi
C_ERR_END = "\033[0m" if USE_COLORS else "】"
C_NOTE = "\033[95m" if USE_COLORS else ""  # Tím nhạt: Nhận xét phân tích

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
    "và",
    "của",
    "là",
    "các",
    "những",
    "được",
    "trong",
    "cho",
    "với",
    "tại",
    "đã",
    "đang",
    "sẽ",
    "về",
    "từ",
    "này",
    "đó",
    "khi",
    "để",
    "như",
    "một",
    "có",
    "không",
    "người",
    "năm",
    "ngày",
    "theo",
    "cũng",
    "nhưng",
    "vào",
    "ra",
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
        highlighted = (
            f"{C_ERR_BG}{first_word}{C_ERR_END}" + highlighted[len(first_word) :]
        )

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
        highlighted = re.sub(
            r"(\b\S*\d\S*\b)", rf"{C_ERR_BG}\1{C_ERR_END}", highlighted
        )

    # 8. Tỷ lệ từ tiếng Anh / không dấu quá cao (>25%)
    ascii_tokens = [w for w in words if re.fullmatch(r"[A-Za-z]{4,}", w)]
    if words and (len(ascii_tokens) / len(words)) > 0.25:
        warnings.append(
            f"Nhiều từ tiếng Anh/không dấu ({len(ascii_tokens)}/{len(words)} từ)"
        )
        highlighted = re.sub(
            r"\b([A-Za-z]{4,})\b", rf"{C_ERR_BG}\1{C_ERR_END}", highlighted
        )

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
        f"✅ Đạt chuẩn: {len(good_samples)} ({len(good_samples) / max(1, len(samples)) * 100:.1f}%) | "
        f"⚠️ Bị gắn cờ: {len(bad_samples)}"
    )
    if reason_stats:
        print("Chi tiết các lỗi phát hiện:")
        for k, v in sorted(reason_stats.items(), key=lambda x: -x[1]):
            print(f"   - {k}: {v} mẫu")

    random.seed(42)
    num_good = min(MAX_HUMAN_GOOD_DISPLAY, len(good_samples))
    print(f"\n--- IN THỬ {num_good} MẪU HUMAN ✅ TỐT (NGẪU NHIÊN) ---")
    for item in random.sample(good_samples, num_good):
        print(
            f"[{item['id']}] ({item['word_count']} từ) | Nguồn: {item['source_file']}"
        )
        print(f"   {C_HUMAN}{item['text']}{C_RESET}\n")

    if bad_samples:
        num_bad = min(MAX_HUMAN_BAD_DISPLAY, len(bad_samples))
        print(f"--- IN THỬ {num_bad} MẪU HUMAN ⚠️ CẦN XEM LẠI ---")
        for item, w_list, highlighted in bad_samples[:num_bad]:
            print(
                f"[{item['id']}] ({item['word_count']} từ) | {C_WARN}Lý do: {', '.join(w_list)}{C_RESET}"
            )
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
        f"({len(good_samples) / max(1, len(samples)) * 100:.1f}%) | ⚠️ Bị gắn cờ: {len(bad_samples)}"
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
