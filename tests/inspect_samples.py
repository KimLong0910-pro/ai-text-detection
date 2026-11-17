import json
import os
import random
import re

HUMAN_FILE = "data/processed/human_10k.jsonl"
HYBRID_FILE = "data/processed/hybrid_test.jsonl"


def evaluate_human(item: dict) -> list:
    """Bắt các lỗi định dạng báo chí, dấu câu và văn bản phi tự nhiên trong Human."""
    warnings = []
    text = item["text"].strip()

    # 1. Dấu ngoặc kép còn dính khoảng trắng kiểu tokenized (vd: " tăng tốc ")
    if re.search(r'"\s+|\s+"', text):
        warnings.append("Dấu ngoặc kép còn khoảng trắng thừa")

    # 2. Dấu gạch nối bị tách rời (vd: al - Assad)
    if " - " in text:
        warnings.append("Còn dấu gạch nối bị tách khoảng trắng (' - ')")

    # 3. Sót chú thích ảnh, tin liên quan hoặc tên phóng viên cuối bài
    if re.search(
        r"(Ảnh\s*:|Xem thêm|Đọc thêm|Tin liên quan|\(Ảnh)", text, re.IGNORECASE
    ):
        warnings.append("Còn sót rác báo chí (Ảnh / Xem thêm)")

    # 4. Lỗi viết thường đầu đoạn hoặc thiếu dấu kết câu
    if text and text[0].islower():
        warnings.append("Chữ cái đầu đoạn viết thường")
    if not text.endswith((".", "!", "?", '"', "”")):
        warnings.append("Cuối bài thiếu dấu kết câu")

    # 5. Tỷ lệ chữ số quá cao (>15% số từ là số -> dễ là bảng tỷ số, xổ số, thống kê)
    words = text.split()
    num_tokens = sum(1 for w in words if any(c.isdigit() for c in w))
    if words and (num_tokens / len(words)) > 0.15:
        warnings.append(
            f"Quá nhiều con số ({num_tokens}/{len(words)} từ - dễ là tin tỷ số/xổ số)"
        )

    return warnings


def evaluate_hybrid(item: dict) -> list:
    """Bắt các lỗi nối câu và văn phong AI trong Hybrid."""
    warnings = []
    human = item["human_part"].strip()
    ai = item["ai_part"].strip()

    if human.endswith((".", "!", "?")) and ai and ai[0].islower():
        warnings.append("Chữ cái đầu của AI viết thường sau dấu chấm")
    if not ai.endswith((".", "!", "?", '"', "”")):
        warnings.append("Câu cuối AI bị ngắt lửng")

    h_words = human.lower().split()
    a_words = ai.lower().split()
    for i in range(max(0, len(a_words) - 6)):
        phrase = " ".join(a_words[i : i + 6])
        if phrase in " ".join(h_words):
            warnings.append(f"Lặp lại cụm từ của người viết: '{phrase}'")
            break

    if re.search(r"[\u4e00-\u9fff]", ai):
        warnings.append("Lẫn ký tự tiếng Trung")
    if len(a_words) < 50:
        warnings.append(f"Phần AI hơi ngắn ({len(a_words)} từ)")

    return warnings


def inspect_human():
    print("=" * 90)
    print("PHẦN 1: KIỂM TRA CHẤT LƯỢNG BỘ DATA GỐC (human_10k.jsonl)")
    print("=" * 90)
    if not os.path.exists(HUMAN_FILE):
        print(f"Không tìm thấy {HUMAN_FILE}\n")
        return

    with open(HUMAN_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    good_samples = []
    bad_samples = []
    reason_stats = {}

    for item in samples:
        w_list = evaluate_human(item)
        if not w_list:
            good_samples.append(item)
        else:
            bad_samples.append((item, w_list))
            for w in w_list:
                key = w.split("(")[0].strip()
                reason_stats[key] = reason_stats.get(key, 0) + 1

    print(
        f"Tổng số mẫu: {len(samples)} | ✅ Đạt chuẩn: {len(good_samples)} | ⚠️ Bị gắn cờ: {len(bad_samples)}"
    )
    if reason_stats:
        print("Chi tiết các lỗi phát hiện:")
        for k, v in sorted(reason_stats.items(), key=lambda x: -x[1]):
            print(f"   - {k}: {v} mẫu")

    random.seed(42)
    print("\n--- IN THỬ 3 MẪU HUMAN ✅ TỐT (NGẪU NHIÊN) ---")
    for item in random.sample(good_samples, min(3, len(good_samples))):
        print(
            f"[{item['id']}] ({item['word_count']} từ) | Nguồn: {item['source_file']}"
        )
        print(f"   {item['text']}\n")

    if bad_samples:
        print("--- IN THỬ TỐI ĐA 5 MẪU HUMAN ⚠️ CẦN XEM LẠI ---")
        for item, w_list in bad_samples[:5]:
            print(
                f"[{item['id']}] ({item['word_count']} từ) | Lý do: {', '.join(w_list)}"
            )
            print(f"   {item['text']}\n")


def inspect_hybrid():
    print("=" * 90)
    print("PHẦN 2: KIỂM TRA CHẤT LƯỢNG BỘ DATA LAI (hybrid_test.jsonl)")
    print("=" * 90)
    if not os.path.exists(HYBRID_FILE):
        print(f"Không tìm thấy {HYBRID_FILE}\n")
        return

    with open(HYBRID_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    good_count = 0
    for item in samples:
        warnings = evaluate_hybrid(item)
        status = "✅ TỐT" if not warnings else "⚠️ CẦN XEM LẠI"
        if not warnings:
            good_count += 1

        h_tail = " ".join(item["human_part"].split()[-20:])
        print(f"[{item['id']}] | Model: {item['generator_model']} | {status}")
        if warnings:
            for w in warnings:
                print(f"   -> Lý do: {w}")
        print(f"   [...NGƯỜI]: ...{h_tail}")
        print(f"   [AI TIẾP ]: {item['ai_part']}")
        print("-" * 85)

    print(
        f"TỔNG KẾT HYBRID: {good_count}/{len(samples)} mẫu đạt chuẩn tốt hoàn toàn.\n"
    )


if __name__ == "__main__":
    inspect_human()
    inspect_hybrid()
