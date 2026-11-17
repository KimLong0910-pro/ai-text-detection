# Vietnamese AI-Generated & Hybrid Text Detection Dataset

Dự án xây dựng bộ dữ liệu tiếng Việt quy mô lớn (**10.000 mẫu văn bản do người viết** và **10.000 mẫu văn bản lai Human–AI** có gán nhãn ranh giới từng từ) phục vụ nghiên cứu bài toán phát hiện văn bản sinh bởi AI (AI-Generated Text Detection & Boundary Localization).

---

## 1. Tiến độ & Những gì đã hoàn thành

### Giai đoạn 1: Xây dựng & làm sạch dữ liệu người viết (`human_10k.jsonl`)
* **Nguồn dữ liệu:** Bộ tin tức báo chí tiếng Việt *VietNews* (xuất bản trước năm 2020 — đảm bảo 100% nguyên bản do người viết, không nhiễm dữ liệu LLM).
* **Quy trình tiền xử lý (`scripts/build_human_data.py`):**
  * Lọc độ dài chuẩn từ **150 đến 450 từ** và loại bỏ trùng lặp văn bản tuyệt đối (`seen_texts`).
  * Chuẩn hóa bảng mã Unicode **NFC**, khôi phục từ ghép (xóa dấu gạch dưới `_` của dữ liệu tokenized).
  * Loại bỏ nhiễu đặc trưng của báo điện tử: tiền tố nguồn tin đầu bài, chú thích ảnh (`Ảnh: ...`), và các dòng điều hướng cuối bài (`Xem thêm:`, `Đọc thêm:`, `Tin liên quan:`).
  * Xử lý lỗi tách khoảng trắng của bộ tokenizer gốc quanh dấu câu, dấu ngoặc kép (`" chữ "` -> `"chữ"`) và dấu gạch nối (` - ` -> `-`), chuẩn hóa viết hoa đầu đoạn.
* **Kết quả:** Đã tạo thành công **10.000 mẫu** (`data/processed/human_10k.jsonl`), kiểm thử đạt **0 lỗi**.

### Giai đoạn 2: Xây dựng pipeline sinh dữ liệu lai (`build_hybrid_data.py`)
* **Chiến lược cắt câu tự nhiên (`find_sentence_cut`):** Cắt đoạn văn gốc tại vị trí ngẫu nhiên từ **30%–60%** độ dài và dò tìm dấu kết câu gần nhất (`.`, `!`, `?`) để phần người viết kết thúc trọn vẹn trước khi AI viết tiếp.
* **Đa dạng hóa mô hình sinh (Multi-LLM):** Sử dụng 3 họ mô hình mã nguồn mở chạy cục bộ qua Ollama:
  * `qwen2.5:7b` (Alibaba)
  * `gemma2:9b` (Google)
  * `llama3.1:8b` (Meta — thay thế `mistral:7b` sau khi kiểm thử thực tế cho thấy `mistral:7b` dễ lặp câu và dịch ngượng tiếng Việt)
* **Tối ưu phần cứng (Apple Silicon M4 16GB):** Gom lô theo từng model (`keep_alive: "15m"`), giới hạn độ dài sinh (`num_predict: 280`), tự động cắt gọn câu lửng cuối đoạn và hỗ trợ **tự động chạy tiếp (Auto-Resume)** nếu nhấn `Ctrl + C` tạm dừng giữa chừng.
* **Kết quả:** Đã hoàn tất kiểm thử Pilot (`IS_TEST_MODE = True`, 15 mẫu chia đều 5 mẫu/model) với **0 lỗi**.

### Giai đoạn 3: Hệ thống kiểm thử & đánh giá chất lượng
* `tests/validate_dataset.py`: Kiểm tra cấu trúc JSONL, phát hiện trùng lặp ID/nội dung, kiểm tra khớp độ dài giữa số từ và mảng `word_labels`, thống kê phân bố model.
* `tests/inspect_samples.py`: Tự động gắn cờ cảnh báo lỗi bề mặt trên cả bộ Human và Hybrid (sót chú thích báo chí, khoảng trắng quanh ngoặc kép, tỷ lệ số liệu quá cao, AI lặp từ của người viết, lẫn chữ Hán hoặc viết thường sau dấu chấm) và in mẫu trực quan.

---

## 2. Cấu trúc thư mục & Định dạng dữ liệu

### Cấu trúc thư mục
```text
ai-text-detection/
├── data/
│   ├── raw/                     # Bộ dữ liệu gốc VietNews (ignored trên Git)
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
└── README.md                    # Tài liệu hướng dẫn toàn bộ dự án
```

### Định dạng mẫu trong file `.jsonl`
* **Mẫu Human (`human_10k.jsonl`):**
  ```json
  {
    "id": "human_00000",
    "source_file": "000123.txt.seg",
    "source": "vietnews_pre2020",
    "label": 0,
    "word_count": 285,
    "text": "Nội dung bài báo đã làm sạch..."
  }
  ```
* **Mẫu Hybrid (`hybrid_10k.jsonl`):**
  ```json
  {
    "id": "hybrid_00000",
    "parent_human_id": "human_00000",
    "generator_model": "qwen2.5:7b",
    "label": 2,
    "boundary_word_idx": 142,
    "human_part": "Phần đầu do người viết...",
    "ai_part": "Phần sau do AI viết tiếp...",
    "text": "Phần đầu do người viết... Phần sau do AI viết tiếp...",
    "word_labels": [0, 0, "...", 1, 1]
  }
  ```
  *(Quy ước nhãn: `label = 0` là Human, `label = 1` là AI-only, `label = 2` là Hybrid; trong `word_labels`: `0` là từ của người, `1` là từ của AI).*

---

## 3. Hướng dẫn cài đặt & Lệnh chạy từ đầu

> **Lưu ý về đường dẫn:** Không đặt thư mục dự án bên trong các thư mục cha có chứa dấu cách hoặc dấu tiếng Việt (nên dùng đường dẫn không dấu, không cách như `/Documents/CODE/Projects/ai-text-detection`) để tránh lỗi khi chạy Python/VS Code.

### Bước 1: Cài đặt môi trường & tải mô hình Ollama
```bash
# 1. Tạo và kích hoạt môi trường ảo Python
python3 -m venv venv
source venv/bin/activate

# 2. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 3. Tải 3 mô hình LLM về máy qua Ollama
ollama pull qwen2.5:7b
ollama pull gemma2:9b
ollama pull llama3.1:8b
```

### Bước 2: Tải dữ liệu gốc & tạo bộ dữ liệu người viết (Human)
*(Do thư mục `data/` bị loại trừ khỏi GitHub để tránh quá tải dung lượng, người mới clone repo cần tải dữ liệu gốc hoặc nhận trực tiếp file `human_10k.jsonl` từ nhóm).*
```bash
# Tải repo VietNews gốc vào đúng vị trí (nếu chưa có)
mkdir -p data/raw
git clone [https://github.com/ThanhChinhBK/vietnews.git](https://github.com/ThanhChinhBK/vietnews.git) data/raw/vietnews_repo

# Chạy script lọc và làm sạch 10.000 mẫu Human
python scripts/build_human_data.py
```

### Bước 3: Sinh bộ dữ liệu lai (Hybrid)
* **Chạy thử nghiệm (Test Mode — 15 mẫu, 5 mẫu/model):**
  Đảm bảo `IS_TEST_MODE = True` trong `scripts/build_hybrid_data.py`, sau đó chạy:
  ```bash
  rm -f data/processed/hybrid_test.jsonl
  python scripts/build_hybrid_data.py
  ```
* **Chạy chính thức toàn bộ 10.000 mẫu (chia đôi chạy song song 2 máy):**
  Đổi `IS_TEST_MODE = False` trong `scripts/build_hybrid_data.py` và dùng `caffeinate -i` để chống macOS tự ngủ khi chạy lâu:
  * **Máy 1 (Long — 5.000 mẫu đầu `[0, 5000)`):**
    ```bash
    caffeinate -i python scripts/build_hybrid_data.py
    ```
  * **Máy 2 (Phát — 5.000 mẫu sau `[5000, 10000)`):** Sửa `OUTPUT_FILE = "data/processed/hybrid_part2_phat.jsonl"`, `START_INDEX = 5000`, `END_INDEX = 10000` (chia lô `1700 / 1700 / 1600` cho 3 model) và chạy lệnh tương tự.

### Bước 4: Kiểm tra & soi chất lượng dữ liệu
```bash
# 1. Kiểm tra lỗi cấu trúc, trùng lặp và khớp nhãn từ
python tests/validate_dataset.py

# 2. Quét toàn bộ mẫu Human & Hybrid, gắn cờ cảnh báo và in mẫu kiểm tra
python tests/inspect_samples.py

# 3. Xem nhanh 2 mẫu cuối cùng vừa sinh ra
tail -n 2 data/processed/hybrid_test.jsonl
```

---

## 4. Quy tắc làm việc với Git & Kế hoạch tiếp theo

### Quy tắc Git
* File `.gitignore` đã chặn sẵn `venv/`, `.DS_Store`, `data/raw/` và `data/processed/*.jsonl`. **Tuyệt đối không ép push các file `.jsonl` lớn lên GitHub**; các file dữ liệu đã xử lý sẽ được chia sẻ qua Google Drive hoặc AirDrop.
* Lệnh đẩy code lên GitHub:
  ```bash
  git add .
  git commit -m "Update pipeline, tests and README"
  git push
  ```

### Kế hoạch tiếp theo
- [ ] Chạy chính thức đủ 10.000 mẫu Hybrid (ghép `hybrid_part1_long.jsonl` và `hybrid_part2_phat.jsonl` thành `hybrid_10k.jsonl`).
- [ ] Bổ sung dữ liệu sinh hoàn toàn bởi AI (`ai_only`) nếu mở rộng sang bài toán phân loại 3 lớp (Human / AI / Hybrid).
- [ ] Chia tập `train / val / test` theo `parent_human_id` để chống rò rỉ dữ liệu (Data Leakage) và huấn luyện các mô hình baseline (PhoBERT, XLM-RoBERTa).


## 🧹 Dọn dẹp dữ liệu & Giải phóng dung lượng (macOS)

> **⚠️ Lưu ý quan trọng trước khi dọn dẹp:** File `.gitignore` đã chặn không cho đẩy dữ liệu `.jsonl` và trọng số mô hình lên GitHub. Hãy chắc chắn bạn đã sao lưu các file kết quả quan trọng (`data/processed/*.jsonl` và thư mục `best_model/`) lên **Google Drive** hoặc **Hugging Face Hub** trước khi chạy các lệnh xóa bên dưới.

### 1. Giải phóng bộ nhớ đệm dữ liệu thô (Làm ngay sau khi chạy xong `build_human_data.py`)

Sau khi đã xuất ra file `data/processed/human_binhvq_10k.jsonl`, bạn có thể xóa ~3 GB file tạm (`.parquet`/`.arrow`) của Hugging Face mà không ảnh hưởng đến các bước sau:

```bash
rm -rf ~/.cache/huggingface/datasets
```

### 2. Gỡ bỏ các model LLM sinh dữ liệu (Làm sau khi hoàn tất tạo dữ liệu Hybrid)

Giải phóng ~15–16 GB dung lượng ổ cứng từ 3 model Ollama đã tải:

```bash
ollama rm qwen2.5:7b gemma2:9b llama3.1:8b
```

*(Nếu không còn sử dụng Ollama cho bất kỳ tác vụ nào khác, có thể xóa toàn bộ thư mục cấu hình và model bằng lệnh `rm -rf ~/.ollama`).*

### 3. Dọn dẹp toàn diện sau khi kết thúc dự án

Xóa toàn bộ bộ nhớ đệm Hugging Face (bao gồm cả trọng số model huấn luyện):

```bash
rm -rf ~/.cache/huggingface
```

Dọn sạch các file nặng trong thư mục dự án nhưng vẫn giữ lại toàn bộ mã nguồn sạch (`scripts/`, `tests/`, `README.md`):

```bash
rm -rf venv data/raw data/processed/*.jsonl checkpoints results logs __pycache__
```
