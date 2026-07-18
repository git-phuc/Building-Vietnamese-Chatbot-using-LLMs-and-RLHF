# Bước 2 — SFT (Instruction tuning) · §6.2

**Mục tiêu:** dạy model CPT *format hội thoại trợ lý* — model sau CPT chỉ biết dự đoán token kế tiếp, chưa biết đóng vai chat. Fine-tune trên 12.7k hội thoại multi-turn tiếng Việt, loss chỉ tính trên lượt assistant.

| Notebook | Vai trò | Chạy khi nào | GPU |
|---|---|---|---|
| `SFT-Train-Qwen3-1.7B.ipynb` | QLoRA r=16 (rsLoRA) trên model CPT 4-bit; ChatML template; `train_on_responses_only`; 2 epoch ≈ 800 step; cùng cơ chế resume như CPT-B | **Run All mỗi session** (~1 session T4 là xong) | T4 x1 |

**Điều kiện tiên quyết:** Bước 1 xong — repo `<user>/Qwen3-1.7B-vi-cpt` (merged) đã có trên Hub.

**Input:** `<user>/Qwen3-1.7B-vi-cpt` + dataset `5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca`.
**Output trên Hub:**
- `<user>/Qwen3-1.7B-vi-sft-ckpt` — checkpoint resume (mỗi 100 step)
- `<user>/Qwen3-1.7B-vi-sft` — **bản merged cuối** → input cho Bước 3 (RM) và làm policy khởi đầu cho Bước 4 (DPO)

**Hai quyết định thiết kế** (giải thích đầy đủ ở `docs/spec.md` §6.2): dùng ChatML thuần thay template Qwen3 gốc (template gốc chèn block `<think>`); mask toàn bộ lượt user khi tính loss.

**Tiêu chí xong:** hỏi bằng chat template → trả lời đúng vai trợ lý tiếng Việt, dừng đúng `<|im_end|>`.
