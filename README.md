# Building Vietnamese Chatbot using LLMs and RLHF

Đồ án vừa học vừa làm: tự xây **toàn bộ pipeline RLHF từ một base model raw** (`Qwen/Qwen3-1.7B-Base`, chưa instruct-tuned) đến chatbot tiếng Việt deploy được — CPT → SFT → Reward Model → DPO → Eval → GGUF/Ollama.

## Đọc gì ở đâu

| File / Mục | Vai trò |
|---|---|
| `Building Vietnamese Chatbot using LLMs and RLHF.md` — **Mục 6** | **Pipeline đang thực thi** — sơ đồ tổng quan ở 6.0, chi tiết + code từng bước ở 6.1–6.9 |
| — Mục 2, 3, 4 | Tài liệu tham khảo (baseline slide Llama, ý tưởng cải tiến, kho dataset) — KHÔNG phải việc đang làm |
| `notebooks/` | Notebook Kaggle chạy được thật, khớp 1-1 với §6.x tương ứng |
| `README.md` (file này) | Bảng trạng thái: đang ở bước nào, xong gì rồi |
| `CLAUDE.md` | Hướng dẫn cho coding agent (Claude Code) |

## Pipeline & trạng thái

Compute: **Kaggle T4 16GB là chính** (30h/tuần, ≤9h/session — mọi bước train đều checkpoint/resume qua HF Hub), **Modal là dự phòng** (bắt buộc cho PPO ablation).

| # | Bước | Chi tiết ở | Output trên Hub | Trạng thái |
|---|---|---|---|---|
| 1 | CPT tiếng Việt (QLoRA r=64) | §6.1 | `<user>/Qwen3-1.7B-vi-cpt` | 🟡 **Notebook A (data prep) đang chạy trên Kaggle** (Save & Run All, 2026-07-17) — chờ xong để kiểm tra `vi-cpt-corpus-2048` rồi chạy Notebook B |
| 2 | SFT (QLoRA r=16) | §6.2 | `<user>/Qwen3-1.7B-vi-sft` | ⬜ chưa bắt đầu |
| 3 | Reward Model (OpenRLHF) | §6.3 | `<user>/Qwen3-1.7B-vi-rm` | ⬜ chưa bắt đầu |
| 4a | DPO — con đường chính (TRL) | §6.4a | `<user>/Qwen3-1.7B-vi-dpo` | ⬜ chưa bắt đầu |
| 4b | PPO — ablation, chỉ Modal | §6.4b | `<user>/Qwen3-1.7B-vi-ppo` | ⬜ tùy chọn |
| 5 | Evaluation (VMLU, LLM-judge, RM score) | §6.5 | — | ⬜ chưa bắt đầu |
| 6 | GGUF + Deploy (Ollama + Open WebUI) | §6.6 | `...-dpo-gguf` | ⬜ chưa bắt đầu |
| 7 | Feedback loop (iterative RLHF) | §6.7 | — | ⬜ stretch goal |

> **Quy ước:** ⬜ chưa bắt đầu · 🟡 đang làm · ✅ xong (đã qua smoke test §6.9). Cập nhật cột Trạng thái ngay khi bắt đầu hoặc hoàn thành một bước — đây là nơi duy nhất ghi tiến độ.

## Việc tiếp theo (bước 1 — CPT)

1. Tạo Kaggle secret `HF_TOKEN` (Add-ons → Secrets), sửa `HF_USER` trong 2 notebook.
2. Upload `notebooks/cpt_a_prepare_data.ipynb` lên Kaggle, chạy đúng 1 lần (không cần GPU) → data đã pack nằm ở `<user>/vi-cpt-corpus-2048`.
3. Upload `notebooks/cpt_b_train.ipynb`, bật GPU T4, Run All mỗi session — tự resume, tự dừng trước khi hết giờ; lặp đến khi đạt `max_steps=6000` thì notebook tự push bản merged `<user>/Qwen3-1.7B-vi-cpt`.
4. Smoke test theo §6.9 (resume đúng step, perplexity tiếng Việt giảm, eval loss tiếng Anh không tăng vọt) → đổi trạng thái bước 1 thành ✅.
