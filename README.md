# Building Vietnamese Chatbot using LLMs and RLHF

Đồ án vừa học vừa làm: tự xây **toàn bộ pipeline RLHF từ một base model raw** (`Qwen/Qwen3-1.7B-Base`, chưa instruct-tuned) đến chatbot tiếng Việt deploy được — CPT → SFT → Reward Model → DPO → Eval → GGUF/Ollama.

## Đọc gì ở đâu

| File / Mục | Vai trò |
|---|---|
| `docs/spec.md` — **Mục 6** | **Pipeline đang thực thi** (spec chính) — sơ đồ tổng quan ở 6.0, chi tiết + code từng bước ở 6.1–6.9 |
| — Mục 2, 3, 4 | Tài liệu tham khảo (baseline slide Llama, ý tưởng cải tiến, kho dataset) — KHÔNG phải việc đang làm |
| `notebooks/` | Notebook Kaggle chạy được thật, **chia folder theo bước** (`01-cpt/` → `06-deploy/`), mỗi folder có README riêng: mục tiêu, notebook nào chạy khi nào, input/output |
| `research/` | Paper nền tảng theo từng topic (CPT, SFT, RLHF, DPO, QLoRA, LLM Việt/SEA, eval) — mỗi file một topic, map vào §6.x |
| `report/` | **Report chính thức** — *"RLHF on a Shoestring"*: `report.md` (bản nháp tiếng Việt) + `main.tex`/`references.bib` (bản nộp tiếng Anh, compile pdfLaTeX trên Overleaf) — sửa bên nào thì đồng bộ bên kia |
| `docs/` | Tài liệu: `spec.md` (spec chính, xem dòng đầu bảng) + `orchestrator.md` (thiết kế multi-agent cho giai đoạn product, bước 6–7) + slide PDF môn học |
| `README.md` (file này) | Bảng trạng thái: đang ở bước nào, xong gì rồi |
| `CLAUDE.md` | Hướng dẫn cho coding agent (Claude Code) |

## Pipeline & trạng thái

Compute: **Kaggle T4 16GB là chính** (30h/tuần, ≤9h/session — mọi bước train đều checkpoint/resume qua HF Hub), **Modal là dự phòng** (bắt buộc cho PPO ablation).

| # | Bước | Chi tiết ở | Output trên Hub | Trạng thái |
|---|---|---|---|---|
| 1 | CPT tiếng Việt (QLoRA r=64) | §6.1 | `<user>/Qwen3-1.7B-vi-cpt` | 🟡 **Notebook A ✅ xong (2026-07-18)** — data tại `Phuc-HugigFace/vi-cpt-corpus-2048` (train 80/20 vi-en + eval_vi + eval_en). **Đang làm: Notebook B** — import bản mới (config `whoami()`) lên Kaggle, GPU T4, Run All mỗi session đến 3000 step (batch hiệu dụng 64 → ~400M token) |
| 2 | SFT (QLoRA r=16) | §6.2 | `<user>/Qwen3-1.7B-vi-sft` | ⬜ chưa bắt đầu — notebook đã sẵn sàng: `notebooks/02-sft/SFT-Train-Qwen3-1.7B.ipynb` (chờ bước 1 xong) |
| 3 | Reward Model (OpenRLHF) | §6.3 | `<user>/Qwen3-1.7B-vi-rm` | ⬜ chưa bắt đầu |
| 4a | DPO — con đường chính (TRL) | §6.4a | `<user>/Qwen3-1.7B-vi-dpo` | ⬜ chưa bắt đầu |
| 4b | PPO — ablation, chỉ Modal | §6.4b | `<user>/Qwen3-1.7B-vi-ppo` | ⬜ tùy chọn |
| 5 | Evaluation (VMLU, LLM-judge, RM score) | §6.5 | — | ⬜ chưa bắt đầu |
| 6 | GGUF + Deploy (Ollama + Open WebUI) | §6.6 | `...-dpo-gguf` | ⬜ chưa bắt đầu |
| 7 | Feedback loop (iterative RLHF) | §6.7 | — | ⬜ stretch goal |

> **Quy ước:** ⬜ chưa bắt đầu · 🟡 đang làm · ✅ xong (đã qua smoke test §6.9). Cập nhật cột Trạng thái ngay khi bắt đầu hoặc hoàn thành một bước — đây là nơi duy nhất ghi tiến độ.

## Việc tiếp theo (bước 1 — CPT)

1. Tạo Kaggle secret `HF_TOKEN` (Add-ons → Secrets) — phải là **Write** token; username HF các notebook tự lấy từ token (`whoami()`), không cần sửa gì.
2. Upload `notebooks/01-cpt/CPT-Step-A-Prepare-Qwen3-1.7B.ipynb` lên Kaggle, chạy đúng 1 lần (không cần GPU) → data đã pack nằm ở `<user>/vi-cpt-corpus-2048`.
3. Upload `notebooks/01-cpt/CPT-Step-B-Train-Qwen3-1.7B.ipynb`, bật GPU T4, Run All mỗi session — tự resume, tự dừng trước khi hết giờ; lặp đến khi đạt `max_steps=3000` thì notebook tự push bản merged `<user>/Qwen3-1.7B-vi-cpt`.
4. Smoke test theo §6.9 (resume đúng step, perplexity tiếng Việt giảm, eval loss tiếng Anh không tăng vọt) → đổi trạng thái bước 1 thành ✅.
