# Bước 6 — GGUF + Deploy · §6.6 (và feedback loop §6.7)

**Mục tiêu:** đưa model DPO ra sản phẩm dùng được: convert GGUF (llama.cpp) → chạy local bằng Ollama → giao diện chat Open WebUI. Stretch goal (§6.7): export rating từ WebUI → preference data v2 → vòng RLHF thứ hai.

**Dự kiến:** ⬜ chưa bắt đầu — phần này chuyển từ notebook sang **code product** (script + config trong `app/`, cấu trúc xem `docs/orchestrator.md`). Đây là giai đoạn kích hoạt cơ chế orchestrator nhiều agent.

**Input:** `<user>/Qwen3-1.7B-vi-dpo`.
**Output:** `<user>/Qwen3-1.7B-vi-dpo-gguf` + hệ thống chạy được (Ollama + Open WebUI).
