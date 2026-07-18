# Bước 3 — Reward Model · §6.3

**Mục tiêu:** train model chấm điểm câu trả lời (value head trên nền SFT) từ preference data — nền tảng cho RLHF: RM cho biết câu nào "người dùng thích hơn".

**Notebook dự kiến:** `RM-Train-Qwen3-1.7B.ipynb` — ⬜ chưa viết. OpenRLHF `train_rm`, `--pretrain` trỏ vào bản SFT; script mẫu ở `docs/spec.md` §6.3.

**Input:** `<user>/Qwen3-1.7B-vi-sft` + dataset `thuanan/Vi-Alpaca-Preference`.
**Output:** `<user>/Qwen3-1.7B-vi-rm`.

**Rủi ro đã biết:** OpenRLHF + DeepSpeed có thể không cài được trên Kaggle → phương án dự phòng là Modal (xem §6.3).
