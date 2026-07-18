# Bước 4 — DPO (chính) / PPO (ablation) · §6.4

**Mục tiêu:** tối ưu model SFT theo preference — đẩy xác suất về phía câu trả lời được thích hơn. DPO là con đường chính (chỉ cần policy + reference, vừa 1 chiếc T4); PPO chỉ là ablation cho thí nghiệm E2.

| Notebook dự kiến | Vai trò | Hạ tầng |
|---|---|---|
| `DPO-Train-Qwen3-1.7B.ipynb` — ⬜ chưa viết | TRL `DPOTrainer` + QLoRA, cùng cơ chế resume | Kaggle T4 |
| `PPO-Train-Qwen3-1.7B` (script, không phải notebook) — ⬜ tùy chọn | OpenRLHF PPO qua Ray, 4 nhóm model | **Chỉ Modal multi-GPU — cấm chạy Kaggle** |

**Input:** `<user>/Qwen3-1.7B-vi-sft` (+ `<user>/Qwen3-1.7B-vi-rm` cho PPO) + dataset `thuanan/Vi-Alpaca-Preference` (PPO prompts: `thuanan/Prompt-Vi-Alpaca-Preference-2k`).
**Output:** `<user>/Qwen3-1.7B-vi-dpo` (và `<user>/Qwen3-1.7B-vi-ppo` nếu chạy ablation).
