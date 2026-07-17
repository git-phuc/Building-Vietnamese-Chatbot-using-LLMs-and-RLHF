# DPO & Preference Optimization — §6.4a (con đường chính)

## Topic là gì

DPO (Direct Preference Optimization) chứng minh rằng bài toán RLHF-với-KL-penalty có nghiệm đóng, nên có thể **bỏ hẳn RM và RL loop**: train trực tiếp trên cặp (chosen, rejected) bằng một classification loss, chỉ cần policy + reference model. Đây là lý do §6.4a chọn DPO làm con đường chính — 2 model (thay vì 4 của PPO) vừa khít một Kaggle T4 khi dùng QLoRA.

Điểm cần nhớ khi đọc: DPO chỉ **tái phân phối xác suất giữa những output model đã sinh được** — nó không tạo năng lực mới. Trần chất lượng do CPT+SFT quyết định (xem [02-sft-instruction-tuning.md](02-sft-instruction-tuning.md)).

Trong đồ án: TRL `DPOTrainer` + adapter QLoRA tái dụng từ bước SFT, data `thuanan/Vi-Alpaca-Preference`, output `<user>/Qwen3-1.7B-vi-dpo`.

## Paper chính

- **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** (Rafailov et al., 2023) — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
  Paper gốc, bắt buộc đọc. Phần 4 (derive loss từ nghiệm đóng của RLHF objective) là phần đáng hiểu kỹ nhất — nó giải thích vì sao β (KL coefficient) là hyperparameter quan trọng nhất của DPOTrainer.

- **A General Theoretical Paradigm to Understand Learning from Human Preferences** (IPO — Azar et al., 2023) — [arXiv:2310.12036](https://arxiv.org/abs/2310.12036)
  Chỉ ra DPO dễ overfit khi preference gần deterministic và đề xuất IPO fix; đọc nếu thấy DPO train xong model trả lời cực đoan/lặp.

- **KTO: Model Alignment as Prospect Theoretic Optimization** (Ethayarajh et al., 2024) — [arXiv:2402.01306](https://arxiv.org/abs/2402.01306)
  Không cần cặp chosen/rejected — chỉ cần nhãn tốt/xấu rời lẻ. Liên quan trực tiếp §6.7: rating 👍/👎 export từ Open WebUI đúng dạng KTO, không phải dạng cặp.

- **SimPO: Simple Preference Optimization with a Reference-Free Reward** (Meng et al., 2024) — [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)
  Bỏ luôn reference model (nhẹ hơn nữa cho T4), dùng average log-prob có length normalization; ứng viên ablation nếu DPO chật VRAM.

## Đọc gì trước

DPO (bắt buộc, trước khi chạy §6.4a) → KTO (khi làm §6.7 feedback loop) → IPO/SimPO (chỉ khi gặp vấn đề tương ứng).
