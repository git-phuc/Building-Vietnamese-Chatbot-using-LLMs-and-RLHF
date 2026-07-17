# Research — paper nền tảng cho từng bước pipeline

Mỗi file = một topic, gồm: mô tả topic, vì sao liên quan đến đồ án (khớp §6.x trong spec), danh sách paper kèm link + tóm tắt 1-2 câu, và gợi ý đọc paper nào trước.

| File | Topic | Khớp bước pipeline |
|---|---|---|
| [01-continual-pretraining.md](01-continual-pretraining.md) | Continual Pretraining (CPT), catastrophic forgetting, data corpus | §6.1 — CPT tiếng Việt |
| [02-sft-instruction-tuning.md](02-sft-instruction-tuning.md) | SFT / instruction tuning, superficial alignment | §6.2 — SFT |
| [03-rlhf-reward-model.md](03-rlhf-reward-model.md) | RLHF cổ điển: reward model + PPO | §6.3 — RM, §6.4b — PPO ablation |
| [04-dpo-preference-optimization.md](04-dpo-preference-optimization.md) | DPO và các biến thể preference optimization | §6.4a — DPO (con đường chính) |
| [05-peft-qlora.md](05-peft-qlora.md) | LoRA / QLoRA / rsLoRA — train 1.7B trên T4 16GB | Mọi bước train trên Kaggle |
| [06-vietnamese-sea-llms.md](06-vietnamese-sea-llms.md) | LLM tiếng Việt & Đông Nam Á (Sailor2, SeaLLMs, VinaLLaMA, PhoGPT) | Toàn pipeline — "người đi trước" đã làm đúng công thức này |
| [07-evaluation.md](07-evaluation.md) | Đánh giá: VMLU, LLM-as-a-judge | §6.5 — Evaluation |

> Quy ước: mỗi paper ghi dạng `Tên (tác giả chính, năm) — arXiv link`. Chỉ liệt kê paper thực sự phục vụ quyết định thiết kế trong spec, không sưu tầm tràn lan.
