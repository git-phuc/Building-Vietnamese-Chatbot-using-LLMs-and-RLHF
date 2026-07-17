# PEFT: LoRA / QLoRA / rsLoRA — mọi bước train trên Kaggle T4

## Topic là gì

Parameter-Efficient Fine-Tuning là điều kiện tồn tại của cả đồ án: full-param fine-tune 1.7B **không vừa T4 16GB** (riêng Adam optimizer state fp32 đã vượt 16GB). Chuỗi kỹ thuật:

- **LoRA**: đóng băng weight gốc, chỉ train cặp ma trận low-rank A·B chèn vào các projection layer → giảm số tham số train ~1000 lần.
- **QLoRA**: load weight gốc ở 4-bit (NF4) + train LoRA fp16 bên trên + paged optimizer → 1.7B train được trong vài GB VRAM.
- **rsLoRA**: sửa hệ số scale từ α/r thành α/√r — với rank cao (r=64 ở CPT) scale gốc làm adapter học quá chậm; lý do spec đặt `use_rslora=True`.

Trong đồ án: CPT dùng QLoRA r=64 (cần capacity lớn để "nhét" kiến thức ngôn ngữ), SFT/DPO dùng r=16 (chỉnh hành vi cần ít capacity hơn), tất cả qua Unsloth.

## Paper chính

- **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021) — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
  Paper gốc; phần đáng đọc là giả thuyết "intrinsic rank thấp" giải thích vì sao update ít tham số vẫn đủ.

- **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023) — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
  NF4 quantization + double quantization + paged optimizers; chứng minh 4-bit base + LoRA fp16 khớp chất lượng fine-tune 16-bit. Đây là kỹ thuật mọi notebook train của ta đứng trên.

- **A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA** (rsLoRA — Kalajdzievski, 2023) — [arXiv:2312.03732](https://arxiv.org/abs/2312.03732)
  Chứng minh scale α/r làm gradient "tắt" ở rank cao, đề xuất α/√r — căn cứ cho `use_rslora=True` trong config CPT r=64 và SFT.

## Đọc gì trước

QLoRA (quan trọng nhất — hiểu vì sao notebook load 4-bit) → LoRA (nền) → rsLoRA (ngắn, đọc để hiểu 1 dòng config).
