# SFT / Instruction Tuning — §6.2

## Topic là gì

Supervised Fine-Tuning = dạy model **hành vi trợ lý** bằng các cặp (instruction, response) chất lượng cao — chuyển một base model chỉ biết "viết tiếp văn bản" thành model biết trò chuyện theo format. Insight quan trọng nhất của mảng này là **superficial alignment hypothesis**: SFT chủ yếu dạy *style và format*, còn kiến thức/năng lực ngôn ngữ đến từ pretraining — vì vậy vài chục nghìn conversation không thay được CPT (đây chính là câu trả lời cho câu hỏi "CPT có tác dụng không khi đằng sau đã có conv tiếng Việt").

Trong đồ án: SFT QLoRA r=16 (`use_rslora=True`) từ checkpoint CPT trên `5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca`, system prompt cố định.

## Paper chính

- **LIMA: Less Is More for Alignment** (Zhou et al., 2023) — [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)
  Chỉ 1,000 mẫu SFT chọn lọc kỹ đủ tạo trợ lý mạnh từ LLaMA-65B → bằng chứng trực tiếp cho superficial alignment hypothesis: "hầu hết kiến thức được học trong pretraining, alignment chỉ dạy format". Paper quan trọng nhất để hiểu vai trò tương đối CPT vs SFT.

- **Training language models to follow instructions with human feedback** (InstructGPT — Ouyang et al., 2022) — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
  Paper định hình cả pipeline SFT → RM → RLHF mà đồ án đang tái tạo. Phần SFT (bước 1 trong 3 bước) là chuẩn mực mà §6.2 đi theo.

- **The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning** (URIAL — Lin et al., 2023) — [arXiv:2312.01552](https://arxiv.org/abs/2312.01552)
  Đo chính xác alignment thay đổi *token nào*: chủ yếu là các token mở đầu, discourse marker, style — phần kiến thức gần như giữ nguyên từ base. Củng cố định lượng cho LIMA.

## Đọc gì trước

LIMA (hiểu SFT dạy gì và không dạy gì) → InstructGPT phần SFT (pipeline chuẩn) → URIAL (nếu muốn bằng chứng token-level).
