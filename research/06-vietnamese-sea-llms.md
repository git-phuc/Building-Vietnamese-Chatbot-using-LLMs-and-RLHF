# LLM tiếng Việt & Đông Nam Á — "người đi trước" của pipeline này

## Topic là gì

Các model tiếng Việt/SEA mạnh đều đi đúng công thức mà Section 6 đang tái tạo ở quy mô nhỏ: **base đa ngôn ngữ → CPT trên corpus bản địa (có replay) → SFT → preference alignment**. Không model nào bỏ qua CPT để SFT thẳng — đây là bằng chứng thực nghiệm cho quyết định giữ bước §6.1. Sailor2 đồng thời là **reference benchmark** ở §6.5 (đối chiếu, không phải mục tiêu phải đạt).

## Paper chính

- **Sailor2: Sailing in South-East Asia with Inclusive Multilingual LLMs** (Dou et al., 2025) — [arXiv:2502.12982](https://arxiv.org/abs/2502.12982)
  Quan trọng nhất cho đồ án: từ Qwen2.5, CPT 500B token (400B SEA + **100B replay** — đúng tỉ lệ ~20% ta mô phỏng), rồi SFT + preference tuning. Được viết như "cookbook" đầy đủ 5 khâu: data, pretrain, post-train, customize, eval — gần như bản phóng to của Section 6.

- **Sailor: Open Language Models for South-East Asia** (Dou et al., 2024) — [arXiv:2404.03608](https://arxiv.org/abs/2404.03608)
  Bản tiền nhiệm, chi tiết hơn về khâu CPT: tỉ lệ trộn ngôn ngữ, deduplication, ablation về replay — đọc kèm 01-continual-pretraining.

- **SeaLLMs — Large Language Models for Southeast Asia** (Nguyen et al., 2023) — [arXiv:2312.00738](https://arxiv.org/abs/2312.00738)
  Cùng công thức CPT + SFT + DPO cho SEA từ Llama-2; có bàn về mở rộng vocabulary cho ngôn ngữ non-Latin (liên quan ý tưởng tokenizer efficiency ở Section 3).

- **VinaLLaMA: LLaMA-based Vietnamese Foundation Model** (Nguyen et al., 2023) — [arXiv:2312.11011](https://arxiv.org/abs/2312.11011)
  CPT 800B token tiếng Việt trên Llama-2 + SFT bằng 1M mẫu synthetic; case tiếng Việt thuần đúng công thức của ta.

- **PhoGPT: Generative Pre-training for Vietnamese** (Nguyen et al., VinAI, 2023) — [arXiv:2311.02945](https://arxiv.org/abs/2311.02945)
  Hướng ngược lại: pretrain **từ đầu** 4B trên 102B token tiếng Việt với tokenizer riêng 20K vocab — mốc so sánh "from scratch vs CPT" hữu ích khi viết báo cáo.

- **Qwen3 Technical Report** (Qwen Team, 2025) — [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)
  Base model của ta (`Qwen3-1.7B-Base`): kiến trúc, data pretrain 36T token đa ngôn ngữ — đọc để biết model đã có sẵn bao nhiêu tiếng Việt trước khi ta CPT.

## Đọc gì trước

Sailor2 (cookbook khớp đồ án nhất) → Sailor 1 (chi tiết CPT) → VinaLLaMA/PhoGPT (bối cảnh tiếng Việt, trích dẫn trong báo cáo).
