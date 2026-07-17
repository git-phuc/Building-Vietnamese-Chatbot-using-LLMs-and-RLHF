# RLHF cổ điển: Reward Model + PPO — §6.3, §6.4b

## Topic là gì

RLHF cổ điển gồm 2 nửa: (1) **Reward Model** — train một model chấm điểm câu trả lời từ dữ liệu so sánh cặp (chosen/rejected), dùng Bradley-Terry loss; (2) **PPO** — dùng RM làm reward signal để tối ưu policy bằng reinforcement learning, có KL penalty giữ policy không trôi xa khỏi SFT. PPO mạnh nhưng đắt: cần đồng thời 4 model (actor, critic, RM, reference) → lý do §6.4b chỉ là ablation trên Modal, không chạy được trên 1 T4.

Trong đồ án: RM train bằng OpenRLHF `train_rm` trên `thuanan/Vi-Alpaca-Preference` (§6.3); RM còn được tái dụng làm **evaluator** ở §6.5 (reward score trên held-out split).

## Paper chính

- **Deep Reinforcement Learning from Human Preferences** (Christiano et al., 2017) — [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
  Paper gốc của toàn bộ ý tưởng: học reward function từ so sánh cặp của con người thay vì reward thủ công.

- **Learning to Summarize from Human Feedback** (Stiennon et al., 2020) — [arXiv:2009.01325](https://arxiv.org/abs/2009.01325)
  Lần đầu áp RLHF vào LLM (tóm tắt văn bản); định nghĩa công thức RM + PPO + KL penalty mà mọi framework (kể cả OpenRLHF) dùng đến nay.

- **Training language models to follow instructions with human feedback** (InstructGPT — Ouyang et al., 2022) — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
  SFT → RM → PPO thành pipeline hoàn chỉnh cho trợ lý đa nhiệm — chính là khung mà Section 6 tái tạo ở quy mô 1.7B tiếng Việt.

- **Training a Helpful and Harmless Assistant with RLHF** (Bai et al., Anthropic, 2022) — [arXiv:2204.05862](https://arxiv.org/abs/2204.05862)
  Chi tiết thực nghiệm phong phú nhất về train RM (calibration, scaling) và trade-off helpful/harmless; nguồn tham khảo tốt khi debug RM ở §6.3.

- **Proximal Policy Optimization Algorithms** (Schulman et al., 2017) — [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
  Thuật toán RL nền của bước PPO — đọc để hiểu clipped objective, không cần đọc sâu nếu chỉ chạy DPO.

- **OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework** (Hu et al., 2024) — [arXiv:2405.11143](https://arxiv.org/abs/2405.11143)
  Paper của framework ta dùng cho `train_rm` và PPO ablation; giải thích kiến trúc Ray + vLLM tách 4 model group — lý do PPO bắt buộc multi-GPU.

- *(đọc thêm)* **Constitutional AI: Harmlessness from AI Feedback** (Bai et al., 2022) — [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
  Thay human feedback bằng AI feedback (RLAIF) — liên quan trực tiếp nếu làm §6.7 feedback loop v2 mà thiếu người gán nhãn.

## Đọc gì trước

InstructGPT (khung tổng) → Stiennon 2020 (công thức RM+PPO cụ thể) → OpenRLHF (hiểu tool đang dùng). Christiano/Schulman đọc sau nếu muốn gốc rễ.
