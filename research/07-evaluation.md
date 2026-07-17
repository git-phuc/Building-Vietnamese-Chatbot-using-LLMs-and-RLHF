# Evaluation — §6.5

## Topic là gì

§6.5 dùng 3 thước đo bổ trợ nhau, mỗi cái bắt một loại tiến bộ khác nhau:

1. **VMLU** (multiple-choice, 58 môn học) — đo *kiến thức* tiếng Việt: kỳ vọng nhích lên sau CPT, gần như không đổi sau DPO.
2. **LLM-as-a-judge win-rate** — đo *chất lượng hội thoại* mà benchmark trắc nghiệm không thấy: kỳ vọng tăng rõ sau SFT và DPO.
3. **RM reward score** trên held-out split — đo trực tiếp mục tiêu preference; **held-out phải chưa từng dùng train RM/DPO**, nếu không là tự chấm bài mình dạy.

Bẫy cần biết của LLM-judge: position bias (đảo thứ tự 2 câu trả lời rồi chấm 2 lần), verbosity bias (thích câu dài), self-enhancement bias (judge thiên vị output giống phong cách của chính nó).

## Paper / tài nguyên chính

- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** (Zheng et al., 2023) — [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
  Paper chuẩn hoá phương pháp LLM-judge: đo agreement judge-vs-người (~80%, ngang người-vs-người), liệt kê và cách khử các bias kể trên. Đọc trước khi viết script win-rate ở §6.5.

- **VMLU Benchmarks: A comprehensive benchmark toolkit for Vietnamese LLMs** (ACL 2025) — [aclanthology.org/2025.acl-long.563](https://aclanthology.org/2025.acl-long.563/) · leaderboard: [vmlu.ai](https://vmlu.ai)
  Bộ 10,880 câu trắc nghiệm 58 môn (STEM, nhân văn, xã hội...) — benchmark kiến thức tiếng Việt chính của §6.5; bản ACL 2025 mở rộng thêm reading comprehension/reasoning/conversation.

- **ViLLM-Eval: A Comprehensive Evaluation Suite for Vietnamese Large Language Models** (2024) — [arXiv:2404.11086](https://arxiv.org/abs/2404.11086)
  Bộ eval tiếng Việt bổ sung (LAMBADA-vi, exam...) — dự phòng/đối chiếu nếu cần nhiều hơn VMLU.

- *(bối cảnh)* **Sailor2** ([arXiv:2502.12982](https://arxiv.org/abs/2502.12982)) chương evaluation — cách một team SEA-LLM thiết kế eval đa ngôn ngữ, gồm cả SeaBench/win-rate kiểu judge; Sailor2-1B là mốc reference-benchmark của ta.

## Đọc gì trước

Zheng 2023 (bắt buộc trước khi làm judge win-rate) → VMLU toolkit paper (hiểu benchmark chấm gì) → ViLLM-Eval nếu cần mở rộng.
