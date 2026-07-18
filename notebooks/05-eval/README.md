# Bước 5 — Evaluation · §6.5

**Mục tiêu:** đo lường định lượng cho 3 thí nghiệm của report (E1: CPT có đáng không; E2: DPO vs PPO; E3: replay vs forgetting) và so mốc tham chiếu Sailor2-1B.

**Notebook dự kiến:** `Eval-Qwen3-1.7B.ipynb` — ⬜ chưa viết. Ba thước đo:
1. **VMLU** — benchmark kiến thức tiếng Việt (multiple choice)
2. **LLM-judge win-rate** — so cặp câu trả lời, chấm 2 chiều đảo vị trí để khử position bias
3. **RM score** — điểm reward trung bình trên held-out

**Quy tắc bất di bất dịch:** held-out split chưa từng dùng train RM/DPO/PPO.

**Input:** mọi checkpoint đã train (`-cpt`, `-sft`, `-dpo`, và base gốc để làm mốc E1).
**Output:** bảng số liệu điền vào `report/report.md` §6 và `report/main.tex` (Results).
