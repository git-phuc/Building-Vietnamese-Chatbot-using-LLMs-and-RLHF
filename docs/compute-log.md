# Compute log — thời gian & chi phí thực tế

Sổ ghi chép mọi session train/chuẩn bị data, để viết mục *Experimental Setup / Compute Budget* trong report (`report/main.tex`). **Cập nhật ngay sau mỗi session** — số liệu lấy từ timestamp trong tab Logs của Kaggle (dòng `[heartbeat]`/`Đã push checkpoint-...`) và từ W&B (run duration, `perf/sec_per_step`).

## Thông số cố định (đo thực tế, dùng lại trong report)

| Đại lượng | Giá trị | Nguồn |
|---|---|---|
| Phần cứng | Kaggle T4 x2 (16GB/GPU), free tier | — |
| Quota | 30 GPU-giờ/tuần; 1 session batch sống ~11.5–12h (đo 2026-07-19; T4 x2 tính giờ như x1) | session CPT #1 |
| Tốc độ CPT | ~95–103 s/step (QLoRA r=64, 4-bit, seq 2048) | session CPT #1 |
| Token/step (CPT) | batch hiệu dụng 64 × 2048 = **131.072 token/step** | cấu hình §6.1 |
| Mục tiêu CPT | 3.000 step ≈ **393M token** ≈ ~80 GPU-giờ ≈ 8–9 session | cấu hình §6.1 |
| Corpus CPT | 384.367 block × 2048 ≈ 787M token (3.000 step đi qua ~½ corpus, <1 epoch) | Notebook A |
| Chi phí tiền | **0 USD** đến nay (Kaggle free). Modal chỉ dùng nếu chạy PPO ablation (§6.4b) — ghi $ vào đây nếu phát sinh | — |

## Nhật ký session

| # | Ngày | Bước | Máy | Thời lượng | Tiến độ (step) | Ghi chú |
|---|---|---|---|---|---|---|
| 0 | 2026-07-18 | CPT — Notebook A (pack data) | Kaggle CPU (không tốn quota GPU) | ~?h *(điền nếu nhớ)* | — | Output: `vi-cpt-corpus-2048` (train 80/20 vi-en + eval_vi + eval_en) |
| 1 | 2026-07-18 | CPT — Notebook B, chạy thử interactive | Kaggle T4 | ~2h *(ước lượng)* | 0 → ~44, **không giữ được gì** | Dừng trước checkpoint đầu (step 200) → mất trắng; bài học: đừng dừng tay trước mốc save |
| 2 | 2026-07-19 | CPT — Notebook B, batch session đầu | Kaggle T4 x2 | 11h31' (41.459s) | 0 → **400** | ckpt-200 @5h46', ckpt-400 @11h30'; ~95→103 s/step; notebook bản cũ nên không có log loss/W&B train; dừng an toàn bằng budget stop |

**Tổng GPU đã dùng: ~13.5h · CPT: 400/3000 step (~52M token)**

## Cách điền một dòng mới (sau mỗi session)

1. **Thời lượng**: timestamp cuối cùng trong tab Logs (giây → giờ), hoặc W&B run duration + ~30' setup.
2. **Tiến độ**: dòng `global_step = X / 3000` cuối log, hoặc tên checkpoint mới nhất trên Hub.
3. **Ghi chú**: s/step trung bình (từ `[heartbeat]`), eval loss nếu qua mốc 600, sự cố nếu có.
4. Cập nhật dòng **Tổng** và cột Trạng thái trong `README.md`.

## Ước tính còn lại (cập nhật dần khi có số đo mới)

| Bước | Ước tính | Căn cứ |
|---|---|---|
| CPT (còn 2.600 step) | ~72–75 GPU-giờ ≈ 7 session ≈ 2.5–3 tuần quota | ~100 s/step, ~350 step/session (BUDGET_H=10) |
| SFT (~800 step, r=16) | ~1 session (≤10h) | step SFT nhẹ hơn CPT; đo lại khi chạy |
| RM / DPO / Eval | chưa có số đo — điền sau | — |
| PPO ablation (tùy chọn) | Modal, tính $ riêng | §6.4b |
