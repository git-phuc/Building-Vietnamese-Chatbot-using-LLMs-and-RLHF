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
| 3 | 2026-07-19 | CPT — Notebook B, session 2 (LỖI) | Kaggle T4 | ~0.4h | 400 → 400 (không mất gì) | `wandb.init()` timeout 90s → `CommError` ném trong `on_train_begin` giết cả train trước step đầu tiên. Fix: notebook tự init W&B (timeout 300s), lỗi thì tắt W&B và train tiếp |
| 4 | 2026-07-19 | CPT — Notebook B, session 3 (LỖI, mất 200 step) | Kaggle T4 | ~6.5h | 400 → 400 (**mất 200 step đã train**) | Resume + log + W&B đều chạy đúng; loss ~3.0, 105-106 s/step. Nhưng eval ở step 600 **OOM**: lúc eval model trả nguyên logits [16, 2048, 151k vocab] → fp32 cần 18.5 GiB > 14.56 GiB của T4 (lúc train không sao vì Unsloth fused loss không materialize logits). Eval chạy TRƯỚC save cùng mốc 600 → chết trước khi push ckpt-600. Fix: `per_device_eval_batch_size=2`, `prediction_loss_only=True`; sau đó hạ `save_steps` 200→20 (push mỗi ~35', retry 3 lần, Hub tự dọn giữ 2 ckpt; eval giữ mốc 600) — từ giờ sự cố kiểu gì cũng chỉ mất ≤20 step |

| 5 | 2026-07-20 | CPT — Notebook B, session 4 (THÀNH CÔNG trọn vẹn) | Kaggle T4 x2 | ~10.0h train (36.040s) + setup | 400 → **737** | Bản notebook save-20 chạy chuẩn đầu-cuối: 104 s/step; eval@600 batch 2 hết ~8,5'/tập (rẻ hơn dự tính), eval_vi=5.935 / eval_en=5.788 (mốc eval đầu tiên — theo dõi xu hướng, không so tuyệt đối với train loss ~3.0); push + dọn Hub OK (còn [600, 737]); budget stop tự save tại 737. Train loss ~3.00–3.05 đi ngang (mới 0,12 epoch, LR còn gần đỉnh — bình thường) |
| 6 | 2026-07-20 | CPT — Notebook B, session 5 (THÀNH CÔNG) | Kaggle T4 x2 | ~10.0h (36.191s) | 737 → **1190** | 92-94 s/step (nhanh hơn session trước, không qua mốc eval); loss ~3.0-3.06 đi ngang, grad_norm ổn định; budget stop tự save tại 1190, push + dọn Hub OK (còn [1000, 1190]) |
| 7 | 2026-07-20 | CPT — Notebook B, session 6 (THÀNH CÔNG, retry hoạt động đúng) | Kaggle T4 x2 | ~10.0h (36.257s) | 1190 → **1546** | 98-99 s/step; eval@1200 batch 2: eval_vi=6.144 / eval_en=5.929; budget stop tại 1546, push checkpoint-1546 gặp lỗi tạm thời `503 Service Unavailable` từ HF Hub lần 1/3 — cơ chế retry (sleep 30s) tự phục hồi, push thành công lần 2, không mất step nào; dọn Hub OK (còn [1400, 1546]) |
| 8 | 2026-07-20 | CPT — Notebook B, session 7 (THÀNH CÔNG) | Kaggle T4 x2 | ~10.0h (36.207s) | 1546 → **1858** | 112 s/step (chậm hơn 2 session trước — có thể do T4 đơn thay vì x2, hoặc contention); eval@1800 batch 2 ~9,2'/tập: eval_vi=6.138 / eval_en=5.954 — **nhỉnh hơn eval@600** (vi 5.935→6.144(@1200)→6.138, en 5.788→5.929(@1200)→5.954): chưa có xu hướng giảm rõ rệt qua 3 mốc đầu, cần theo dõi tiếp ở 2400/3000 trước khi kết luận (train loss vẫn ổn định ~3.0, không có dấu hiệu overfit/collapse); push + dọn Hub OK (còn [1800, 1858]); budget stop tự save tại 1858 |

**Tổng GPU đã dùng: ~61h · CPT: 1858/3000 step (~244M token)**

## Cách điền một dòng mới (sau mỗi session)

1. **Thời lượng**: timestamp cuối cùng trong tab Logs (giây → giờ), hoặc W&B run duration + ~30' setup.
2. **Tiến độ**: dòng `global_step = X / 3000` cuối log, hoặc tên checkpoint mới nhất trên Hub.
3. **Ghi chú**: s/step trung bình (từ `[heartbeat]`), eval loss nếu qua mốc 600, sự cố nếu có.
4. Cập nhật dòng **Tổng** và cột Trạng thái trong `README.md`.

## Ước tính còn lại (cập nhật dần khi có số đo mới)

| Bước | Ước tính | Căn cứ |
|---|---|---|
| CPT (còn 1.142 step) | ~33 GPU-giờ ≈ 3-4 session ≈ 1-1.5 tuần quota | 92-112 s/step đo session 4-7, ~310-370 step/session (BUDGET_H=10) |
| SFT (~800 step, r=16) | ~1 session (≤10h) | step SFT nhẹ hơn CPT; đo lại khi chạy |
| RM / DPO / Eval | chưa có số đo — điền sau | — |
| PPO ablation (tùy chọn) | Modal, tính $ riêng | §6.4b |
