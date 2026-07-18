# Bước 1 — CPT (Continual Pretraining) · §6.1

**Mục tiêu:** bồi thêm ~400M token tiếng Việt cho `Qwen/Qwen3-1.7B-Base` (raw, chưa instruct) để nâng nền kiến thức/ngôn ngữ tiếng Việt trước khi dạy chat. Có 20% English replay chống catastrophic forgetting.

| Notebook | Vai trò | Chạy khi nào | GPU |
|---|---|---|---|
| `CPT-Step-A-Prepare-Qwen3-1.7B.ipynb` | Tải wiki-vi + fineweb-2 (vie) + fineweb-edu (en replay), tokenize, pack block 2048 token, push lên Hub | **Đúng 1 lần** | Không cần |
| `CPT-Step-B-Train-Qwen3-1.7B.ipynb` | QLoRA r=64 trên base 4-bit, lr=1.5e-5, `max_steps=3000`; tự resume từ checkpoint Hub, tự dừng trước 8h | **Run All mỗi session** đến khi đủ 3000 step | T4 x1 |

**Input:** `Qwen/Qwen3-1.7B-Base` + corpus tự build.
**Output trên Hub:**
- `<user>/vi-cpt-corpus-2048` — data đã pack (từ A; split `train` trộn 80/20 vi-en, `eval_vi`/`eval_en` tách riêng để đo forgetting)
- `<user>/Qwen3-1.7B-vi-cpt-ckpt` — checkpoint resume (từ B, mỗi 200 step)
- `<user>/Qwen3-1.7B-vi-cpt` — **bản merged cuối** → input cho Bước 2 (SFT)

**Tiêu chí xong (smoke test §6.9):** resume đúng step giữa các session; `eval_vi_loss` giảm dần; `eval_en_loss` không tăng vọt (thí nghiệm E3 trong report); model sinh tiếng Việt mạch lạc hơn base.
