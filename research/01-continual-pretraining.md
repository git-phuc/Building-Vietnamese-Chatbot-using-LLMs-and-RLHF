# Continual Pretraining (CPT) — §6.1

## Topic là gì

Continual pretraining = tiếp tục pretrain một model đã có sẵn (next-token prediction trên văn bản thô, không phải hội thoại) trên corpus mới — ở đây là tiếng Việt — để bơm thêm **kiến thức + độ trôi chảy ngôn ngữ** trước khi alignment. Hai vấn đề kỹ thuật trung tâm:

1. **Catastrophic forgetting** — học tiếng Việt nhưng quên tiếng Anh/năng lực reasoning gốc. Thuốc chữa tiêu chuẩn: **replay** (trộn lại một phần data giống phân phối pretrain gốc — pipeline của ta dùng ~20% fineweb-edu tiếng Anh) + learning rate nhỏ có warmup/decay.
2. **Chất lượng corpus** — web crawl thô rất bẩn; các pipeline lọc như FineWeb quyết định phần lớn chất lượng cuối.

Trong đồ án: notebook `CPT-Step-A-Prepare-Qwen3-1.7B.ipynb` (pack corpus 2048-token) + `CPT-Step-B-Train-Qwen3-1.7B.ipynb` (QLoRA r=64, lr=1.5e-5, max_steps=6000, eval_vi/eval_en để phát hiện forgetting).

## Paper chính

- **Simple and Scalable Strategies to Continually Pre-train Large Language Models** (Ibrahim et al., 2024) — [arXiv:2403.08763](https://arxiv.org/abs/2403.08763)
  Paper "sách giáo khoa" cho CPT: chứng minh LR re-warming + re-decaying + **replay data cũ** là đủ để CPT đạt chất lượng ngang train lại từ đầu. Là căn cứ trực tiếp cho quyết định trộn 20% English replay và dùng cosine schedule trong §6.1.

- **Don't Stop Pretraining: Adapt Language Models to Domains and Tasks** (Gururangan et al., 2020) — [arXiv:2004.10964](https://arxiv.org/abs/2004.10964)
  Paper kinh điển khai sinh khái niệm DAPT/TAPT: tiếp tục pretrain trên domain đích luôn giúp downstream task, kể cả khi model gốc đã lớn. "Domain" ở đây của ta là ngôn ngữ tiếng Việt.

- **An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning** (Luo et al., 2023) — [arXiv:2308.08747](https://arxiv.org/abs/2308.08747)
  Đo đạc forgetting một cách hệ thống khi fine-tune tiếp LLM 1B–7B: model càng nhỏ quên càng nặng — đúng cỡ 1.7B của ta, giải thích vì sao phải theo dõi `eval_en` loss chứ không thể chủ quan.

- **Vi-Mistral-X: Building a Vietnamese Language Model with Advanced Continual Pre-training** (2024) — [arXiv:2403.15470](https://arxiv.org/abs/2403.15470)
  Case study CPT tiếng Việt cụ thể trên Mistral: cùng bài toán với §6.1, đáng đọc để đối chiếu lựa chọn corpus và hyperparameter.

## Paper về data corpus

- **The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale** (Penedo et al., 2024) — [arXiv:2406.17557](https://arxiv.org/abs/2406.17557)
  Mô tả pipeline lọc web-crawl của FineWeb/fineweb-edu — nguồn English replay của ta.

- **FineWeb2: One Pipeline to Scale Them All** (Penedo et al., 2025) — [arXiv:2506.20920](https://arxiv.org/abs/2506.20920)
  Mở rộng FineWeb sang 1000+ ngôn ngữ (20TB) — chính là nguồn `fineweb-2 vie_Latn` mà Notebook A dùng làm corpus tiếng Việt chính.

## Đọc gì trước

Ibrahim 2024 (vì sao replay + LR schedule) → Gururangan 2020 (vì sao CPT giúp downstream) → FineWeb2 (data ta đang dùng được lọc thế nào).
