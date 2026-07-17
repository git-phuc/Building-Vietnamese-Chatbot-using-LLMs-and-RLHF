# Building Vietnamese Chatbot using LLMs and RLHF

**Báo cáo tổng hợp: Pipeline bắt buộc (theo tài liệu AI VIETNAM) + Phân tích ý tưởng cải tiến**

> **Cách đọc tài liệu này** — chỉ có MỘT pipeline đang được thực thi, nằm ở **Mục 6**:
> - **Mục 6** = pipeline thật của đồ án (Qwen3-1.7B-Base raw → CPT → SFT → RM → DPO → Eval → Deploy). Làm việc gì cũng bắt đầu từ đây, sơ đồ tổng quan ở 6.0.
> - **Mục 2, 3, 4** = tài liệu tham khảo (baseline slide Llama, ý tưởng cải tiến gốc, kho dataset) — KHÔNG phải việc đang làm.
> - **Trạng thái hiện tại** (đang ở bước nào, xong gì rồi) tra ở bảng trong `README.md`.
> - Không có Mục 5 — đã gộp vào Mục 7 (Tài liệu tham khảo) khi tái cấu trúc; giữ nguyên số "Mục 6" vì các tài liệu khác đã tham chiếu tên này.

---

## 1. Project Statement

> Build a Vietnamese Chatbot application by fine-tuning an LLM on Vietnamese conversation data and then further improve the response from the model by utilizing RLHF (Reinforcement Learning from Human Feedback).

Nguồn: *Building Vietnamese Chatbot using LLMs and RLHF*, AI VIETNAM All-in-One Course (TA Session), Thuan Duong – Senior TA, 2026.

Tài liệu gốc trình bày một pipeline **thực dụng, đã chạy được**, dùng model nền `Llama-3.2-1B-Instruct`. Phần 2 dưới đây tóm tắt lại chính xác các bước **cần làm theo tài liệu này** (baseline bắt buộc). Phần 3 là phân tích của mình về những điểm có thể **cải tiến thêm** so với baseline, kèm cách áp dụng cụ thể vào từng bước của pipeline.

---

## 2. Phần cần làm: Pipeline theo slide AI VIETNAM (baseline — tóm tắt, không phải pipeline đang thực thi)

> Đây là **tài liệu tham khảo/đối chiếu** từ slide gốc, dùng model có sẵn `Llama-3.2-1B-Instruct`. Pipeline thật đang được thực hiện cho đồ án là **Mục 6**. Phần này chỉ tóm tắt ngắn gọn 11 bước để biết cấu trúc gốc; chi tiết hyperparameter/code đầy đủ xem lịch sử git hoặc slide gốc nếu cần.

11 bước, chia 4 khối: **SFT → Reward Model → RLHF (PPO) → Chat Interface**. Base: `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`, framework Unsloth (QLoRA).

| Khối | Bước | Tóm tắt |
|---|---|---|
| SFT | 1–4 | QLoRA (r=16, 7 target modules, `use_rslora=True`) trên `5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca` (12.7k hội thoại vi), `SFTTrainer` bs=32/lr=1e-4/max_steps=400 → merge & push `thuanan/Llama-3.2-1B-Instruct-Chat-sft` |
| Reward Model | 5–6 | Thu thập preference (`thuanan/Vi-Alpaca-Preference`, chosen/rejected) → OpenRLHF `train_rm` (ZeRO stage 2, LoRA r=16) → `thuanan/Llama-3.2-1B-RM-DPO` |
| RLHF/PPO | 7–8 | OpenRLHF `train_ppo_ray` trên Ray cluster (3 GPU), 4 nhóm model (Actor/Reference/Reward/Critic), KL penalty vs reference → `thuanan/Llama-3.2-1B-RLHF-2k-vi-alpaca` |
| Chat Interface | 9–11 | Convert GGUF (llama.cpp) → deploy Ollama + Open WebUI (Docker Compose) → rating export JSON làm nguồn iterative RLHF |

Alternative nhẹ (không nằm trong luồng chính): vLLM serve HF model trực tiếp + Gradio ChatInterface, dùng khi cần demo nhanh không cần GGUF/deploy production.

---

## 3. Phân tích ý tưởng cải tiến — áp dụng vào pipeline trên

Phần này là **đề xuất bổ sung của mình**, không nằm trong slide gốc. Mỗi ý tưởng được trình bày kèm vị trí áp dụng cụ thể vào pipeline ở Mục 2, mức độ ưu tiên, và chi phí triển khai ước tính — để bạn có thể chọn làm ablation study cho phần đóng góp riêng của đồ án.

### 3.1. Continual Pretraining (CPT) trước SFT — *chưa có trong slide*

**Vấn đề:** Slide dùng thẳng `Llama-3.2-1B-Instruct` (đã multilingual instruction-tuned) làm điểm xuất phát cho SFT. Tokenizer của Llama xử lý tiếng Việt (dấu thanh, âm tiết) không tối ưu bằng các model có exposure tiếng Việt tốt hơn (ví dụ Qwen2.5), dẫn đến sequence dài hơn và model "hiểu" tiếng Việt ở mức hạn chế hơn so với tiếng Anh.

**Cách áp dụng:** Chèn một bước **CPT** giữa "Load Base Model" (Bước 2.1) và "Apply QLoRA" (Bước 2.2):
- Data: vài GB – vài chục GB corpus tiếng Việt chất lượng (News, Wikipedia_vi, OSCAR-vi/CC100-vi đã lọc)
- Trộn lại 10-30% data gốc (English) để giảm catastrophic forgetting
- Learning rate thấp hơn SFT khoảng 5-10 lần, warmup dài
- 1-2 epoch là đủ, theo dõi eval loss trên tập tiếng Anh song song để phát hiện forgetting

**Ablation đề xuất:** So sánh model **có CPT** vs **không CPT** (baseline của slide), đo trên benchmark VMLU — đây là điểm khác biệt định lượng rõ ràng nhất để làm nổi bật đóng góp của đồ án.

**Chi phí:** Trung bình — cần thêm data pipeline + vài giờ compute trên 1 GPU với model 1B.

### 3.2. DPO thay thế PPO — giảm chi phí compute

**Vấn đề:** PPO trong slide (Bước 2.3, mục 2) cần chạy đồng thời 4 nhóm model qua Ray cluster (Actor, Reference, Reward, Critic) — tốn tài nguyên đáng kể, cần setup Docker + Ray phức tạp.

**Cách áp dụng:** OpenRLHF (framework đã dùng trong slide) hỗ trợ sẵn `train_dpo.py`. DPO bỏ qua hoàn toàn bước train Reward Model + Critic model, tối ưu trực tiếp trên cặp preference `(chosen, rejected)` — giảm số model cần load từ 4 xuống 2 (policy + reference).

**Ablation đề xuất:** Chạy DPO trên cùng `thuanan/Vi-Alpaca-Preference` dataset, so sánh với PPO ở 2 tiêu chí: (a) thời gian training + GPU-hour, (b) chất lượng response (win-rate qua LLM-as-judge). Đây là ablation có giá trị thực tiễn cao vì trả lời câu hỏi "có đáng để setup Ray/PPO phức tạp không, hay DPO đã đủ tốt".

**Chi phí:** Thấp — tái sử dụng data đã có, chỉ đổi script training.

### 3.3. Benchmark định lượng — slide mới chỉ có qualitative

**Vấn đề:** Slide 33 và 73 chỉ show output mẫu để đánh giá bằng mắt, không có con số cụ thể chứng minh RLHF thực sự cải thiện model.

**Cách áp dụng:** Thêm vào Bước 2.4 (trước khi deploy), một bước **Evaluation** độc lập:
- **VMLU**: đo kiến thức/reasoning tiếng Việt của SFT model vs RLHF model
- **Win-rate**: dùng GPT-4/Gemini làm LLM-judge, chấm theo cặp response (SFT vs RLHF) trên tập test held-out
- **Reward score trung bình**: dùng chính RM đã train, chấm trên tập test KHÔNG dùng để train RM/PPO (tránh đánh giá lạc quan)

**Chi phí:** Thấp — chủ yếu là compute inference, không cần train thêm.

### 3.4. Iterative RLHF — tận dụng feedback loop có sẵn

**Vấn đề:** Slide chỉ dừng ở 1 vòng SFT→RM→PPO. Trong khi đó, chính slide đã xây sẵn hạ tầng feedback (Open WebUI rating UI, export JSON ở Bước 2.4 mục 11) — nhưng chưa nối lại thành vòng lặp.

**Cách áp dụng:** Sau khi model RLHF v1 được deploy và người dùng dùng thử (thu thập feedback qua Open WebUI), lấy JSON feedback đó làm preference data mới (`chosen`/`rejected` dựa trên rating) để train **RM v2**, rồi chạy PPO v2 trên Actor đã là RLHF v1 (không phải SFT gốc). Đây chính là cách InstructGPT/ChatGPT thực sự vận hành.

**Chi phí:** Cao nhất trong các đề xuất — cần thời gian thu thập đủ feedback thực tế trước khi vòng 2 có ý nghĩa thống kê. Phù hợp làm phần "hướng phát triển" hơn là thực nghiệm chính.

### 3.5. Safety/harmlessness alignment

**Vấn đề:** Slide có ví dụ câu hỏi nhạy cảm về chủ quyền ("Trường Sa, Hoàng Sa là của ai?" — trang 15) nhưng không đề cập RLHF có tối ưu cho an toàn/harmlessness hay chỉ thuần helpfulness.

**Cách áp dụng:** Bổ sung một tập nhỏ preference data về safety (câu hỏi chính trị nhạy cảm, y tế, pháp luật) vào `Vi-Alpaca-Preference` trước khi train RM ở Bước 2.3, đảm bảo response được ưu tiên không chỉ vì "hay" mà còn "an toàn, cân bằng".

**Chi phí:** Trung bình — cần tự thu thập/tạo thêm vài trăm cặp preference cho domain nhạy cảm.

### 3.6. Tokenizer efficiency (liên quan 3.1)

**Vấn đề:** Không được đề cập trong slide. Tokenizer Llama split tiếng Việt kém hiệu quả hơn so với model tối ưu multilingual tốt hơn.

**Cách áp dụng:** Nếu làm ablation 3.1 (CPT), có thể đo thêm compression ratio (số token/câu tiếng Việt) trước/sau khi mở rộng vocab — tuy nhiên đây là optional, độ phức tạp cao, có thể bỏ qua nếu ưu tiên tốc độ hoàn thành đồ án.

**Chi phí:** Cao — có thể bỏ qua nếu không phải trọng tâm.

### 3.7. Tổng hợp mức ưu tiên

| Đề xuất | Vị trí áp dụng trong pipeline | Chi phí | Giá trị làm ablation |
|---|---|---|---|
| 3.2 DPO vs PPO | Bước 2.3 (thay thế RM+PPO) | Thấp | Cao |
| 3.3 Benchmark định lượng | Trước Bước 2.4 | Thấp | Cao |
| 3.1 Continual Pretraining | Trước Bước 2.2 | Trung bình | Cao |
| 3.5 Safety alignment | Bổ sung data ở Bước 2.3 | Trung bình | Trung bình |
| 3.4 Iterative RLHF | Sau Bước 2.4 | Cao | Trung bình (hướng phát triển) |
| 3.6 Tokenizer efficiency | Gắn với 3.1 | Cao | Thấp (optional) |

**Khuyến nghị:** Nếu thời gian có hạn, nên ưu tiên làm **3.2 (DPO vs PPO)** và **3.3 (benchmark định lượng)** trước — chi phí thấp, kết quả rõ ràng, dễ trình bày thành bảng so sánh trong báo cáo cuối kỳ. **3.1 (CPT)** là ablation có chiều sâu học thuật nhất nếu còn thời gian.

---

## 4. Nguồn dữ liệu public bổ sung (ngoài dataset trong slide)

| Dataset | Loại | Ghi chú |
|---|---|---|
| `bkai-foundation-models/vi-alpaca` | SFT | ~50K instructions, theo Self-Instruct + Stanford Alpaca |
| 5CD-AI Team (HuggingFace) | SFT/DPO/CoT/Coding | Loạt dataset dịch, đã bao gồm DPO Instruction dataset |
| `Bactrian-X` | SFT đa ngôn ngữ | 3.4M cặp, 52 ngôn ngữ (có tiếng Việt) |
| `OpenAssistant/oasst1` | Preference | 161K dialogue trees, đa ngôn ngữ |
| `linhtran92/viet_bud500` | ASR (nếu mở rộng voice) | ~500 giờ audio đa giọng vùng miền |

---

## 6. Pipeline tự triển khai (từ Qwen3-1.7B raw) — kế hoạch thực thi đồ án

> **Đây là pipeline chính đang được thực hiện cho đồ án**, khác với Mục 2 (baseline nguyên văn theo slide AI VIETNAM, dùng `Llama-3.2-1B-Instruct` có sẵn — giữ lại chỉ để tham khảo/đối chiếu) và Mục 3 (những ý tưởng cải tiến, nay đã được cụ thể hóa thành các bước dưới đây). Mục tiêu là **tự tay đi qua toàn bộ pipeline RLHF từ một base model raw** (chưa instruct-tuned) để hiểu bản chất từng bước, không chỉ dùng lại model đã huấn luyện sẵn. Sailor2 (model tiếng Việt/Đông Nam Á mạnh, đã public) chỉ đóng vai trò **reference benchmark** để so sánh chất lượng cuối cùng, không phải điểm xuất phát.

### 6.0. Sơ đồ tổng quan

```
┌───────────────────┐
│ Qwen3-1.7B-Base RAW │  (chưa instruct-tuned)
└─────────┬───────────┘
          │
          ▼
┌──────────────────────────┐
│ BƯỚC 1: CPT (tiếng Việt)  │  corpus vi (Wiki/News/OSCAR-vi/CC100-vi)
│  + 10-30% English replay  │  LR thấp (~5-10x < SFT), 1-2 epoch
└─────────┬──────────────────┘
          │  → Qwen3-1.7B-vi-cpt
          ▼
┌──────────────────────────┐
│ BƯỚC 2: SFT (QLoRA)       │  5CD-AI Vi-Multi-turn-Chat-Alpaca
│  r=16, target modules Qwen│  hoặc bkai-foundation/vi-alpaca
└─────────┬──────────────────┘
          │  → Qwen3-1.7B-vi-sft
          ▼
┌──────────────────────────┐
│ BƯỚC 3: Reward Model      │  Vi-Alpaca-Preference (chosen/rejected)
│  (OpenRLHF train_rm)      │
└─────────┬──────────────────┘
          │  → Qwen3-1.7B-vi-rm
          │
          ├─────────────────────────────┐
          ▼ (con đường CHÍNH)           ▼ (ablation, cần multi-GPU)
┌───────────────────────┐      ┌───────────────────────┐
│ BƯỚC 4a: DPO           │      │ BƯỚC 4b: PPO (Ray)     │
│ Kaggle-friendly        │      │ Modal / multi-GPU      │
│ chỉ cần policy + ref   │      │ Actor/Ref/RM/Critic    │
└──────────┬──────────────┘      └──────────┬─────────────┘
           │→ vi-dpo                        │→ vi-ppo
           └─────────────┬────────────────────┘
                          ▼
           ┌───────────────────────────┐
           │ BƯỚC 5: Evaluation         │  VMLU, win-rate LLM-judge,
           │ (SFT vs DPO vs PPO)        │  reward score trên held-out
           └─────────────┬───────────────┘
                          ▼
           ┌───────────────────────────┐
           │ BƯỚC 6: GGUF + Deploy      │  llama.cpp convert → Ollama
           │ (Ollama + Open WebUI)      │  + Open WebUI
           └─────────────┬───────────────┘
                          ▼
           ┌───────────────────────────┐
           │ BƯỚC 7: Feedback loop      │  Open WebUI rating → JSON
           │ (iterative RLHF, optional) │  → RM v2 → DPO v2
           └─────────────────────────────┘
```

Vì mọi bước train đều chạy trên **Kaggle T4 16GB (quota 30h/tuần, tối đa ~9h/session)**, không bước nào chắc chắn train xong trong 1 session — bắt buộc phải thiết kế **checkpoint/resume**:

```
Kaggle session (≤9h, quota 30h/tuần)
┌─────────────────────────────────────────────┐
│ Load checkpoint gần nhất từ HF Hub (private  │
│ repo) hoặc Kaggle Dataset output (nếu có)    │
│         │                                    │
│         ▼                                    │
│ Train N steps (save_steps nhỏ, vd mỗi 50-100)│
│         │                                    │
│         ▼                                    │
│ Session sắp hết giờ (~8h) → save checkpoint  │
│ → push lên HF Hub hoặc lưu Kaggle Dataset    │
│   version mới (Save & Run All → output)      │
└──────────────────┬────────────────────────────┘
                    │  session kết thúc, notebook dừng
                    ▼
         Session Kaggle tiếp theo: mở lại notebook,
         load checkpoint vừa lưu, lặp lại vòng trên
         cho đến khi đạt max_steps
```

Nguyên tắc áp dụng ở **mọi** bước train dưới đây: `TrainingArguments(save_strategy="steps", save_steps=<nhỏ>, resume_from_checkpoint=True)` + đẩy checkpoint lên `huggingface_hub` (`push_to_hub` hoặc `HfApi().upload_folder`) cuối mỗi session, hoặc dùng Kaggle Dataset làm nơi lưu output giữa các session nếu muốn tránh phụ thuộc mạng.

Mỗi checkpoint phải chứa đủ để resume đúng trạng thái (không chỉ trọng số): `adapter_model.bin`/`.safetensors` (LoRA), `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json` (chứa `global_step` hiện tại). `Trainer.save_state()`/`resume_from_checkpoint=<path>` của HF đã tự lưu đủ các thành phần này khi checkpoint được tạo qua `save_strategy="steps"` — chỉ cần đảm bảo cả thư mục checkpoint (không chỉ file trọng số) được đẩy lên Hub/Kaggle Dataset.

### 6.1. Bước 1 — Continual Pretraining (CPT) tiếng Việt

| Thành phần | Lựa chọn |
|---|---|
| Base model | `Qwen/Qwen3-1.7B-Base` (raw, chưa instruct) |
| Data tiếng Việt | `wikimedia/wikipedia` config `20231101.vi` (chất lượng cao, ~1.3M bài) + `HuggingFaceFW/fineweb-2` config `vie_Latn` (web corpus đã lọc, quy mô lớn, stream). *Không dùng CC100/OSCAR — dataset kiểu script/gated, `datasets` ≥ 3.0 không load được (đã vấp lỗi này trên Kaggle)* |
| English replay | `HuggingFaceFW/fineweb-edu` (`sample-10BT`, stream) — trộn ~20% để giảm catastrophic forgetting |
| Framework | **QLoRA rank lớn (r=64) trên base 4-bit**, KHÔNG full-parameter — ở 1.7B, full-param + Adam optimizer states (fp32, 2x params) đã vượt quá 16GB VRAM của T4 trước cả khi tính activations. LoRA r=64 (lớn hơn r=16 của SFT vì CPT cần cập nhật nhiều "kiến thức" hơn) là mức khả thi duy nhất trên 1 GPU T4 |
| Learning rate | ~5-10 lần thấp hơn SFT (1.5e-5), warmup 10% |
| Khối lượng train | **`max_steps=6000`** (không dùng epoch — data là mix nhiều nguồn nên "epoch" không có nghĩa). Effective batch 32 seq × 2048 token ≈ 65k token/step → ~400M token, vừa khớp 5-8 session Kaggle |
| Hub repos (3 cái, đừng nhầm) | `<user>/vi-cpt-corpus-2048` (data đã pack, private) · `<user>/Qwen3-1.7B-vi-cpt-ckpt` (checkpoint resume, private) · `<user>/Qwen3-1.7B-vi-cpt` (bản merged cuối cùng — input cho Bước 2) |

**Tổ chức thành 2 notebook Kaggle riêng biệt** — tách data prep khỏi train để mỗi session train không phải tokenize lại từ đầu:

#### Notebook A — chuẩn bị data (chạy đúng 1 lần, không cần GPU) — file: `notebooks/cpt_a_prepare_data.ipynb`

```python
# Cell 1: cài đặt + login
# !pip install -q datasets huggingface_hub
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
login(UserSecretsClient().get_secret("HF_TOKEN"))  # tạo secret HF_TOKEN trong Kaggle Add-ons trước

# Cell 2: tải 3 nguồn, lấy đủ ~400M token (tỉ lệ vi:en ≈ 80:20)
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

wiki_vi = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train")  # ~1.3M bài, tải hết

# Stream thẳng ra đĩa bằng from_generator — KHÔNG Dataset.from_list (gom list Python
# trong RAM → kernel died trên Kaggle, đã vấp). Số doc: vừa đủ ~400M token, không lấy thừa.
def stream_texts(repo, config, n, min_len=200):
    def gen():
        ds = load_dataset(repo, config, split="train", streaming=True)
        k = 0
        for r in ds:
            if len(r["text"]) > min_len:
                yield {"text": r["text"]}
                k += 1
                if k >= n:
                    break
    return Dataset.from_generator(gen)

web_vi     = stream_texts("HuggingFaceFW/fineweb-2", "vie_Latn", 250_000)   # ~250-350M token
fineweb_en = stream_texts("HuggingFaceFW/fineweb-edu", "sample-10BT", 100_000)  # ~100M token

# Cell 3: tokenize + pack thành block 2048 token (chuẩn CPT: nối tài liệu bằng EOS rồi cắt khúc)
from transformers import AutoTokenizer
from itertools import chain
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
BLOCK = 2048

def pack(ds):
    def tokenize(batch):
        return {"ids": [tok(t)["input_ids"] + [tok.eos_token_id] for t in batch["text"]]}
    def group(batch):
        flat = list(chain.from_iterable(batch["ids"]))
        n = (len(flat) // BLOCK) * BLOCK
        return {"input_ids": [flat[i:i+BLOCK] for i in range(0, n, BLOCK)]}
    ds = ds.select_columns(["text"]).map(tokenize, batched=True, remove_columns=["text"], num_proc=4)
    return ds.map(group, batched=True, batch_size=1000, remove_columns=["ids"], num_proc=4)

vi = pack(concatenate_datasets([wiki_vi.select_columns(["text"]),
                                web_vi])).shuffle(seed=42)
en = pack(fineweb_en).shuffle(seed=42)

# Cell 4: trộn 80/20, tách eval vi + eval en (theo dõi forgetting), push lên Hub
n_en = min(len(en), int(len(vi) * 0.25))            # 0.25 × vi ≈ 20% tổng
train = concatenate_datasets([vi.select(range(len(vi) - 1000)),
                              en.select(range(n_en - 1000))]).shuffle(seed=42)
DatasetDict({
    "train":   train,
    "eval_vi": vi.select(range(len(vi) - 1000, len(vi))),
    "eval_en": en.select(range(n_en - 1000, n_en)),
}).push_to_hub("<user>/vi-cpt-corpus-2048", private=True)
```

#### Notebook B — train (chạy lại mỗi session, tự resume) — file: `notebooks/cpt_b_train.ipynb`

```python
# Cell 1: cài đặt + login (như Notebook A) + hằng số
# !pip install -q unsloth
import os, time
from huggingface_hub import HfApi, login, snapshot_download
from kaggle_secrets import UserSecretsClient
login(UserSecretsClient().get_secret("HF_TOKEN"))

DATA_REPO  = "<user>/vi-cpt-corpus-2048"
CKPT_REPO  = "<user>/Qwen3-1.7B-vi-cpt-ckpt"    # checkpoint resume (adapter+optimizer+scheduler)
FINAL_REPO = "<user>/Qwen3-1.7B-vi-cpt"         # CHỈ push khi đã đạt max_steps
OUT_DIR, MAX_STEPS, BUDGET_H = "./cpt-checkpoints", 6000, 8.0

# Cell 2: kéo checkpoint mới nhất từ Hub về (None nếu là session đầu tiên)
api = HfApi()
api.create_repo(CKPT_REPO, private=True, exist_ok=True)
steps = sorted(int(f.split("/")[0].split("-")[1]) for f in api.list_repo_files(CKPT_REPO)
               if f.startswith("checkpoint-"))
last_ckpt = None
if steps:
    name = f"checkpoint-{steps[-1]}"
    snapshot_download(CKPT_REPO, allow_patterns=f"{name}/*", local_dir=OUT_DIR)
    last_ckpt = os.path.join(OUT_DIR, name)
print("Resume từ:", last_ckpt or "(bắt đầu mới)")

# Cell 3: model + LoRA r=64 (cấu hình cố định của Bước 1)
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen3-1.7B-Base", max_seq_length=2048, load_in_4bit=True, dtype=None)
model = FastLanguageModel.get_peft_model(
    model, r=64, lora_alpha=64, lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","up_proj","down_proj","o_proj","gate_proj"],
    use_rslora=True, use_gradient_checkpointing="unsloth", random_state=42)

# Cell 4: callback — mỗi lần save thì đẩy NGUYÊN thư mục checkpoint lên Hub
# (phải upload cả folder: optimizer.pt/scheduler.pt/rng_state/trainer_state.json thì mới
#  resume đúng được; push_to_hub=True của TrainingArguments chỉ đẩy adapter nên KHÔNG dùng),
# và tự dừng khi gần hết giờ session để không mất tiến độ
from transformers import TrainerCallback
class KaggleSessionCallback(TrainerCallback):
    def __init__(self): self.t0 = time.time()
    def on_save(self, args, state, control, **kw):
        name = f"checkpoint-{state.global_step}"
        api.upload_folder(repo_id=CKPT_REPO, folder_path=os.path.join(args.output_dir, name),
                          path_in_repo=name)
        if time.time() - self.t0 > BUDGET_H * 3600:
            control.should_training_stop = True

# Cell 5: train
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
data = load_dataset(DATA_REPO)
trainer = Trainer(
    model=model,
    train_dataset=data["train"],
    eval_dataset={"vi": data["eval_vi"], "en": data["eval_en"]},  # en tăng mạnh = forgetting
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[KaggleSessionCallback()],
    args=TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=4, gradient_accumulation_steps=8,
        learning_rate=1.5e-5, lr_scheduler_type="cosine", warmup_ratio=0.1,
        max_steps=MAX_STEPS,
        bf16=True, optim="paged_adamw_8bit",
        save_strategy="steps", save_steps=200, save_total_limit=2,
        eval_strategy="steps", eval_steps=600,
        logging_steps=20, report_to="none",
    ),
)
trainer.train(resume_from_checkpoint=last_ckpt)   # None → train mới; có path → nối tiếp đúng step

# Cell 6: CHỈ chạy khi state.global_step đã đạt MAX_STEPS —
# merge LoRA vào base rồi push bản merged; Bước 2 (SFT) load thẳng repo này
if trainer.state.global_step >= MAX_STEPS:
    model.push_to_hub_merged(FINAL_REPO, tokenizer, save_method="merged_16bit")
```

**Cách vận hành mỗi session Kaggle:** mở Notebook B → Run All → notebook tự kéo checkpoint mới nhất, train tiếp, tự dừng và push trước khi hết giờ. Lặp lại cho đến khi log hiện `global_step >= 6000` và Cell 6 push bản merged. Không phải sửa bất kỳ dòng code nào giữa các session.

**Vocab:** giữ nguyên tokenizer/vocab gốc của Qwen3 (không expand vocab tiếng Việt) — vocab Qwen3 (~151k, byte-level BPE, kế thừa từ dòng Qwen) đã cover tiếng Việt khá tốt, mở rộng vocab đòi hỏi train lại embedding mới từ đầu (tốn thêm compute, phức tạp resize + re-tie weights) mà lợi ích không chắc bù được chi phí trên quota Kaggle. Chỉ cân nhắc lại nếu sau khi CPT xong mà compression ratio (số token/câu tiếng Việt) vẫn kém rõ rệt.

Output cuối: `<user>/Qwen3-1.7B-vi-cpt` (merged 16-bit) trên Hub — đây là base model cho Bước 2.

### 6.2. Bước 2 — SFT (QLoRA)

> **Notebook chạy được thật: `notebooks/sft_train.ipynb`** — cùng cơ chế checkpoint/resume với `cpt_b_train.ipynb` (Run All mỗi session, tự kéo checkpoint từ Hub, push nguyên folder checkpoint mỗi `save_steps`, tự dừng theo ngân sách giờ). Chỉ chạy sau khi Bước 1 đã push bản merged.

| Hạng mục | Giá trị |
|---|---|
| Base nạp vào | `<user>/Qwen3-1.7B-vi-cpt` (output Bước 1, đã merge) — **không** phải Qwen3 gốc |
| Dataset | `5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca` (12.7k hội thoại multi-turn, dạng ShareGPT `from/value`); dự phòng: `bkai-foundation-models/vi-alpaca` |
| QLoRA | r=16, `lora_alpha=16`, 7 target modules như CPT, `use_rslora=True` |
| Train | bs=4 × grad_accum=8 (effective 32 như script gốc), lr=1e-4 (cao hơn CPT ~7x — đúng nguyên tắc CPT lr thấp hơn SFT 5-10x), cosine, warmup 5%, 2 epoch ≈ ~800 step ≈ 1 session T4 |
| Checkpoint | `save_steps=100` → push `<user>/Qwen3-1.7B-vi-sft-ckpt`; eval 200 hội thoại held-out mỗi 200 step |
| Output | `<user>/Qwen3-1.7B-vi-sft` (merged 16-bit, input cho Bước 3 RM) |

Hai quyết định thiết kế đáng ghi lại (đều nằm trong notebook):

**1. Chat template = ChatML thuần, không dùng template Qwen3 gốc.** Template mặc định của tokenizer Qwen3 chèn block `<think>` (thiết kế cho thinking mode) — không cần cho SFT thường và làm nhiễu format. ChatML dùng đúng các token `<|im_start|>/<|im_end|>` có sẵn trong vocab Qwen nên không phải thêm token mới:

```python
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
tokenizer = get_chat_template(tokenizer, chat_template="chatml")

ds = standardize_sharegpt(load_dataset("5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca", split="train"))
def to_text(ex):  # system prompt cố định + toàn bộ lượt hội thoại
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + ex["conversations"]
    return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}
```

**2. `train_on_responses_only` — chỉ tính loss trên lượt assistant.** Không cho model "học vẹt" câu hỏi của user; với data multi-turn điều này mask mọi lượt user, loss chỉ chảy qua các lượt assistant:

```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(trainer,
    instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
```

Smoke test cuối notebook: hỏi *"Xin chào! Bạn có thể giới thiệu về Việt Nam trong 3 câu không?"* qua chat template — model phải trả lời theo vai trợ lý tiếng Việt, dừng đúng `<|im_end|>` (đối chiếu tiêu chí 6.9).

### 6.3. Bước 3 — Reward Model

Tái dùng nguyên script OpenRLHF `train_rm` của Mục 2.3 (Bước 6), chỉ đổi `--pretrain` sang checkpoint SFT mới:

```bash
deepspeed --module openrlhf.cli.train_rm \
    --save_path ./checkpoint/Qwen3-1.7B-rm \
    --train_batch_size 96 --micro_train_batch_size 8 \
    --pretrain <user>/Qwen3-1.7B-vi-sft \
    --value_head_prefix score --bf16 \
    --max_epochs 1 --max_len 2048 --zero_stage 2 --learning_rate 5e-6 \
    --dataset thuanan/Vi-Alpaca-Preference --apply_chat_template \
    --chosen_key chosen --rejected_key rejected \
    --flash_attn --packing_samples --gradient_checkpointing \
    --lora_rank 16 --lora_alpha 32
```

**Rủi ro cần lưu ý:** OpenRLHF + DeepSpeed có thể khó cài trên Kaggle (image sẵn có, quyền hệ thống hạn chế) — nếu gặp lỗi cài đặt, phương án dự phòng là chuyển bước RM + DPO sang Modal (môi trường tự build image, kiểm soát được dependency). Output: `<user>/Qwen3-1.7B-vi-rm`.

### 6.4a. Bước 4a — DPO (con đường chính)

DPO được chọn làm con đường chính vì **chỉ cần load 2 model (policy + reference)**, không cần Ray cluster/multi-GPU như PPO — phù hợp Kaggle single-T4. Hai lựa chọn framework:

| Framework | Ưu điểm | Nhược điểm |
|---|---|---|
| OpenRLHF `train_dpo` | Cùng hệ sinh thái với RM ở Bước 3, tái dùng data format `chosen/rejected` sẵn có | Vẫn phụ thuộc DeepSpeed như Bước 3 |
| TRL `DPOTrainer` | Cài đặt nhẹ hơn, tích hợp tốt với Unsloth/QLoRA đã dùng ở Bước 2 | Cần convert data format sang chuẩn TRL |

**Khuyến nghị:** dùng **TRL `DPOTrainer`** kết hợp Unsloth QLoRA (tái dùng adapter từ Bước 2) — nhẹ hơn cho Kaggle, tránh rủi ro cài DeepSpeed đã nêu ở Bước 3.

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    output_dir="./dpo-checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    beta=0.1,  # hệ số điều chỉnh độ lệch so với reference model, tương đương vai trò KL penalty của PPO
    max_steps=400,
    save_strategy="steps",
    save_steps=50,
    bf16=True,
    push_to_hub=True,
    hub_model_id="<user>/Qwen3-1.7B-vi-dpo",
)
trainer = DPOTrainer(
    model=sft_model,          # <user>/Qwen3-1.7B-vi-sft
    ref_model=None,           # None → tự tạo reference từ bản sao model gốc (đủ cho LoRA)
    args=dpo_config,
    train_dataset=preference_dataset,  # thuanan/Vi-Alpaca-Preference
)
trainer.train(resume_from_checkpoint=True)
```

Output: `<user>/Qwen3-1.7B-vi-dpo`.

### 6.4b. Bước 4b — PPO (ablation, không bắt buộc)

Giữ nguyên script Ray/PPO của Mục 2.3 (Bước 7), chỉ đổi `--pretrain`/`--reward_pretrain` sang checkpoint Qwen3 tương ứng:

```bash
ray start --head --node-ip-address 0.0.0.0 --num-gpus 3

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --pretrain <user>/Qwen3-1.7B-vi-sft \
    --reward_pretrain <user>/Qwen3-1.7B-vi-rm \
    --prompt_data thuanan/Prompt-Vi-Alpaca-Preference-2k \
    --input_key context_messages --apply_chat_template \
    --actor_learning_rate 5e-7 --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 --normalize_reward \
    --micro_train_batch_size 4 --train_batch_size 64 \
    --micro_rollout_batch_size 8 --rollout_batch_size 512 \
    --max_epochs 2 --prompt_max_len 1024 --generate_max_len 1024 \
    --zero_stage 3 --bf16 --gradient_checkpointing
```

**Quan trọng:** bước này **KHÔNG khả thi trên Kaggle** (chỉ có 1 GPU T4) — 4 nhóm model (Actor/Reference/Reward/Critic) của PPO cần multi-GPU. **Chỉ chạy trên Modal** (hoặc môi trường multi-GPU tương đương). Đây là ablation để so sánh với DPO (Mục 3.2), không phải deliverable bắt buộc — chỉ làm nếu còn thời gian/ngân sách Modal. Output: `<user>/Qwen3-1.7B-vi-ppo`.

### 6.5. Bước 5 — Evaluation định lượng

Cụ thể hóa Mục 3.3 thành bước thực thi, chạy trước khi deploy. Mỗi task nên gắn với 1 benchmark + 1 baseline riêng để có điểm neo so sánh, thay vì chỉ nhìn số tuyệt đối:

| Task | Benchmark | Baseline đối chiếu | Đo ablation nào |
|---|---|---|---|
| a. World knowledge/reasoning tiếng Việt | VMLU (trắc nghiệm đa lĩnh vực) | VMLU leaderboard (vmlu.ai) — so với model cùng size class | Base vs CPT (6.1) |
| b. Chất lượng generation tự nhiên | Perplexity trên held-out corpus vi | So Base vs CPT (kỳ vọng giảm sau CPT); eval loss English song song để phát hiện forgetting | CPT (6.1) |
| c. Follow instruction (dịch, tóm tắt, QA) | Test set đa task tự tạo | LLM-judge (GPT-4o/Gemini) chấm 1-10 | SFT (6.2) |
| d. Preference win-rate | Cặp response cùng prompt | LLM-judge chấm A/B, SFT vs DPO vs PPO | DPO/PPO (6.4a/6.4b) |
| e. RM validation | Accuracy chosen > rejected | RM chấm trên test set không dùng để train RM | RM (6.3) |
| f. Safety (định tính, optional) | Test case tay (chính trị nhạy cảm/y tế/pháp luật) | Đánh giá tay theo Mục 3.5 | Safety alignment (nếu làm) |

Ghi chú khi đối chiếu:
- **Kỳ vọng điểm VMLU thực tế**: với model size ~1.7B, mức điểm hợp lý để kỳ vọng là **~35-48** (nhỉnh hơn class 0.5B một chút; so với model instruct lớn hơn thường đạt 60+) — đừng lấy điểm của model lớn làm mốc để đánh giá pipeline nhỏ này thất bại.
- Tham khảo thêm số liệu từ VinaLLaMA (arXiv:2312.11011, benchmark ARC/HellaSwag/MMLU/TruthfulQA tiếng Việt) và các bài benchmark LLM tiếng Việt gần đây để hiệu chỉnh kỳ vọng theo đúng size class.
- **Reward score trung bình**: dùng chính RM đã train ở Bước 3, chấm trên tập held-out tách riêng từ `Vi-Alpaca-Preference` (không dùng để train RM/DPO/PPO) để tránh đánh giá lạc quan.
- **(Tuỳ chọn)** So sánh với **Sailor2** như reference benchmark — không kỳ vọng vượt qua (Sailor2 lớn hơn nhiều và có nhiều dữ liệu hơn), mục đích là có điểm neo để đánh giá pipeline tự làm.

### 6.6. Bước 6 — Convert GGUF + Deploy

Tái dùng gần như nguyên vẹn Mục 2.4 (Bước 9-10), chỉ đổi tên model nguồn:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py <user>/Qwen3-1.7B-vi-dpo \
    --outfile <user>/Qwen3-1.7B-vi-dpo.gguf --outtype bf16
```

Deploy qua Docker Compose (Ollama + Open WebUI) — cấu hình y hệt Mục 2.4 Bước 10, chỉ đổi tên model khi `ollama run hf.co/<user>/Qwen3-1.7B-vi-dpo-gguf`.

### 6.7. Bước 7 — Feedback loop (iterative RLHF, hướng mở rộng)

Cụ thể hóa Mục 3.4: sau khi có bản deploy đầu tiên (DPO v1) và người dùng thử qua Open WebUI, xuất JSON rating → xây preference data v2 (`chosen`/`rejected` theo rating) → train RM v2 → DPO v2 trên actor đã là DPO v1 (không phải SFT gốc). **Đây là hướng phát triển sau khi hoàn thành Bước 1-6, không phải deliverable bắt buộc của đồ án.**

### 6.8. Bảng compute & checkpoint theo từng bước

| Bước | Compute | Ước tính session Kaggle (T4) | Checkpoint mỗi |
|---|---|---|---|
| 1. CPT | Kaggle (QLoRA r=64, không full-param) | 5-8 session (~9h/session) cho max_steps=6000 (~400M token) | 200 steps (xem 6.1 Notebook B) |
| 2. SFT | Kaggle | 1 session | 100 steps (giống Mục 2, max_steps=400) |
| 3. Reward Model | Kaggle (rủi ro cài DeepSpeed) → fallback Modal | 1-2 session | theo epoch (data preference thường nhỏ) |
| 4a. DPO | Kaggle | 1-2 session | 50 steps |
| 4b. PPO (ablation) | **Modal bắt buộc** (multi-GPU) | N/A (không chạy Kaggle) | theo cấu hình OpenRLHF |
| 5. Evaluation | Kaggle (chỉ inference) | 1 session | không cần |
| 6. GGUF + Deploy | Local/Kaggle (convert) + máy chủ riêng (Docker deploy) | 1 session convert | không cần |
| 7. Feedback loop | Kaggle (train lại RM/DPO v2) | như Bước 3-4a | như Bước 3-4a |

### 6.9. Verification / smoke test cho từng bước

Trước khi coi một bước là "xong", kiểm tra nhanh theo các tiêu chí sau (không thay thế Evaluation ở Bước 5, chỉ để bắt lỗi sớm ngay trong lúc train):

- **Checkpoint/resume hoạt động đúng**: chạy train tới step N bất kỳ (vd step 50) → chủ động dừng kernel → mở lại notebook, load checkpoint → xác nhận `global_step` tiếp tục đúng từ N và loss tiếp tục giảm (không nhảy lại từ đầu, không loss spike bất thường).
- **CPT học được gì**: perplexity trên held-out tiếng Việt giảm rõ so với model raw; eval loss trên tập tiếng Anh không tăng quá nhiều (dấu hiệu forgetting nếu tăng mạnh).
- **SFT hoạt động đúng**: inference thử trên 5-10 prompt tiếng Việt bất kỳ → model trả lời đúng format chat, không lặp từ, không chuyển sang tiếng Anh khi được hỏi tiếng Việt.
- **DPO cải thiện so với SFT**: win-rate theo LLM-judge của DPO ≥ 60% so với SFT trên tập test pairs held-out (nếu thấp hơn, kiểm tra lại `beta` hoặc chất lượng preference data trước khi kết luận DPO không hiệu quả).
- **End-to-end deploy**: sau khi convert GGUF và chạy `ollama run`, chat thử qua Open WebUI xác nhận model phản hồi được, không lỗi load.

---

## 7. Tài liệu tham khảo

1. Duong, Thuan. *Building Vietnamese Chatbot using LLMs and RLHF*. AI VIETNAM All-in-One Course (TA Session), 2026. — Tài liệu gốc của pipeline Mục 2.
2. Nguyen et al. *Efficient Finetuning Large Language Models For Vietnamese Chatbot*. arXiv:2309.04646, 2023.
3. Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023.
4. OpenRLHF — https://github.com/OpenRLHF/OpenRLHF
5. Unsloth — https://github.com/unslothai/unsloth
6. llama.cpp — https://github.com/ggml-org/llama.cpp
7. `awesome-vietnamese-nlp` (vndee) — https://github.com/vndee/awsome-vietnamese-nlp
8. HuggingFace Vietnamese datasets — https://huggingface.co/datasets?language=language:vi&sort=trending
9. Qwen3 model card — https://huggingface.co/Qwen/Qwen3-1.7B-Base
10. VMLU benchmark — https://vmlu.ai/
11. Sailor2 — https://huggingface.co/sail/Sailor2-1B-Chat (reference benchmark, không phải model xuất phát)
