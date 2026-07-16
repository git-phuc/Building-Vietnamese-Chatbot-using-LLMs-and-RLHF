# Building Vietnamese Chatbot using LLMs and RLHF

**Báo cáo tổng hợp: Pipeline bắt buộc (theo tài liệu AI VIETNAM) + Phân tích ý tưởng cải tiến**

---

## 1. Project Statement

> Build a Vietnamese Chatbot application by fine-tuning an LLM on Vietnamese conversation data and then further improve the response from the model by utilizing RLHF (Reinforcement Learning from Human Feedback).

Nguồn: *Building Vietnamese Chatbot using LLMs and RLHF*, AI VIETNAM All-in-One Course (TA Session), Thuan Duong – Senior TA, 2026.

Tài liệu gốc trình bày một pipeline **thực dụng, đã chạy được**, dùng model nền `Llama-3.2-1B-Instruct`. Phần 2 dưới đây tóm tắt lại chính xác các bước **cần làm theo tài liệu này** (baseline bắt buộc). Phần 3 là phân tích của mình về những điểm có thể **cải tiến thêm** so với baseline, kèm cách áp dụng cụ thể vào từng bước của pipeline.

---

## 2. Phần cần làm: Pipeline theo slide AI VIETNAM (baseline)

Đây là pipeline 11 bước được slide mô tả (trang 21), chia thành 4 khối lớn: **Fine-tuning (SFT) → Reward Model → RLHF (PPO) → Chat Interface**.

### 2.1. Setup môi trường

| Thành phần | Công cụ | Ghi chú |
|---|---|---|
| Fine-tuning framework | [Unsloth](https://unsloth.ai/) | 2x faster, giảm >70% VRAM so với Hugging Face + FA2 |
| Model hub | Hugging Face (`huggingface-cli login` / token) | Cần token dạng `hf_xxx` từ https://huggingface.co/settings/tokens |
| Base model | `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` | `MAX_SEQ_LENGTH = 2048`, `load_in_4bit=True`, `dtype=torch.bfloat16` |

### 2.2. Fine-tuning LLMs cho Chatbot (SFT)

**Bước 1 — Apply QLoRA**

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","up_proj","down_proj","o_proj","gate_proj"],
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```
Kết quả: `trainable params: 11,272,192 / 1,247,086,592` (~0.9% trainable) — đúng bản chất của QLoRA: base model giữ 4-bit, chỉ train adapter 16-bit.

**Bước 2 — Chuẩn bị dataset**

- Dataset: [`5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca`](https://huggingface.co/datasets/5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca) — 12.7k hội thoại multi-turn tiếng Việt, định dạng `{"from": "human"/"gpt", "value": ...}`.
- Convert sang chat format chuẩn (system/user/assistant) bằng `tokenizer.apply_chat_template`.
- System prompt cố định: *"Bạn là một trợ lý AI thân thiện, hãy trả lời bằng tiếng Việt."*
- Tokenize với `max_length=2048`, `padding="max_length"`.

**Bước 3 — Train SFT**

```python
training_args = TrainingArguments(
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_steps=400,
    bf16=True,
)
trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer)
trainer.train()
```
Kết quả thực nghiệm trong slide: loss giảm từ 1.527 → 1.277 sau 400 step (~4/5 epoch trên 12,697 mẫu).

**Bước 4 — Save & Push to Hub**

Merge checkpoint, `model.push_to_hub_merged(...)` và `tokenizer.push_to_hub(...)` → ví dụ `thuanan/Llama-3.2-1B-Instruct-Chat-sft`.

### 2.3. Training LLMs với RLHF

**Bước 5 — Thu thập preference data**

- Cơ chế: hiển thị 2 response cho cùng 1 prompt, người dùng chọn *chosen* / *rejected* (giống giao diện feedback của ChatGPT).
- Dataset kết quả: [`thuanan/Vi-Alpaca-Preference`](https://huggingface.co/datasets/thuanan/Vi-Alpaca-Preference) — format `{id, chosen, rejected, prompt, chosen_score, rejected_score}`, xây từ Alpaca + Dolly tiếng Việt.

**Bước 6 — Train Reward Model** (dùng [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF))

```bash
deepspeed --module openrlhf.cli.train_rm \
    --save_path ./checkpoint/Llama-3.2-1B-rm-dpo \
    --train_batch_size 96 --micro_train_batch_size 8 \
    --pretrain thuanan/Llama-3.2-1B-Instruct-Chat-sft \
    --value_head_prefix score --bf16 \
    --max_epochs 1 --max_len 2048 --zero_stage 2 --learning_rate 5e-6 \
    --dataset thuanan/Vi-Alpaca-Preference --apply_chat_template \
    --chosen_key chosen --rejected_key rejected \
    --flash_attn --packing_samples --gradient_checkpointing \
    --lora_rank 16 --lora_alpha 32
```
Kết quả log thực tế: `acc=0.958`, `chosen_reward≈1.7`, `reject_reward≈-1.5` — RM phân biệt tốt cặp preference. Sau đó merge LoRA (`openrlhf.cli.lora_combiner`) và push lên Hub (`thuanan/Llama-3.2-1B-RM-DPO`).

**Bước 7 — RLHF với PPO** (Ray cluster)

Kiến trúc gồm 4 nhóm model chạy song song qua Ray: **Actor (vLLM inference)**, **Reference model**, **Reward model**, **Critic model** — mỗi nhóm có nhiều replica trên nhiều GPU, đồng bộ qua ZeRO.

```bash
ray start --head --node-ip-address 0.0.0.0 --num-gpus 3

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --pretrain thuanan/Llama-3.2-1B-Instruct-Chat-sft \
    --reward_pretrain thuanan/Llama-3.2-1B-RM-DPO \
    --prompt_data thuanan/Prompt-Vi-Alpaca-Preference-2k \
    --input_key context_messages --apply_chat_template \
    --actor_learning_rate 5e-7 --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 --normalize_reward \
    --micro_train_batch_size 4 --train_batch_size 64 \
    --micro_rollout_batch_size 8 --rollout_batch_size 512 \
    --max_epochs 2 --prompt_max_len 1024 --generate_max_len 1024 \
    --zero_stage 3 --bf16 --gradient_checkpointing
```

Cơ chế PPO trong RLHF (theo diagram slide 35): Actor (tuned LM) sinh response → Reward model chấm điểm → trừ đi **KL penalty** so với Initial LM (`-λ·D_KL(π_PPO‖π_base)`) để tránh model "đánh lừa" reward model bằng câu trả lời vô nghĩa → cập nhật Actor bằng PPO.

Kết quả training thực tế: `reward` tăng dần từ ~1.8 lên ~3.0, `return` tăng tương ứng theo global step — cho thấy model học được để tối ưu theo reward.

**Bước 8 — Push model RLHF final lên Hub** → `thuanan/Llama-3.2-1B-RLHF-2k-vi-alpaca`.

### 2.4. Building Chat Interface

**Bước 9 — Convert sang GGUF** (dùng [llama.cpp](https://github.com/ggml-org/llama.cpp))

```bash
git clone https://github.com/ggerganov/llama.cpp.git
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py llama-rlhf \
    --outfile thuanan/Llama-3.2-1B-RLHF-2k-vi-alpaca.gguf --outtype bf16
```
Push file `.gguf` lên Hugging Face Hub qua `HfApi().upload_file(...)`.

**Bước 10 — Deploy với Ollama + Open WebUI** (Docker Compose)

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]
  open-webui:
    image: ghcr.io/open-webui/open-webui:latest
    ports: ["3001:8080"]
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
```

```bash
docker exec -it ollama bash
ollama run hf.co/thuanan/Llama-3.2-1B-RLHF-2k-vi-alpaca-gguf
```

**Bước 11 — Test đa model song song + thu thập feedback**

Open WebUI cho phép so sánh nhiều model cùng lúc (SFT vs RLHF) trên cùng 1 prompt, và có sẵn cơ chế rating (1-10) + tag lý do (*accurate information*, *followed instructions*...) — dữ liệu này xuất ra JSON, chính là nguồn để **lặp lại vòng RLHF tiếp theo**.

*(Ngoài luồng chính, slide cũng giới thiệu phương án nhẹ hơn dùng vLLM serve trực tiếp model HF (không cần convert GGUF) kết hợp Gradio ChatInterface — phù hợp khi cần demo nhanh, không cần deploy production.)*

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

## 5. Tài liệu tham khảo

1. Duong, Thuan. *Building Vietnamese Chatbot using LLMs and RLHF*. AI VIETNAM All-in-One Course (TA Session), 2026. — Tài liệu gốc của pipeline Mục 2.
2. Nguyen et al. *Efficient Finetuning Large Language Models For Vietnamese Chatbot*. arXiv:2309.04646, 2023.
3. Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023.
4. OpenRLHF — https://github.com/OpenRLHF/OpenRLHF
5. Unsloth — https://github.com/unslothai/unsloth
6. llama.cpp — https://github.com/ggml-org/llama.cpp
7. `awesome-vietnamese-nlp` (vndee) — https://github.com/vndee/awsome-vietnamese-nlp
8. HuggingFace Vietnamese datasets — https://huggingface.co/datasets?language=language:vi&sort=trending
