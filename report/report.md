# RLHF on a Shoestring: Building a Vietnamese Chatbot from a Raw Base Model on Free-Tier GPUs

> **Report chính thức của đồ án** — viết dần theo tiến độ pipeline (xem bảng trạng thái ở `README.md`). Các mục đánh dấu ⬜ là chỗ chờ số liệu thật từ §6.5; phần văn đã hoàn chỉnh có thể dùng thẳng cho báo cáo/slide bảo vệ. Bản LaTeX để nộp: `report/main.tex` (+ `report/references.bib`) — **viết bằng tiếng Anh, layout paper 2 cột** (abstract full-width dài ~1/3 trang); file .md này là bản nháp làm việc tiếng Việt — sửa nội dung bên nào thì cập nhật bên kia.
>
> Tên rút gọn khi cần: *RLHF on a Shoestring*.

## Abstract (bản nháp — cập nhật số liệu sau §6.5)

Đồ án xây dựng và vận hành **trọn vẹn pipeline RLHF** cho chatbot tiếng Việt, xuất phát từ một base model thô chưa qua instruction tuning (`Qwen/Qwen3-1.7B-Base`): Continual Pretraining (CPT) trên ~400M token tiếng Việt → Supervised Fine-Tuning trên 12.7k hội thoại multi-turn → Reward Model → Direct Preference Optimization → đánh giá định lượng → triển khai GGUF/Ollama. Toàn bộ huấn luyện chạy trên hạ tầng miễn phí (Kaggle T4 16GB, quota 30h/tuần) nhờ cơ chế checkpoint-resume qua HuggingFace Hub cho phép chia nhỏ mọi bước theo phiên. Đồ án trả lời bằng số liệu ba câu hỏi thiết kế: giá trị của CPT đối với tiếng Việt, mức độ thay thế của DPO đối với PPO, và hiệu quả của English replay trong việc chống catastrophic forgetting. ⬜ *[chèn 1-2 câu kết quả chính sau khi có số]*

## 1. Đặt vấn đề & động cơ

Đa số đồ án "chatbot tiếng Việt" cùng cấp đi theo lối: lấy một model đã instruction-tuned sẵn (Llama-Instruct, Qwen-Instruct) → fine-tune nhẹ → demo. Cách đó ra sản phẩm nhanh nhưng bỏ qua toàn bộ phần khó và đáng học nhất: **model biết chat từ đâu ra?** Đồ án này chọn con đường ngược lại — bắt đầu từ raw weights chỉ biết dự đoán token kế tiếp, tự tay xây từng tầng biến nó thành trợ lý tiếng Việt, đúng quy trình công nghiệp mà InstructGPT (Ouyang et al., 2022) định hình và Sailor2 (Dou et al., 2025) áp dụng cho ngôn ngữ Đông Nam Á.

Ràng buộc tự đặt: **chỉ dùng compute miễn phí** (Kaggle T4, session ≤9h). Ràng buộc này không phải điểm yếu phải giấu mà là một phần của đề bài: mọi kỹ thuật trong đồ án đều phải sống được trong điều kiện session bị cắt bất kỳ lúc nào.

## 2. Đóng góp

Đóng góp của đồ án không nằm ở việc tạo ra một model vượt các LLM tiếng Việt hiện có — điều bất khả thi ở quy mô tài nguyên sinh viên — mà ở ba điểm:

1. **Pipeline hoàn chỉnh từ gốc.** Xây dựng và vận hành trọn vẹn chuỗi CPT → SFT → RM → DPO → Eval → Deploy từ base model thô, thay vì tinh chỉnh trên model đã instruction-tuned sẵn như đa số công trình cùng cấp. Mỗi tầng có notebook độc lập, chạy được thật, kèm smoke test.

2. **Ba câu hỏi thiết kế được trả lời bằng số liệu** (chi tiết §5) — những câu các công bố lớn thường bỏ ngỏ ở quy mô nhỏ:
   - Continual pretraining có đáng chi phí với tiếng Việt không? (CPT-ablation trên VMLU + win-rate)
   - DPO thay được PPO đến đâu, với chi phí bao nhiêu? (chất lượng + GPU-hour)
   - English replay 20% có thực sự chặn được catastrophic forgetting không? (đường `eval_en_loss` xuyên suốt CPT)

3. **Quy trình khả tái lập trên hạ tầng miễn phí.** Cơ chế checkpoint-resume qua HuggingFace Hub (push nguyên checkpoint folder — adapter + optimizer + scheduler + RNG — mỗi `save_steps`, tự dừng trước ngân sách giờ, session sau Run All là nối tiếp đúng step) biến mọi bước huấn luyện thành chuỗi phiên 8 giờ có thể lặp vô hạn. Sailor2 công bố cookbook cho người có hàng trăm GPU; đồ án này là cookbook cho người có một chiếc T4 mượn.

Sailor2 được dùng làm **mốc đối chiếu** trong đánh giá, không phải điểm xuất phát.

## 3. Công trình liên quan

Tổng hợp đầy đủ theo topic (kèm arXiv link + tóm tắt) tại thư mục [`research/`](research/README.md). Các mốc chính đối với từng luận điểm của đồ án:

- **Vai trò CPT vs SFT**: LIMA (Zhou et al., 2023) — superficial alignment hypothesis; Ibrahim et al. (2024) — replay + LR schedule cho CPT.
- **Công thức chuẩn cho ngôn ngữ bản địa**: Sailor/Sailor2, SeaLLMs, VinaLLaMA, PhoGPT — tất cả đều CPT trước khi alignment, không có ngoại lệ.
- **RLHF & thay thế**: InstructGPT (SFT→RM→PPO), DPO (Rafailov et al., 2023) — nghiệm đóng bỏ RM/RL loop.
- **Điều kiện tồn tại trên T4**: QLoRA (Dettmers et al., 2023), rsLoRA (Kalajdzievski, 2023).

## 4. Pipeline

Thiết kế chi tiết từng bước tại spec chính (`docs/spec.md`, Mục 6); notebook chạy được tại `notebooks/`. Tóm tắt:

| Bước | Nội dung | Notebook | Output (HF Hub) |
|---|---|---|---|
| 1. CPT | QLoRA r=64, ~400M token (wiki-vi + fineweb-2 vie_Latn + 20% fineweb-edu replay), 3000 step (batch hiệu dụng 64) | `CPT-Step-A-Prepare-Qwen3-1.7B` + `CPT-Step-B-Train-Qwen3-1.7B` | `Qwen3-1.7B-vi-cpt` |
| 2. SFT | QLoRA r=16, 12.7k hội thoại multi-turn, ChatML, loss chỉ trên lượt assistant | `SFT-Train-Qwen3-1.7B` | `Qwen3-1.7B-vi-sft` |
| 3. RM | OpenRLHF `train_rm`, preference data Vi-Alpaca | ⬜ | `Qwen3-1.7B-vi-rm` |
| 4. DPO | TRL `DPOTrainer` (chính); PPO trên Modal (ablation) | ⬜ | `Qwen3-1.7B-vi-dpo` |
| 5. Eval | VMLU + LLM-judge win-rate + RM score trên held-out | ⬜ | — |
| 6. Deploy | GGUF → Ollama + Open WebUI | ⬜ | `...-dpo-gguf` |

Lựa chọn nền tảng đáng bảo vệ: **Qwen3-1.7B-Base thay vì Llama** — tokenizer ~152k vocab train đa ngôn ngữ cho fertility tiếng Việt thấp gần bằng tiếng Anh (Llama-2 32k vocab băm vụn âm tiết có dấu; đây là thứ CPT không sửa được nếu không mở rộng vocab như SeaLLMs), và điểm xuất phát tiếng Việt đã khá — với ngân sách token cố định, xuất phát cao + bồi nhẹ đi xa hơn xuất phát thấp + liều lớn (VinaLLaMA cần 800B token để cứu Llama-2, gấp 2000 lần ngân sách của đồ án).

## 5. Thiết kế thực nghiệm & ablation

Ba thí nghiệm, mỗi cái một biến duy nhất:

| # | Câu hỏi | Điều kiện so sánh | Thước đo | Kết quả |
|---|---|---|---|---|
| E1 | CPT có đáng không? | SFT-từ-CPT vs SFT-từ-base-thô (mọi thứ khác giữ nguyên) | VMLU, LLM-judge win-rate | ⬜ |
| E2 | DPO thay được PPO? | DPO (T4) vs PPO (Modal multi-GPU), cùng preference data | Win-rate, RM score, GPU-hour | ⬜ |
| E3 | Replay chống forgetting? | Đường `eval_vi_loss` / `eval_en_loss` trong suốt 3000 step CPT | Loss curve từ `trainer_state.json` | ⬜ |

Quy tắc đánh giá: held-out split chưa từng dùng train RM/DPO/PPO; LLM-judge chấm 2 chiều (đảo vị trí) để khử position bias (Zheng et al., 2023); Sailor2-1B làm mốc đối chiếu tham khảo.

## 6. Kết quả

⬜ *Điền dần khi các bước hoàn thành — mỗi bước xong dán bảng số + 1 đoạn nhận xét. Nguồn số liệu: log `trainer_state.json` các checkpoint (CPT/SFT), script eval §6.5.*

## 7. Câu hỏi phản biện dự kiến & phương án trả lời

| Câu hỏi | Trả lời |
|---|---|
| "Của em hơn gì những model đang có?" | Không cạnh tranh chất lượng với model train bằng trăm GPU — đồ án tái tạo *cách các model đó được tạo ra* ở quy mô sinh viên, và đo đạc từng tầng. Phần "hơn" nằm ở bảng ablation §5: những con số mà việc dùng model có sẵn không bao giờ trả lời được. |
| "Sao không dùng luôn Sailor2?" | Dùng model sẵn thì không trả lời được câu hỏi nghiên cứu nào cả — không có tầng nào để đo. Sailor2 vẫn có mặt trong đồ án, đúng vai của nó: mốc đối chiếu ở phần eval. |
| "Model em thua ChatGPT thì làm để làm gì?" | Đúng, và mục tiêu chưa bao giờ là thắng — mục tiêu là kiểm soát từng tầng của quy trình. ChatGPT không cho xem đường loss của reward model. |
| "Sao chọn 1.7B nhỏ thế?" | Câu hỏi của đồ án là câu hỏi *phương pháp*, không phải *quy mô* — mọi kết luận ablation đo được ở 1.7B. Scale lên là vấn đề tiền, không phải vấn đề khoa học. |
| "CPT của em có ích thật không hay chỉ làm theo trend?" | Không khẳng định chay — E1 được thiết kế để trả lời chính câu này bằng số. Nếu delta nhỏ, đó cũng là một phát hiện trung thực và được ghi nhận trong báo cáo. |
| "Chạy trên Kaggle free thì nghiêm túc được không?" | Ràng buộc compute là một phần đề bài: mọi bước train đều checkpoint-resume qua Hub, tái lập được 100% từ notebook công khai. Khả tái lập là thứ nhiều công bố lớn còn thiếu. |

## 8. Reproducibility statement

Toàn bộ notebook tại `notebooks/` chạy trực tiếp trên Kaggle (T4 free tier); dataset và checkpoint trung gian công khai/private trên HuggingFace Hub theo quy ước đặt tên `<user>/Qwen3-1.7B-vi-{cpt,sft,rm,dpo}`; mọi seed cố định (42); log huấn luyện đầy đủ nằm trong `trainer_state.json` của từng checkpoint trên Hub. Một người khác với tài khoản Kaggle + HF token có thể tái tạo từng bước bằng Run All.
