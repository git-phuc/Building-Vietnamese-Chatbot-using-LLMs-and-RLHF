# Thiết kế orchestrator — giai đoạn product (bước 6–7)

> Tài liệu thiết kế, viết trước để đến lúc dùng là bật được ngay. **Chưa kích hoạt** — giai đoạn train (bước 1–5) là tuyến tính, một phiên chat + branch `main` là đủ (nút cổ chai là GPU, không phải tốc độ viết code). Kích hoạt khi bắt đầu bước 6 (deploy) — thời điểm xuất hiện nhiều workstream độc lập thật sự.

## 1. Mô hình

- **Orchestrator** = phiên Claude Code chính (phiên đang chat với user). Nhiệm vụ: chia việc, spawn agent, review kết quả, merge về `main`, cập nhật bảng trạng thái README.
- **Agent** = subagent chạy song song, context riêng, nhận một workstream trọn gói (đề bài + tiêu chí xong).
- **Worktree** = bản sao làm việc riêng cho mỗi agent (`.claude/worktrees/<tên>`, branch `worktree-<tên>` tách từ `main`) — các agent không giẫm file của nhau.

## 2. Workstream giai đoạn product

| Workstream | Nội dung | Phụ thuộc | Output |
|---|---|---|---|
| `gguf` | Script convert DPO checkpoint → GGUF (llama.cpp), quantize Q4_K_M/Q8_0 | Bước 4 xong | `app/gguf/` + repo Hub `-dpo-gguf` |
| `serving` | Ollama Modelfile (system prompt, template ChatML, stop token) + hướng dẫn chạy | `gguf` | `app/serving/` |
| `webui` | Cấu hình Open WebUI trỏ vào Ollama, bật rating để thu feedback | `serving` | `app/webui/` |
| `feedback` | Export rating WebUI → chuẩn hóa thành preference data v2 (chosen/rejected) cho §6.7 | `webui` chạy thật | `app/feedback/` + dataset Hub v2 |
| `eval-harness` | Đóng gói script eval §6.5 thành CLI chạy lại được (so v1 vs v2 sau này) | Bước 5 xong | `app/eval/` |
| `report` | Điền số liệu vào `report/main.tex`, vẽ hình loss/win-rate | Có số từ eval | `report/` |

Chuỗi phụ thuộc chính: `gguf → serving → webui → feedback`; `eval-harness` và `report` song song với chuỗi đó. Tức là tối đa ~3 agent chạy đồng thời là hợp lý — nhiều hơn không có việc.

## 3. Cấu trúc code product (tạo khi bắt đầu bước 6)

```
app/
├── gguf/       # convert + quantize scripts
├── serving/    # Ollama Modelfile + run scripts
├── webui/      # Open WebUI config (docker-compose, env)
├── feedback/   # export rating → preference v2
└── eval/       # eval CLI (VMLU, LLM-judge, RM score)
```

## 4. Luật vận hành (rút từ bài học của chính repo này)

1. **`main` là nguồn sự thật duy nhất.** Agent chỉ commit trong worktree của mình; không agent nào merge vào `main` — chỉ orchestrator merge, sau khi review.
2. **Một worktree = một workstream = một agent.** Không dùng lại worktree cho việc khác (tránh tình trạng folder tên `sft` mà ruột là bản repo cũ — đã từng gây nhầm lẫn, phải dọn ngày 2026-07-18).
3. **Merge xong xóa ngay** worktree + branch (`git worktree remove` + `git branch -d`). Worktree sống lâu = rác + nhầm lẫn.
4. **Đề bài cho agent phải trọn gói**: input ở đâu, output đặt đâu, tiêu chí xong là gì, và pointer tới §6.x liên quan — agent không thấy hội thoại chính nên không được viết đề bài kiểu "như đã bàn".
5. **README gốc vẫn là bảng trạng thái duy nhất** — orchestrator cập nhật sau mỗi lần merge, agent không tự sửa README gốc.
6. Không push lên origin trừ khi user yêu cầu (luật chung của repo).
