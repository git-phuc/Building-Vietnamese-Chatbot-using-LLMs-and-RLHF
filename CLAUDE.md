# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository nature

This is a **planning/design repository, not a codebase.** It contains a single Vietnamese-language document, `Building Vietnamese Chatbot using LLMs and RLHF.md`, which is the canonical project spec. There is no source code, build system, package manifest, or test suite — do not invent `build`/`lint`/`test` commands. Treat the markdown file as the source of truth the user is working from.

The project goal: fine-tune an LLM on Vietnamese conversation data (SFT), then improve responses with RLHF, then ship a chat interface. The document now contains **two distinct pipelines** — do not conflate them:
- **Section 2** = the *baseline* pipeline taken verbatim from the AI VIETNAM course slides (11 steps, uses the ready-made `Llama-3.2-1B-Instruct`). Reference/comparison material only — this is NOT the pipeline the user is actually executing for the project.
- **Section 3** = the author's own *proposed improvements* over that baseline (CPT, DPO-vs-PPO, quantitative benchmarks, iterative RLHF, safety alignment, tokenizer efficiency). Superseded by Section 6, which turns these ideas into concrete executable steps.
- **Section 6** = **the pipeline actually being executed for this project.** Build the full RLHF pipeline from a raw (non-instruct) base model, `Qwen/Qwen3-1.7B-Base`, to learn every stage hands-on rather than starting from an already-tuned model. Sailor2 is a reference benchmark to compare against in Section 6.5 eval, never a starting point. **When the user says "tiếp tục đồ án" / "làm theo pipeline chính" / asks to implement a step without naming Section 2, default to Section 6, not Section 2.**

## Pipeline map (Section 2)

Four blocks. When asked to implement a step, locate it here before writing code — filenames/object names in the doc are the authoritative identifiers.

1. **Setup + SFT (Steps 1–4)** — Unsloth framework, `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` base, QLoRA (`r=16`, 7 target modules, `use_rslora=True`), `MAX_SEQ_LENGTH=2048`, 4-bit + bf16. Dataset `5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca`. `SFTTrainer`, bs=32, grad-accum=2, lr=1e-4, `max_steps=400`, `paged_adamw_8bit`. Output merged SFT model → Hub (e.g. `thuanan/Llama-3.2-1B-Instruct-Chat-sft`).
2. **Reward Model (Steps 5–6)** — Preference data `thuanan/Vi-Alpaca-Preference` (chosen/rejected format). Trained with **OpenRLHF** `openrlhf.cli.train_rm`, value head prefix `score`, ZeRO stage 2, LoRA rank 16. Merge LoRA via `openrlhf.cli.lora_combiner` → `thuanan/Llama-3.2-1B-RM-DPO`.
3. **RLHF/PPO (Steps 7–8)** — **OpenRLHF** `openrlhf.cli.train_ppo_ray` over a **Ray cluster** (`ray start --head --num-gpus 3` → `ray job submit`). 4 model groups (Actor w/ vLLM inference, Reference, Reward, Critic), ZeRO stage 3. Prompt data `thuanan/Prompt-Vi-Alpaca-Preference-2k`, `input_key=context_messages`, `apply_chat_template`. KL penalty vs reference model is the design constraint to preserve. Final → `thuanan/Llama-3.2-1B-RLHF-2k-vi-alpaca`.
4. **Chat Interface (Steps 9–11)** — Convert HF model → GGUF (`llama.cpp/convert_hf_to_gguf.py --outtype bf16`), upload via `HfApi().upload_file`. Deploy with **Ollama + Open WebUI** via Docker Compose (Ollama port 11434, Open-WebUI 3001→8080, GPU reservation). Run with `ollama run hf.co/<...>-gguf`. The doc's Step 11 / Open WebUI rating-export-to-JSON mechanism is the intended source of preference data for **iterative RLHF** (Section 3.4).

A lightweight alternative the doc mentions: vLLM-serve the HF model directly + Gradio ChatInterface (skip GGUF), for fast demo only.

## Pipeline map (Section 6 — the pipeline actually being built)

Seven steps, all from a raw base model. Insertion points and Hub names are authoritative — use these exact identifiers.

1. **CPT (6.1)** — Continual pretraining on Vietnamese corpus (Wikipedia_vi, News, OSCAR-vi/CC100-vi filtered) starting from `Qwen/Qwen3-1.7B-Base` raw. Keep 10-30% English replay data to avoid catastrophic forgetting. LR ~5-10x lower than SFT, 1-2 epochs. At 1.7B, full-parameter CPT does NOT fit a 16GB T4 (Adam fp32 states alone exceed it) — use QLoRA r=64 on a 4-bit base instead (larger rank than SFT's r=16 since CPT updates more). Output → `<user>/Qwen3-1.7B-vi-cpt`.
2. **SFT (6.2)** — QLoRA (same shape as Section 2: r=16, `use_rslora=True`) starting from the CPT checkpoint, not raw Qwen3. Same dataset (`5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca`) and system prompt as Section 2. Output → `<user>/Qwen3-1.7B-vi-sft`.
3. **Reward Model (6.3)** — OpenRLHF `train_rm`, same script shape as Section 2 Step 6, `--pretrain` pointed at the SFT checkpoint above. Flag DeepSpeed install risk on Kaggle; Modal is the fallback. Output → `<user>/Qwen3-1.7B-vi-rm`.
4. **DPO (6.4a) — the main path**, not PPO. Chosen because it only needs policy + reference (no Ray/multi-GPU), so it fits a single Kaggle T4. Recommended implementation: TRL `DPOTrainer` + Unsloth QLoRA adapter reused from Step 2 (lighter than OpenRLHF's DeepSpeed dependency). Output → `<user>/Qwen3-1.7B-vi-dpo`.
4b. **PPO (6.4b) — ablation only**, not required. Reuses Section 2's Ray/OpenRLHF PPO script verbatim with updated `--pretrain`/`--reward_pretrain`. **Only runs on Modal/multi-GPU — never attempt on Kaggle** (needs 4 concurrent model groups). Output → `<user>/Qwen3-1.7B-vi-ppo`.
5. **Evaluation (6.5)** — VMLU, LLM-judge win-rate, and RM reward-score, all on a held-out split never used for RM/DPO/PPO training. Sailor2 is an optional reference-benchmark comparison here, not a target to match exactly.
6. **GGUF + Deploy (6.6)** — Same llama.cpp → Ollama + Open WebUI flow as Section 2 Steps 9-10, just pointed at the DPO checkpoint.
7. **Feedback loop (6.7)** — Optional stretch goal, not a required deliverable: Open WebUI rating export → preference data v2 → RM v2 → DPO v2 (actor starts from DPO v1, not from SFT).

## Key entities (use these exact identifiers)

- **Section 2 (baseline reference) models on Hub:** `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`, `thuanan/Llama-3.2-1B-Instruct-Chat-sft`, `thuanan/Llama-3.2-1B-RM-DPO`, `thuanan/Llama-3.2-1B-RLHF-2k-vi-alpaca`.
- **Section 6 (actual pipeline) checkpoint naming convention:** `<user>/Qwen3-1.7B-vi-{cpt,sft,rm,dpo,ppo}` — base model is `Qwen/Qwen3-1.7B-Base` (raw, non-instruct).
- **Datasets on Hub:** `5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca` (SFT, used by both pipelines), `thuanan/Vi-Alpaca-Preference` (RM/DPO), `thuanan/Prompt-Vi-Alpaca-Preference-2k` (PPO prompts). Section 4 lists *additional* public datasets not used by either pipeline by default (`bkai-foundation-models/vi-alpaca`, `Bactrian-X`, `OpenAssistant/oasst1`, `linhtran92/viet_bud500`, etc.) — these are a resource pool.
- **Frameworks:** Unsloth (SFT/QLoRA), OpenRLHF (RM + PPO; also offers `train_dpo`), TRL `DPOTrainer` (preferred for Section 6's DPO step), Ray (PPO orchestration), llama.cpp (GGUF conversion), Ollama + Open WebUI (deploy).
- **Fixed system prompt** for SFT (both pipelines): *"Bạn là một trợ lý AI thân thiện, hãy trả lời bằng tiếng Việt."*

## Compute constraints (Section 6)

- **Kaggle (T4 16GB, 30h/week quota, ~9h max per session) is primary.** No training step is guaranteed to finish in one session — every training step MUST support checkpoint/resume: `save_strategy="steps"` with small `save_steps`, push to a private HF Hub repo (or Kaggle Dataset output) before the session ends, `resume_from_checkpoint=True` on the next session.
- **Modal is the fallback**, required (not optional) for PPO ablation (Step 4b) since it needs multi-GPU that Kaggle's single T4 can't provide. Also the fallback if OpenRLHF/DeepSpeed fails to install on Kaggle for the Reward Model or DPO steps.
- When recommending or scaffolding any Section 6 training step, always include the checkpoint/resume mechanism — don't produce a plain single-session training script.

## How to help on this repo

- The user's working language is **Vietnamese**; the spec is bilingual (Vietnamese prose, English/code blocks). Match the user's language in replies.
- Default to **Section 6** for actual project work. Only use Section 2's exact config when the user explicitly asks to "follow the slides" / "do the baseline" for reference/comparison purposes.
- When the user asks to "implement step X" without specifying which pipeline, assume Section 6 and produce runnable, self-contained Python/bash following Section 6's config (checkpoint/resume included) — not generic defaults.
- Content goes **into the project's `.md` files** (the spec doc and this file), not scaffolded as standalone `.ipynb`/`src/` code files — the user deliberately chose markdown-as-plan over a code scaffold.
- If the user's request would conflate Section 2/3 (reference) with Section 6 (actual pipeline), surface the distinction before proceeding.
- Hardware note: Section 2's PPO assumes a 3-GPU Ray cluster; Section 6's PPO ablation likewise needs Modal/multi-GPU and must never be pointed at Kaggle. Flag this when recommending a step.

When code is eventually added to this repo, revisit this file to document build/run/test commands for what was actually created.
