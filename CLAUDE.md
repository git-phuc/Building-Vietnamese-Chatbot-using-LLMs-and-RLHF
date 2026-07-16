# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository nature

This is a **planning/design repository, not a codebase.** It contains a single Vietnamese-language document, `Building Vietnamese Chatbot using LLMs and RLHF.md`, which is the canonical project spec. There is no source code, build system, package manifest, or test suite — do not invent `build`/`lint`/`test` commands. Treat the markdown file as the source of truth the user is working from.

The project goal: fine-tune an LLM on Vietnamese conversation data (SFT), then improve responses with RLHF, then ship a chat interface. The document has two parts that must be kept distinct:
- **Section 2** = the *baseline* pipeline taken verbatim from the AI VIETNAM course slides (11 steps, "must do"). When the user asks to "follow the slides" / "do the baseline," implement exactly these.
- **Section 3** = the author's own *proposed improvements* (CPT, DPO-vs-PPO, quantitative benchmarks, iterative RLHF, safety alignment, tokenizer efficiency). These are optional ablations, NOT part of the required baseline. Don't fold them into baseline work unless the user asks.

## Pipeline map (Section 2)

Four blocks. When asked to implement a step, locate it here before writing code — filenames/object names in the doc are the authoritative identifiers.

1. **Setup + SFT (Steps 1–4)** — Unsloth framework, `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` base, QLoRA (`r=16`, 7 target modules, `use_rslora=True`), `MAX_SEQ_LENGTH=2048`, 4-bit + bf16. Dataset `5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca`. `SFTTrainer`, bs=32, grad-accum=2, lr=1e-4, `max_steps=400`, `paged_adamw_8bit`. Output merged SFT model → Hub (e.g. `thuanan/Llama-3.2-1B-Instruct-Chat-sft`).
2. **Reward Model (Steps 5–6)** — Preference data `thuanan/Vi-Alpaca-Preference` (chosen/rejected format). Trained with **OpenRLHF** `openrlhf.cli.train_rm`, value head prefix `score`, ZeRO stage 2, LoRA rank 16. Merge LoRA via `openrlhf.cli.lora_combiner` → `thuanan/Llama-3.2-1B-RM-DPO`.
3. **RLHF/PPO (Steps 7–8)** — **OpenRLHF** `openrlhf.cli.train_ppo_ray` over a **Ray cluster** (`ray start --head --num-gpus 3` → `ray job submit`). 4 model groups (Actor w/ vLLM inference, Reference, Reward, Critic), ZeRO stage 3. Prompt data `thuanan/Prompt-Vi-Alpaca-Preference-2k`, `input_key=context_messages`, `apply_chat_template`. KL penalty vs reference model is the design constraint to preserve. Final → `thuanan/Llama-3.2-1B-RLHF-2k-vi-alpaca`.
4. **Chat Interface (Steps 9–11)** — Convert HF model → GGUF (`llama.cpp/convert_hf_to_gguf.py --outtype bf16`), upload via `HfApi().upload_file`. Deploy with **Ollama + Open WebUI** via Docker Compose (Ollama port 11434, Open-WebUI 3001→8080, GPU reservation). Run with `ollama run hf.co/<...>-gguf`. The doc's Step 11 / Open WebUI rating-export-to-JSON mechanism is the intended source of preference data for **iterative RLHF** (Section 3.4).

A lightweight alternative the doc mentions: vLLM-serve the HF model directly + Gradio ChatInterface (skip GGUF), for fast demo only.

## Key entities (use these exact identifiers)

- **Models on Hub:** `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`, `thuanan/Llama-3.2-1B-Instruct-Chat-sft`, `thuanan/Llama-3.2-1B-RM-DPO`, `thuanan/Llama-3.2-1B-RLHF-2k-vi-alpaca`.
- **Datasets on Hub:** `5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca` (SFT), `thuanan/Vi-Alpaca-Preference` (RM), `thuanan/Prompt-Vi-Alpaca-Preference-2k` (PPO prompts). Section 4 lists *additional* public datasets not used by the baseline (`bkai-foundation-models/vi-alpaca`, `Bactrian-X`, `OpenAssistant/oasst1`, `linhtran92/viet_bud500`, etc.) — these are a resource pool, not baseline requirements.
- **Frameworks:** Unsloth (SFT/QLoRA), OpenRLHF (RM + PPO; also offers `train_dpo` — relevant to Section 3.2), Ray (PPO orchestration), llama.cpp (GGUF conversion), Ollama + Open WebUI (deploy).
- **Fixed system prompt** for SFT: *"Bạn là một trợ lý AI thân thiện, hãy trả lời bằng tiếng Việt."*

## How to help on this repo

- The user's working language is **Vietnamese**; the spec is bilingual (Vietnamese prose, English/code blocks). Match the user's language in replies.
- When the user asks to "implement step X," produce runnable, self-contained Python/bash that follows the exact config in Section 2 — same hyperparameters, dataset keys, Hub names — rather than generic defaults.
- Prefer the ablations in Section 3 **only** when the user names them or asks "what could be improved." If running an ablation, the doc already specifies each one's insertion point, cost, and what to measure (e.g. 3.3's VMLU + LLM-judge win-rate on a held-out set split from RM/PPO training data to avoid optimistic eval).
- If the user's request would conflate a Section 3 improvement with the Section 2 baseline (or vice versa), surface the distinction before proceeding.
- Hardware note from the doc: PPO assumes a 3-GPU Ray cluster; SFT/RM are single-GPU. Flag this when recommending a step, since the user's environment may differ.

When code is eventually added to this repo, revisit this file to document build/run/test commands for what was actually created.
