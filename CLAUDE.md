# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository nature

This is a **planning/design repository plus Kaggle notebooks, not an app codebase.** `docs/spec.md` is the canonical project spec; `README.md` is the **status board** — a step-by-step progress table (⬜/🟡/✅) that must be updated whenever a pipeline step starts or finishes; `notebooks/` holds the runnable Kaggle notebooks (one pair/step). There is no build system, package manifest, or test suite — do not invent `build`/`lint`/`test` commands. To find the current step or what's next, read README.md first, then the matching §6.x in the spec. (Note: the spec has no Section 5 — numbering intentionally jumps 4 → 6; "Section 6" is the canonical name used everywhere.)

The project goal: fine-tune an LLM on Vietnamese conversation data (SFT), then improve responses with RLHF, then ship a chat interface. The document now contains **two distinct pipelines** — do not conflate them:
- **Section 2** = the *baseline* pipeline taken verbatim from the AI VIETNAM course slides (11 steps, uses the ready-made `Llama-3.2-1B-Instruct`). Reference/comparison material only — this is NOT the pipeline the user is actually executing for the project.
- **Section 3** = the author's own *proposed improvements* over that baseline (CPT, DPO-vs-PPO, quantitative benchmarks, iterative RLHF, safety alignment, tokenizer efficiency). Superseded by Section 6, which turns these ideas into concrete executable steps.
- **Section 6** = **the pipeline actually being executed for this project.** Build the full RLHF pipeline from a raw (non-instruct) base model, `Qwen/Qwen3-1.7B-Base`, to learn every stage hands-on rather than starting from an already-tuned model. Sailor2 is a reference benchmark to compare against in Section 6.5 eval, never a starting point. **When the user says "tiếp tục đồ án" / "làm theo pipeline chính" / asks to implement a step without naming Section 2, default to Section 6, not Section 2.**

## Pipeline map (Section 2 — reference only, not active work)

Slide baseline on `Llama-3.2-1B-Instruct`: SFT (QLoRA) → Reward Model (OpenRLHF) → PPO (Ray cluster) → GGUF/Ollama deploy. Don't pull config from here unless the user explicitly asks to follow the slide baseline — full detail is in the doc's Section 2, kept intentionally short here to avoid Llama-specific noise when working on the actual Qwen3 pipeline below.

## Pipeline map (Section 6 — the pipeline actually being built)

Seven steps, all from a raw base model. Insertion points and Hub names are authoritative — use these exact identifiers.

1. **CPT (6.1)** — Continual pretraining from `Qwen/Qwen3-1.7B-Base` raw, organized as **two Kaggle notebooks** (fully specified in §6.1): Notebook A (run once) packs `wikimedia/wikipedia 20231101.vi` + `HuggingFaceFW/fineweb-2 vie_Latn` + ~20% `HuggingFaceFW/fineweb-edu` English replay into 2048-token blocks (never CC100/OSCAR — script-based/gated datasets fail on `datasets` >= 3.0) → `<user>/vi-cpt-corpus-2048` (private). Notebook B (run each session) trains QLoRA r=64 on 4-bit base (full-param does NOT fit a T4 — Adam fp32 states alone exceed 16GB), lr=1.5e-5, `max_steps=6000` (~400M tokens, no epochs), auto-resumes from `<user>/Qwen3-1.7B-vi-cpt-ckpt` (whole checkpoint folder uploaded via `HfApi.upload_folder` on each save — `push_to_hub=True` alone only pushes the adapter and breaks resume), stops itself before the ~8h session budget. Final merged output → `<user>/Qwen3-1.7B-vi-cpt` (separate repo from the ckpt repo).
2. **SFT (6.2)** — QLoRA (same shape as Section 2: r=16, `use_rslora=True`) starting from the CPT checkpoint, not raw Qwen3. Same dataset (`5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca`) and system prompt as Section 2. Output → `<user>/Qwen3-1.7B-vi-sft`.
3. **Reward Model (6.3)** — OpenRLHF `train_rm`, same script shape as Section 2 Step 6, `--pretrain` pointed at the SFT checkpoint above. Flag DeepSpeed install risk on Kaggle; Modal is the fallback. Output → `<user>/Qwen3-1.7B-vi-rm`.
4. **DPO (6.4a) — the main path**, not PPO. Chosen because it only needs policy + reference (no Ray/multi-GPU), so it fits a single Kaggle T4. Recommended implementation: TRL `DPOTrainer` + Unsloth QLoRA adapter reused from Step 2 (lighter than OpenRLHF's DeepSpeed dependency). Output → `<user>/Qwen3-1.7B-vi-dpo`.
4b. **PPO (6.4b) — ablation only**, not required. Reuses Section 2's Ray/OpenRLHF PPO script verbatim with updated `--pretrain`/`--reward_pretrain`. **Only runs on Modal/multi-GPU — never attempt on Kaggle** (needs 4 concurrent model groups). Output → `<user>/Qwen3-1.7B-vi-ppo`.
5. **Evaluation (6.5)** — VMLU, LLM-judge win-rate, and RM reward-score, all on a held-out split never used for RM/DPO/PPO training. Sailor2 is an optional reference-benchmark comparison here, not a target to match exactly.
6. **GGUF + Deploy (6.6)** — Same llama.cpp → Ollama + Open WebUI flow as Section 2 Steps 9-10, just pointed at the DPO checkpoint.
7. **Feedback loop (6.7)** — Optional stretch goal, not a required deliverable: Open WebUI rating export → preference data v2 → RM v2 → DPO v2 (actor starts from DPO v1, not from SFT).

## Key entities (use these exact identifiers)

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
- Design/explanation content (what a step does, why, config rationale) goes into the spec doc's `.md`. Runnable training code additionally gets scaffolded as real `.ipynb` notebooks (one per pipeline step, e.g. `cpt.ipynb`, `sft.ipynb`) meant to be run directly on Kaggle — keep the `.md` and the notebook in sync when either changes.
- Canonical notebooks live in `notebooks/`, organized one folder per pipeline step (`01-cpt/`, `02-sft/`, `03-rm/`, `04-dpo/`, `05-eval/`, `06-deploy/`), each with a step README (goal, notebooks, input/output). Notebook naming pattern: `<STEP>-...-Qwen3-1.7B.ipynb` (e.g. `CPT-Step-A-Prepare-`, `CPT-Step-B-Train-`, `SFT-Train-`) — Kaggle notebook titles match these filenames. If a worktree has an older per-step notebook that diverges from `notebooks/`, the `notebooks/` version wins. Product-phase code will live in `app/` per the multi-agent plan in `docs/orchestrator.md` (not created until step 6 starts).
- If the user's request would conflate Section 2/3 (reference) with Section 6 (actual pipeline), surface the distinction before proceeding.
- Hardware note: Section 2's PPO assumes a 3-GPU Ray cluster; Section 6's PPO ablation likewise needs Modal/multi-GPU and must never be pointed at Kaggle. Flag this when recommending a step.

When code is eventually added to this repo, revisit this file to document build/run/test commands for what was actually created.
