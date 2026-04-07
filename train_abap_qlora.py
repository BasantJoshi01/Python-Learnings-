"""
=============================================================================
  Fine-Tuning Pipeline: Qwen 2.5 Coder 7B → Domain-Specific ABAP Model
  Method  : QLoRA (4-bit quantization + LoRA adapters)
  Target  : AWS EC2 g5.4xlarge  |  NVIDIA A10G 24 GB VRAM
  Author  : Production-ready template
=============================================================================

QUICK-START
-----------
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.2 peft==0.12.0 trl==0.10.1 \
            bitsandbytes==0.43.3 datasets accelerate sentencepiece \
            scipy einops

Run:
    python train_abap_qlora.py \
        --train_file data/abap_train.jsonl \
        --val_file   data/abap_val.jsonl  \
        --output_dir ./abap-qlora-out
=============================================================================
"""

import os
import re
import sys
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1.  CLI ARGUMENTS
# ===========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tune for ABAP code gen")
    p.add_argument("--base_model",  default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--train_file",  required=True,  help="Path to training JSONL")
    p.add_argument("--val_file",    required=False, default=None)
    p.add_argument("--output_dir",  default="./abap-qlora-out")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--epochs",      type=int, default=3)
    p.add_argument("--batch_size",  type=int, default=2,
                   help="Per-device train batch size (keep ≤4 on A10G)")
    p.add_argument("--grad_accum",  type=int, default=8,
                   help="Gradient accumulation steps (effective BS = batch*accum)")
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--lora_r",      type=int, default=64)
    p.add_argument("--lora_alpha",  type=int, default=128)
    p.add_argument("--lora_dropout",type=float, default=0.05)
    p.add_argument("--quant_bits",  type=int, default=4, choices=[4, 8])
    p.add_argument("--resume_from_checkpoint", default=None)
    return p.parse_args()


# ===========================================================================
# 2.  LEGACY ABAP SYNTAX VALIDATOR
#     Rejects samples whose OUTPUT contains forbidden legacy patterns.
#     Called at dataset-load time to keep training data clean.
# ===========================================================================
# Patterns that must NEVER appear in a modern ABAP output
LEGACY_PATTERNS = [
    r"\bOCCURS\b",                   # Old internal table declaration
    r"SELECT\s+\*",                  # SELECT * (always forbidden)
    r"\bTABLES\b\s*:",               # TABLES: keyword in function modules
    r"\bHEADER\s+LINE\b",           # WITH HEADER LINE
    r"\bCALL\s+FUNCTION\b",         # CALL FUNCTION (prefer method calls)
    r"\bFORM\b\s+\w+",              # FORM subroutines
    r"\bPERFORM\b\s+\w+",           # PERFORM calls
    r"TYPE\s+TABLE\s+OF.*OCCURS",   # OCCURS inside TYPE TABLE
    r"\bMESSAGE\s+[EWI]\d+",        # Old MESSAGE statement style
    r"^\s*DATA\s*:\s*BEGIN\s+OF",   # Flat structure via old syntax
]
_LEGACY_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE)
              for p in LEGACY_PATTERNS]


def contains_legacy_syntax(code: str) -> tuple[bool, str]:
    """
    Returns (True, matched_pattern) if legacy ABAP syntax is found.
    Returns (False, "") if the code is clean.
    """
    for rx in _LEGACY_RE:
        m = rx.search(code)
        if m:
            return True, m.group(0).strip()
    return False, ""


def _extract_assistant_output(sample: dict) -> str:
    """
    Pull the assistant's reply out of the messages list.

    Your JSONL format:
        { "messages": [
            { "role": "user",      "content": "You are an expert... Task: ..." },
            { "role": "assistant", "content": "<ABAP code>" }
          ]
        }
    Returns empty string if the structure is unexpected.
    """
    messages = sample.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def validate_sample(sample: dict) -> bool:
    """
    Validate a single JSONL sample in your native messages[] format.

    Rules
    -----
    • Structural check : must have exactly one user turn and one assistant turn.
    • Legacy check     : assistant output must not contain forbidden patterns.
    • Label override   : if a top-level "label": "invalid" key exists, the
                         sample is a deliberate negative example — keep it.
    """
    messages = sample.get("messages", [])

    # ── Structural guard ──────────────────────────────────────────────────
    roles = [m.get("role") for m in messages]
    if "user" not in roles or "assistant" not in roles:
        log.warning("⚠  Dropping malformed sample – missing user or assistant turn.")
        return False

    # ── Honour explicit negative-example label ────────────────────────────
    if sample.get("label", "valid") == "invalid":
        return True      # Kept intentionally for refusal training

    # ── Legacy-syntax check on the assistant output ───────────────────────
    output = _extract_assistant_output(sample)
    if not output.strip():
        log.warning("⚠  Dropping sample – empty assistant content.")
        return False

    dirty, pattern = contains_legacy_syntax(output)
    if dirty:
        log.warning(
            "⚠  Dropping sample – legacy pattern '%s' in assistant output: %s…",
            pattern, output[:80],
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Canonical system prompt injected into every training sample.
# Your dataset embeds this text inside the user turn — we extract and promote
# it to a proper "system" role so Qwen's chat template uses it correctly.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert SAP ABAP 7.4+ code generator. "
    "Write only modern ABAP 7.5 syntax. "
    "Never use OCCURS, SELECT *, TABLES:, HEADER LINE, CALL FUNCTION, "
    "FORM/PERFORM subroutines, or any other legacy constructs. "
    "Return only ABAP code."
)

# The inline system prefix your dataset places at the top of every user turn.
# We strip this and replace it with the canonical system role above.
_INLINE_SYSTEM_PREFIX = re.compile(
    r"^You are an expert SAP ABAP[^\n]*\.\s*Return only ABAP code\.\s*\n*",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_messages(raw_messages: list[dict]) -> list[dict]:
    """
    Convert your dataset's message structure into a clean 3-turn list:
        [ system, user, assistant ]

    Your format embeds the system prompt inside the user turn:
        user: "You are an expert SAP ABAP 7.4+ code generator. Return only ABAP code.
               Task: Implement a reusable ABAP 7.5 snippet that..."
        assistant: "<ABAP code>"

    After normalisation:
        system:    "You are an expert SAP ABAP 7.4+ code generator..."
        user:      "Task: Implement a reusable ABAP 7.5 snippet that..."
        assistant: "<ABAP code>"
    """
    normalized = []
    system_injected = False

    for msg in raw_messages:
        role    = msg.get("role", "")
        content = msg.get("content", "").strip()

        if role == "user":
            # Strip the inline system prefix from the user turn
            clean_content = _INLINE_SYSTEM_PREFIX.sub("", content).strip()

            # Inject canonical system turn exactly once, before the first user turn
            if not system_injected:
                normalized.append({"role": "system", "content": SYSTEM_PROMPT})
                system_injected = True

            normalized.append({"role": "user", "content": clean_content})

        elif role == "assistant":
            normalized.append({"role": "assistant", "content": content})

        # Skip any existing "system" turns — we supply our own canonical one

    return normalized


def format_sample(sample: dict, tokenizer) -> str:
    """
    Convert one JSONL record (your native messages[] format) into the
    full Qwen chat-template string used as training text.

    Input JSONL shape
    -----------------
    {
      "messages": [
        {
          "role": "user",
          "content": "You are an expert SAP ABAP 7.4+ code generator.
                      Return only ABAP code.\\n\\nTask: <task description>"
        },
        {
          "role": "assistant",
          "content": "<modern ABAP 7.5 code>"
        }
      ]
    }

    The tokenizer's apply_chat_template() adds all BOS/EOS and turn-delimiter
    special tokens automatically — no manual string concatenation needed.
    """
    raw_messages = sample.get("messages", [])
    messages     = _normalize_messages(raw_messages)

    # add_generation_prompt=False → assistant turn IS included in the loss
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text


def load_abap_dataset(
    train_file: str,
    val_file: Optional[str],
    tokenizer,
    max_seq_len: int,
) -> tuple[Dataset, Optional[Dataset]]:
    """
    Load, validate, and format JSONL files in your native messages[] format.

    Each line in the JSONL must look like:
        {
          "messages": [
            { "role": "user",      "content": "You are an expert SAP ABAP 7.4+...\\nTask: ..." },
            { "role": "assistant", "content": "REPORT zrep_abap75_xxxxx.\\n..." }
          ]
        }

    Steps applied per file
    ----------------------
    1. HuggingFace load_dataset  — reads every JSONL line into a Dataset row.
                                   Each row has one field: "messages" (a list).
    2. validate_sample()         — drops structurally broken rows and rows
                                   where the assistant output contains legacy syntax.
    3. format_sample()           — calls _normalize_messages() to promote the
                                   inline system prefix to a proper system role,
                                   then applies the Qwen chat template.
    4. token-length filter       — discards formatted samples longer than
                                   max_seq_len to avoid silent truncation during
                                   packing (which corrupts gradients).
    5. Token-length stats        — logs min/mean/max so you can tune max_seq_len.
    """

    def _load_and_clean(path: str) -> Dataset:
        # ── 1. Load raw JSONL ────────────────────────────────────────────
        raw = load_dataset("json", data_files=path, split="train")
        log.info("Loaded %d raw samples from %s", len(raw), path)

        # ── 2. Validate – drop legacy-tainted or malformed samples ───────
        before = len(raw)
        raw = raw.filter(validate_sample)
        dropped = before - len(raw)
        log.info(
            "%d samples remain after validation (%d dropped)",
            len(raw), dropped,
        )

        # ── 3. Format: normalise messages → apply Qwen chat template ─────
        raw = raw.map(
            lambda s: {"text": format_sample(s, tokenizer)},
            remove_columns=raw.column_names,   # keep only the new "text" column
            desc="Applying chat template",
        )

        # ── 4. Token-length filter ────────────────────────────────────────
        #    Tokenise without truncation to get real lengths, then filter.
        def _add_length(sample: dict) -> dict:
            ids = tokenizer(sample["text"], truncation=False)["input_ids"]
            sample["n_tokens"] = len(ids)
            return sample

        raw = raw.map(_add_length, desc="Counting tokens")

        # ── 5. Log token-length distribution before cutting ───────────────
        lengths = raw["n_tokens"]
        log.info(
            "Token lengths — min: %d | mean: %.0f | max: %d",
            min(lengths), sum(lengths) / len(lengths), max(lengths),
        )

        raw = raw.filter(lambda s: s["n_tokens"] <= max_seq_len)
        log.info(
            "%d samples fit within %d token limit",
            len(raw), max_seq_len,
        )

        # Drop the helper column before returning
        raw = raw.remove_columns(["n_tokens"])
        return raw

    train_ds = _load_and_clean(train_file)
    val_ds   = _load_and_clean(val_file) if val_file else None
    return train_ds, val_ds


# ===========================================================================
# 4.  QUANTIZATION CONFIG  (QLoRA = 4-bit NF4 + double quant)
# ===========================================================================
def build_bnb_config(bits: int) -> BitsAndBytesConfig:
    """
    4-bit  → NF4 + double quantization  (~4.5 GB for 7B)
    8-bit  → LLM.int8()                 (~8 GB for 7B)
    Both fit comfortably on A10G 24 GB.
    """
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NF4 best for LLM weights
            bnb_4bit_use_double_quant=True,      # Saves ~0.4 bits/param extra
            bnb_4bit_compute_dtype=torch.bfloat16,  # BF16 stable on Ampere
        )
    else:
        return BitsAndBytesConfig(load_in_8bit=True)


# ===========================================================================
# 5.  MODEL + TOKENIZER LOADER
# ===========================================================================
def load_model_and_tokenizer(
    model_name: str,
    bnb_config: BitsAndBytesConfig,
):
    log.info("Loading tokenizer from %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",   # Right-padding for causal LM training
    )
    # Qwen tokenizer already has a pad token; add one if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading model with %d-bit quantization …", 4 if bnb_config.load_in_4bit else 8)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",          # Automatically place on GPU
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # Flash-Attn 2 if installed
    )

    # Enable gradient checkpointing to cut activation memory ~50 %
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    return model, tokenizer


# ===========================================================================
# 6.  LoRA CONFIG
#     Target the attention + MLP projection layers (standard for Qwen)
# ===========================================================================
def build_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    """
    r=64, alpha=128 → relatively large adapter; good for domain adaptation.
    Reduce r to 16-32 if VRAM is tight or dataset is <5 k samples.
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",                 # Don't adapt bias – saves memory
        target_modules=[             # Qwen 2.5 attention + MLP layers
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        inference_mode=False,
    )


# ===========================================================================
# 7.  TRAINING CONFIGURATION
# ===========================================================================
def build_training_args(
    output_dir: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    has_val: bool,
) -> SFTConfig:
    """
    Key choices for A10G 24 GB VRAM:
    - fp16=False, bf16=True   → BF16 is more stable on Ampere GPUs
    - gradient_checkpointing  → already enabled on model; flag here too
    - optim=paged_adamw_8bit  → bitsandbytes paged optimizer reduces peak mem
    - warmup_ratio=0.03       → ~standard for instruction fine-tuning
    """
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,

        # Batch & accumulation
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,

        # Precision
        bf16=True,
        fp16=False,

        # Optimizer
        optim="paged_adamw_8bit",
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Sequence packing (TRL packs short samples → fewer padding tokens)
        packing=True,
        max_seq_length=2048,          # Must match dataset filter above

        # Logging & saving
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if has_val else "no",
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss" if has_val else None,
        save_total_limit=2,
        report_to="none",             # Set to "wandb" if you use W&B

        # Reproducibility
        seed=42,
        data_seed=42,

        # Gradient checkpointing passthrough
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


# ===========================================================================
# 8.  BASIC EVALUATION HELPER
#     Runs a handful of held-out prompts and checks for legacy patterns.
# ===========================================================================
EVAL_PROMPTS = [
    "Write an ABAP class to read all entries from table MARA where MATNR starts with 'MAT'.",
    "Convert this legacy ABAP to modern syntax:\n"
    "SELECT * FROM mara INTO TABLE gt_mara WHERE mtart = 'FERT'.",
    "Use REDUCE to sum the NET_VALUE field of an internal table lt_items of type "
    "STANDARD TABLE OF ty_item.",
    "Write a method that uses VALUE #( ) to populate a local table of type "
    "tt_address with two rows.",
]


def run_evaluation(model, tokenizer, max_new_tokens: int = 300):
    """
    Qualitative spot-check: generate outputs for fixed prompts and
    flag any legacy syntax that sneaked through.
    """
    log.info("=" * 60)
    log.info("QUALITATIVE EVALUATION")
    log.info("=" * 60)
    model.eval()

    for idx, prompt_text in enumerate(EVAL_PROMPTS, 1):
        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": prompt_text},
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,         # Greedy for reproducibility
                temperature=1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated portion (skip the input tokens)
        generated = tokenizer.decode(
            out[0][encoded.shape[1]:],
            skip_special_tokens=True,
        )

        dirty, pattern = contains_legacy_syntax(generated)
        status = f"⚠  LEGACY FOUND: '{pattern}'" if dirty else "✅ Clean"

        log.info("\n--- Prompt %d ---\n%s", idx, prompt_text)
        log.info("--- Output ---\n%s", generated)
        log.info("--- Status: %s\n", status)

    model.train()


# ===========================================================================
# 9.  MAIN ENTRY POINT
# ===========================================================================
def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Step 1: Build quantization config ──────────────────────────────────
    bnb_config = build_bnb_config(args.quant_bits)

    # ── Step 2: Load model & tokenizer ─────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.base_model, bnb_config)

    # ── Step 3: Load & validate datasets ───────────────────────────────────
    train_ds, val_ds = load_abap_dataset(
        args.train_file,
        args.val_file,
        tokenizer,
        args.max_seq_len,
    )
    log.info("Train: %d  |  Val: %s", len(train_ds),
             len(val_ds) if val_ds else "none")

    # ── Step 4: Attach LoRA adapters ────────────────────────────────────────
    lora_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected output on r=64: ~84 M trainable / 7 B total (~1.2 %)

    # ── Step 5: Build training arguments ────────────────────────────────────
    training_args = build_training_args(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        has_val=val_ds is not None,
    )

    # ── Step 6: Build Trainer ───────────────────────────────────────────────
    callbacks = []
    if val_ds is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        callbacks=callbacks if callbacks else None,
        # dataset_text_field="text" is default for SFTTrainer
    )

    # ── Step 7: Train ───────────────────────────────────────────────────────
    log.info("Starting training …")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ── Step 8: Save final adapter weights ──────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final_adapter")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    log.info("LoRA adapter saved to %s", final_dir)

    # ── Step 9: Qualitative evaluation ──────────────────────────────────────
    run_evaluation(trainer.model, tokenizer)

    log.info("Done ✓")


if __name__ == "__main__":
    main()


# ===========================================================================
# APPENDIX: DATASET QUALITY TIPS  (read before training)
# ===========================================================================
"""
╔══════════════════════════════════════════════════════════════════════════╗
║              DATASET QUALITY TIPS FOR ABAP FINE-TUNING                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  YOUR JSONL RECORD FORMAT (native messages[])                            ║
║  {                                                                       ║
║    "messages": [                                                         ║
║      {                                                                   ║
║        "role": "user",                                                   ║
║        "content": "You are an expert SAP ABAP 7.4+ code generator.      ║
║                    Return only ABAP code.\n\nTask: <task description>"   ║
║      },                                                                  ║
║      {                                                                   ║
║        "role": "assistant",                                              ║
║        "content": "REPORT zrep_abap75_xxxxx.\n\n<modern ABAP code>"    ║
║      }                                                                   ║
║    ]                                                                     ║
║  }                                                                       ║
║                                                                          ║
║  OPTIONAL: add a top-level "label": "invalid" key for negative          ║
║  examples — the pipeline keeps them for refusal training.                ║
║                                                                          ║
║  TASK PREFIXES — vary these in your "Task:" line:                       ║
║  • "Implement a reusable ABAP 7.5 snippet that…"                        ║
║  • "Produce a maintainable ABAP 7.5 report that…"                      ║
║  • "Generate an optimized ABAP 7.5 program that…"                      ║
║  • "Create a modern ABAP 7.5 snippet that…"                            ║
║  • "Convert this legacy ABAP to modern syntax: <old code>"              ║
║  • "Fix this ABAP code and explain what was wrong: <buggy code>"        ║
║                                                                          ║
║  VOLUME                                                                  ║
║  • Minimum  : 1,000 samples for visible domain shift                    ║
║  • Sweet spot: 5,000–15,000 for solid code quality                      ║
║  • 20 % negative (label=invalid) examples recommended                   ║
║                                                                          ║
║  DIVERSITY  — make sure assistant outputs cover:                         ║
║  • SELECT…INTO @DATA / @lt_tab with explicit column lists               ║
║  • VALUE #( ), REDUCE, FILTER, COND, SWITCH, NEW, CONV, CAST            ║
║  • SORT + READ TABLE…BINARY SEARCH pattern                              ║
║  • FOR ALL ENTRIES IN guard (IS INITIAL check before FAE SELECT)        ║
║  • OOP: CLASS DEFINITION/IMPLEMENTATION, interfaces, method calls       ║
║  • Exception handling: TRY/CATCH/RAISE EXCEPTION NEW …                 ║
║  • ABAP Unit: FOR TESTING, cl_abap_unit_assert, mock objects            ║
║  • Dependency injection via constructor + interface                      ║
║  • COLLECT for aggregation                                               ║
║  • REPORT header + START-OF-SELECTION pattern                           ║
║  • Conversion tasks (legacy → modern) with full rewrites                ║
║                                                                          ║
║  QUALITY GATES before training                                           ║
║  1. Every assistant turn must start with valid ABAP (REPORT/CLASS/…)   ║
║  2. Run validate_sample() on every record via the pipeline              ║
║  3. Log token-length distribution — aim for P95 < 1800 tokens          ║
║  4. Manually review 50 random samples per task category                 ║
║  5. No duplicate "Task:" instructions (use MinHash dedup)               ║
║  6. Table names must stay generic (zt_*, not real SAP tables)           ║
║                                                                          ║
║  HYPER-PARAM TUNING HINTS                                                ║
║  • lr too high → loss spikes       → lower to 1e-4                     ║
║  • lr too low  → no convergence    → raise to 3e-4                     ║
║  • loss plateaus early             → increase lora_r to 128             ║
║  • OOM on A10G                     → batch_size=1, grad_accum=16        ║
║  • Overfitting (val >> train loss) → lora_dropout=0.1, fewer epochs    ║
║  • Short samples waste packing     → lower max_seq_len to 1024         ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
