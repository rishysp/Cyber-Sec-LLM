"""Stage 2 - QLoRA fine-tuning of Llama 3.1 8B Instruct with Unsloth on an 8 GB GPU.
Run inside the 'ft' venv (Python 3.11):  python train_qlora.py
Outputs (in $SEC_ROOT/outputs/qlora):  adapter/, metrics.json, log_history.json
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # reduces fragmentation on small cards

import argparse, inspect, json, pathlib, time
from unsloth import FastLanguageModel                      # must be imported before transformers / trl
from unsloth.chat_templates import train_on_responses_only
import torch
from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

ROOT = pathlib.Path(os.environ.get("SEC_ROOT", pathlib.Path.home() / "secrag"))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "models" / "llama31-8b-instruct"))
ap.add_argument("--max-seq", type=int, default=1024)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--bs", type=int, default=1)
ap.add_argument("--ga", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--out", default=str(ROOT / "outputs" / "qlora"))
args = ap.parse_args()
out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# ---- model: load fp16 weights, quantise to NF4 on the fly (QLoRA) ----
model, tok = FastLanguageModel.from_pretrained(
    model_name=args.model, max_seq_length=args.max_seq, dtype=None, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=args.r, lora_alpha=args.r, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=3407)

# ---- data: render with the Llama 3.1 chat template, drop (not truncate) over-long examples ----
ds = load_dataset("json", data_files={"train": str(ROOT / "data" / "train.jsonl"),
                                      "validation": str(ROOT / "data" / "val.jsonl")})


def render(ex):
    text = tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": text.removeprefix(tok.bos_token)}      # the tokenizer re-adds BOS; avoid a double BOS


ds = ds.map(render, remove_columns=ds["train"].column_names)
before = {k: len(v) for k, v in ds.items()}
ds = ds.filter(lambda e: len(tok(e["text"])["input_ids"]) <= args.max_seq)
print("examples before/after length filter:", before, {k: len(v) for k, v in ds.items()})

# ---- trainer (arguments differ slightly across TRL versions, so detect them) ----
bf16 = torch.cuda.is_bf16_supported()
cfg = dict(output_dir=str(out / "ckpt"), per_device_train_batch_size=args.bs, per_device_eval_batch_size=1,
           gradient_accumulation_steps=args.ga, num_train_epochs=args.epochs, learning_rate=args.lr,
           lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.01, optim="adamw_8bit",
           bf16=bf16, fp16=not bf16, logging_steps=10, eval_strategy="steps", eval_steps=50,
           save_strategy="steps", save_steps=50, save_total_limit=2, seed=3407, report_to="none",
           dataset_text_field="text", dataset_num_proc=1, packing=False)
fields = inspect.signature(SFTConfig.__init__).parameters
cfg["max_seq_length" if "max_seq_length" in fields else "max_length"] = args.max_seq
tkw = {"processing_class": tok} if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters else {"tokenizer": tok}


class Vram(TrainerCallback):
    """Tracks (a) PyTorch's reserved peak and (b) whole-device usage (includes CUDA context + other processes)."""
    def __init__(self):
        self.reserved = 0
        self.device = 0

    def on_step_end(self, a, state, control, **kw):
        self.reserved = max(self.reserved, torch.cuda.max_memory_reserved())
        free, total = torch.cuda.mem_get_info()
        self.device = max(self.device, total - free)


vram = Vram()
trainer = SFTTrainer(model=model, train_dataset=ds["train"], eval_dataset=ds["validation"],
                     args=SFTConfig(**cfg), callbacks=[vram], **tkw)
# loss only on the assistant answer, so validation loss measures answer quality, not prompt memorisation
trainer = train_on_responses_only(
    trainer, instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n")

baseline = trainer.evaluate()["eval_loss"]                 # step-0 loss BEFORE any training: your honest "before"
print(f"baseline eval loss: {baseline:.3f}")
t0 = time.time()
trainer.train(resume_from_checkpoint=any((out / "ckpt").glob("checkpoint-*")) or None)
minutes = (time.time() - t0) / 60
final = trainer.evaluate()["eval_loss"]

model.save_pretrained(str(out / "adapter"))
tok.save_pretrained(str(out / "adapter"))
gb = 1024 ** 3
metrics = {"baseline_eval_loss": baseline, "final_eval_loss": final, "train_minutes": round(minutes, 1),
           "peak_torch_reserved_gb": round(vram.reserved / gb, 2), "peak_device_used_gb": round(vram.device / gb, 2),
           "train_examples": len(ds["train"]), "val_examples": len(ds["validation"]),
           "max_seq": args.max_seq, "lora_r": args.r, "effective_batch": args.bs * args.ga, "epochs": args.epochs}
json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
json.dump(trainer.state.log_history, open(out / "log_history.json", "w"), indent=1)
print(json.dumps(metrics, indent=2))
