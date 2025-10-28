import os, math, random
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["WANDB_DISABLED"] = "true"
import pandas as pd
import torch, numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sentence_transformers import SentenceTransformer, InputExample, losses
from peft import LoraConfig, get_peft_model

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# ============== Config ==============
CSV_PATH      = "total_training_data.xlsx"   # <-- set this
TEXT_COL_Q    = "Question"
TEXT_COL_D    = "chunk_text"
LABEL_COL     = "label"              # 0/1
# MAX_LEN       = 1024
BATCH_SIZE    = 4
EPOCHS        = 2
LR            = 1e-5
WARMUP_RATIO  = 0.05
OUTPUT_DIR    = "/home/qdai/embeddings_new_try/qwen3-0p6b-embed-lora-contrastive-without-max-len-attention-layer"
OUTPUT_DIR    = "/home/qdai/embeddings_new_try/qwen3-0p6b-embed-lora-contrastive-without-max-len-all-layers"
MODEL_ID      = "/home/qdai/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"

DEFAULT_INSTRUCT = (
    "Given a search query, retrieve relevant passages that answer the query."
)

def format_query(q: str, instruction: str | None = None) -> str:
    ins = instruction or DEFAULT_INSTRUCT
    return f"<Instruct>: {ins}\n<Query>: {q}"

# ============== Load & Prep Data ==============
df = pd.read_excel(CSV_PATH, dtype=str, keep_default_na=False)
df = df[[TEXT_COL_Q, TEXT_COL_D, LABEL_COL]].dropna()
df[LABEL_COL] = df[LABEL_COL].astype(int).clip(0,1)

# Build InputExamples: (anchor=query+prompt, pair=doc+prompt, label=0/1)
examples = [
    InputExample(
        texts=[format_query(q, ""), d],
        label=float(lbl)
    )
    for q, d, lbl in df[[TEXT_COL_Q, TEXT_COL_D, LABEL_COL]].itertuples(index=False)
]

# Weighted sampler to counter class imbalance (~900 pos vs ~5100 neg)
# pos_count = (df[LABEL_COL] == 1).sum()
# neg_count = (df[LABEL_COL] == 0).sum()
# w_pos = 0.5 / max(pos_count, 1)
# w_neg = 0.5 / max(neg_count, 1)
# weights = torch.tensor([w_pos if e.label > 0 else w_neg for e in examples], dtype=torch.float)

# sampler = WeightedRandomSampler(weights, num_samples=len(examples), replacement=True)
train_loader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

# ============== Model + LoRA ==============
model = SentenceTransformer(MODEL_ID, device="cuda")
# model.max_seq_length = MAX_LEN

# Enforce left padding (Qwen embedding tokenizer)
model.tokenizer.padding_side = "left"
if model.tokenizer.pad_token is None:
    model.tokenizer.pad_token = model.tokenizer.eos_token

# Enable grad checkpointing (VRAM saver)
base_hf = model._first_module().auto_model
base_hf.gradient_checkpointing_enable()

# Attach LoRA to attention projections
lconf = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
    #               "gate_proj", "up_proj", "down_proj",],
)
peft_backbone = get_peft_model(base_hf, lconf)
model._first_module().auto_model = peft_backbone

# ============== Loss & Train ==============
# ContrastiveLoss expects (text1, text2, label in {0,1})
loss = losses.ContrastiveLoss(model=model, margin=0.5)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
use_amp = True
warmup_steps = int(len(train_loader) * EPOCHS * WARMUP_RATIO)

model.fit(
    train_objectives=[(train_loader, loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path=OUTPUT_DIR,
    use_amp=use_amp,
    optimizer_params={"lr": LR},
    scheduler="WarmupLinear",
    save_best_model=True,
)

print("Saved:", OUTPUT_DIR)