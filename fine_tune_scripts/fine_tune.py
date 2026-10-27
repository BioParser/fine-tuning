# The script assumes an Alpaca formatted dataset
# (pylance has issues with the trl import and model merges)

from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from huggingface_hub import login

# ===================================================
# CONFIGS
# ===================================================

# HuggingFace repository name
HF_REPO_ID = "" # HUGGING FACE REPO NAME HERE
# HuggingFace login
login(token="") # HUGGING FACE WRITE TOKEN HERE
# Model id
MODEL_ID = "Qwen/Qwen3.5-0.8B"

# ===================================================
# TRAINING
# ===================================================

# 1. Load the Alpaca dataset from local JSONL
DATASET = load_dataset("json", data_files="./training_data.jsonl", split="train")

# 2. Format the JSONL samples into a single string
def formatting_func(example):
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )

# 3. Training config
training_config = SFTConfig(
    output_dir="./results",
    per_device_train_batch_size=2, # amount of samples per forward pass
    gradient_accumulation_steps=4, # forward passes per single batch (thus batch size is 2 x 4 = 8)
    learning_rate=2e-4,
    logging_steps=10, # logger and metric interval
    save_strategy="no", # when to checkpoint the model, other options are "epoch" and "steps"
)

# 4. PEFT config
# (default config)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 5. Initialize trainer
trainer = SFTTrainer(
    model=MODEL_ID,
    train_dataset=DATASET,
    formatting_func=formatting_func,
    peft_config=peft_config,
    args=training_config,
)

# 6. Start training
trainer.train()

# ===================================================
# MODEL EXPORT
# ===================================================
print("Merging LoRA weights into the base model...") # merge the adapter into the base model
merged_model = trainer.model.merge_and_unload()

# Load a fresh tokenizer instance for export purposes
print("Loading tokenizer for upload...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Pushing merged model and tokenizer to Hugging Face Hub ({HF_REPO_ID})...")
merged_model.push_to_hub(HF_REPO_ID)
tokenizer.push_to_hub(HF_REPO_ID)

print("Export complete!")