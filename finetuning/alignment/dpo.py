from trl import DPOTrainer, DPOConfig
from bitsandbytes import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

model_id = "google/gemma-2b"
adapter_path = "offline_finetuning/models/gemma-2b/1724075462/checkpoint-2000"
train_dataset = ""

tokenizer = AutoTokenizer.from_pretrained(model_id)
#TODO: Add custom tokens
# Load the base model.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/mixtral-8x7b-v0.1",
    load_in_4bit=True,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False

model = PeftModel.from_pretrained(
    model,
    adapter_path,
    is_trainable=True,
    adapter_name="train",
)
model.load_adapter(adapter_path, adapter_name="reference")

training_args = DPOConfig(
    beta=0.1,
    model_adapter_name="train",
    reference_adapter_name="reference",
)
dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()