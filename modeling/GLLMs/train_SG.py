import os
from config import Config as cfg
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from transformers import EarlyStoppingCallback
import pandas as pd
from datasets import Dataset
from peft import LoraConfig
import torch
import random
import json
from tqdm.auto import tqdm
import re
from torch.utils.data import DataLoader

### TASK SPECIFIC ###
path_to_data = "../../data/ct_ade/soc"

labels_to_description = {
    "label_10005329": "Blood and lymphatic system disorders",
    "label_10007541": "Cardiac disorders",
    "label_10010331": "Congenital, familial and genetic disorders",
    "label_10013993": "Ear and labyrinth disorders",
    "label_10014698": "Endocrine disorders",
    "label_10015919": "Eye disorders",
    "label_10017947": "Gastrointestinal disorders",
    "label_10018065": "General disorders and administration site conditions",
    "label_10019805": "Hepatobiliary disorders",
    "label_10021428": "Immune system disorders",
    "label_10021881": "Infections and infestations",
    "label_10022117": "Injury, poisoning and procedural complications",
    "label_10022891": "Investigations",
    "label_10027433": "Metabolism and nutrition disorders",
    "label_10028395": "Musculoskeletal and connective tissue disorders",
    "label_10029104": "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
    "label_10029205": "Nervous system disorders",
    "label_10036585": "Pregnancy, puerperium and perinatal conditions",
    "label_10037175": "Psychiatric disorders",
    "label_10038359": "Renal and urinary disorders",
    "label_10038604": "Reproductive system and breast disorders",
    "label_10038738": "Respiratory, thoracic and mediastinal disorders",
    "label_10040785": "Skin and subcutaneous tissue disorders",
    "label_10041244": "Social circumstances",
    "label_10042613": "Surgical and medical procedures",
    "label_10047065": "Vascular disorders",
    "label_10077536": "Product issues",
}


def create_user_input(row):

    eligibility_criteria = row["eligibility_criteria"]
    group_description = row["group_description"]
    smiles = row["smiles"]

    return f"""The treatment regimen is as follows:

```
{group_description.strip()}
```

The SMILES representation of the drug is as follows:

```
{smiles.strip()}
```

Use the above treatment regimen and SMILES to predict the adverse drug events that may occur according to the following MedDRA system organ classes:

['Blood and lymphatic system disorders',
 'Cardiac disorders',
 'Congenital, familial and genetic disorders',
 'Ear and labyrinth disorders',
 'Endocrine disorders',
 'Eye disorders',
 'Gastrointestinal disorders',
 'General disorders and administration site conditions',
 'Hepatobiliary disorders',
 'Immune system disorders',
 'Infections and infestations',
 'Injury, poisoning and procedural complications',
 'Investigations',
 'Metabolism and nutrition disorders',
 'Musculoskeletal and connective tissue disorders',
 'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
 'Nervous system disorders',
 'Pregnancy, puerperium and perinatal conditions',
 'Psychiatric disorders',
 'Renal and urinary disorders',
 'Reproductive system and breast disorders',
 'Respiratory, thoracic and mediastinal disorders',
 'Skin and subcutaneous tissue disorders',
 'Social circumstances',
 'Surgical and medical procedures',
 'Vascular disorders',
 'Product issues']"""


def create_assistant_output(row):
    label_columns = [col for col in row.index if col.startswith("label_")]
    labels_list = [
        labels_to_description[label_name]
        for label_name in label_columns
        if row[label_name] == 1.0
    ]
    if not labels_list:
        return "[]"

    formatted_list = "[" + ",\n ".join(f"'{item}'" for item in labels_list) + "]"
    return formatted_list


def create_training_instance(row):

    messages = [
        {
            "role": "system",
            "content": "You are an AI system designed to predict adverse drug events based on treatment regimens and SMILES representations of drugs.",
        },
        {"role": "user", "content": create_user_input(row)},
        {"role": "assistant", "content": create_assistant_output(row)},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    return prompt


def create_inference_instance(row):

    messages = [
        {
            "role": "system",
            "content": "You are an AI system designed to predict adverse drug events based on treatment regimens and SMILES representations of drugs.",
        },
        {"role": "user", "content": create_user_input(row)},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return prompt
### END TASK SPECIFIC ###


def shuffle_list(original_list, seed=None):
    shuffled_list = original_list.copy()
    random.seed(seed)
    random.shuffle(shuffled_list)
    return shuffled_list


def extract_generation_prompt(template):
    # If chat template have consistent format this should work for all GLLMs.
    pattern = r"if add_generation_prompt.*?'([^']*)'"
    match = re.search(pattern, template, re.DOTALL)
    if match:
        return match.group(1)
    return None


def generate_predictions(test_dataset, tokenizer, model, max_new_tokens, batch_size):

    model.eval()
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    predictions = []

    terminators = [tokenizer.eos_token_id]

    # Create a DataLoader for batch processing
    data_loader = DataLoader(test_dataset["text"], batch_size=batch_size, shuffle=False)

    for batch in tqdm(data_loader):
        # Tokenize the batch
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True
        )
        input_ids = tokenized.input_ids.to(model.device)
        attention_mask = tokenized.attention_mask.to(model.device)

        # Generate outputs
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            temperature=None,  # Explicitly unset temperature
            top_p=None,        # Explicitly unset top_p
            top_k=None         # Explicitly unset top_k
        )

        for i in range(len(batch)):
            response = outputs[i][input_ids[i].shape[-1] :]
            prediction = tokenizer.decode(response, skip_special_tokens=True)
            predictions.append(prediction)

    tokenizer.padding_side = original_padding_side
    return predictions


# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    cfg.model_name,
    use_fast=True,
    token=cfg.token,
)
if cfg.chat_template is not None:
    tokenizer.chat_template = cfg.chat_template
    print("-"*100)
    print("\nA CUSTOM CHAT TEMPLATE HAS BEEN APPLIED BASED ON config.py\n")
    print("-"*100)

# Initialize the model with Flash Attention 2
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_name,
    attn_implementation="flash_attention_2",  # Try to use Flash Attention 2
    device_map="auto",
    torch_dtype="auto",
    token=cfg.token,
    use_cache=False if cfg.gradient_checkpointing else True
)

if cfg.gradient_checkpointing:
    model.gradient_checkpointing_enable()

# In any case, we always create (or override) the padding token for simplicity
tokenizer.add_special_tokens({"pad_token": "<|PAD_FOR_LORA_FINE_TUNING|>"})
tokenizer.pad_token_id = tokenizer.vocab["<|PAD_FOR_LORA_FINE_TUNING|>"]
model.resize_token_embeddings(len(tokenizer))
# People generally use tokenizer.eos_token_id, but just in case you can add your own special token
# to make sure your model is trained to end turns.
# This will resize the model embedding but in all-linear (Q)LoRa you don't target this module.
# As a consequence, the (unchanged but resized) embedding layer will be saved in the adapter.
# To convert the full model into an HF-friendly format, we need to:
# 1 - Load the base model
# 2 - Resize the token embeddings
# 3 - Use peft to inject the adapter
# 4 - merge_and_unload the model
# 5 - push to hub

# Load data
train = pd.read_csv(os.path.join(path_to_data, "train.csv"))
val = pd.read_csv(os.path.join(path_to_data, "val.csv"))
test = pd.read_csv(os.path.join(path_to_data, "test.csv"))

# Apply function to create prompts
train_prompts = shuffle_list(train.apply(create_training_instance, axis=1).tolist(), seed=37)
val_prompts = val.apply(create_training_instance, axis=1).tolist()
test_prompts = test.apply(create_inference_instance, axis=1).tolist()

# Create the dataset
train_dataset = Dataset.from_dict({"text": train_prompts})
val_dataset = Dataset.from_dict({"text": val_prompts})
test_dataset = Dataset.from_dict({"text": test_prompts})

# Determine max_seq_length if not provided
if cfg.max_seq_length is None:
    dataset_tokenized = train_dataset.map(lambda x: tokenizer(x['text'], add_special_tokens=False), batched=True)
    max_seq_length = min(max([len(i) for i in dataset_tokenized["input_ids"]]), model.config.max_position_embeddings - 1) + 1
    del dataset_tokenized
else:
    max_seq_length = cfg.max_seq_length

# Detect completion ids if they are not provided in the configuration (some tokenizers are picky, so it may be a good idea to define them manually).
if cfg.on_completion_ids is None:
    response_template = extract_generation_prompt(
        tokenizer.chat_template
    )  # This method works for all GLLMs but we need to properly set the chat template
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
else:
    response_template_ids = cfg.on_completion_ids

# Initialize the Data Collator
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# Set up SFTConfig with all required arguments
sft_config = SFTConfig(
    output_dir=cfg.output_dir,
    num_train_epochs=cfg.num_train_epochs,
    per_device_train_batch_size=cfg.per_device_train_batch_size,
    per_device_eval_batch_size=cfg.per_device_eval_batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    max_grad_norm=cfg.max_grad_norm,
    learning_rate=cfg.learning_rate,
    bf16=(True if model.config.torch_dtype == torch.bfloat16 else False),
    fp16=(True if model.config.torch_dtype == torch.float16 else False),
    optim = "paged_adamw_32bit", # Paged AdamW optimizer with 32-bit precision; uses unified memory paging to handle OOM by transferring excess memory to CPU.
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    report_to="tensorboard",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    max_seq_length=max_seq_length,
    dataset_kwargs={
        "add_special_tokens": False,  # Make sure all special tokens are in the prompt
        "append_concat_token": False,  # No need to add additional separator token
    },
    gradient_checkpointing_kwargs={"use_reentrant": True} if cfg.gradient_checkpointing else {},
    save_only_model=True,
)

peft_config = LoraConfig(
    r=cfg.peft_lora_r,
    lora_alpha=cfg.peft_lora_alpha,
    target_modules="all-linear",
    lora_dropout=cfg.peft_lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    # modules_to_save = ["lm_head", "embed_tokens"] # If you want to use chat/instruct tokens and the base model didn't train them.
)

# Initialize the SFTTrainer with SFTConfig
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    dataset_text_field="text",
    args=sft_config,
    packing=False,  # We can't pack in CompletionOnlyLM
    peft_config=peft_config,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=1)  # GLLMs overfit very fast
    ],
)

if trainer.accelerator.is_main_process:
    trainer.model.print_trainable_parameters()

# Train the model
trainer.train()

# Make predictions
predictions = generate_predictions(
    test_dataset,
    tokenizer,
    model,
    max_new_tokens=cfg.max_new_tokens_for_generation,
    batch_size=cfg.per_device_eval_batch_size,
)
# Make sure you set max_new_tokens correctly for the task

# Save predictions to a JSON file
with open(os.path.join(cfg.output_dir, "predictions.json"), "w") as f:
    json.dump(predictions, f, indent=4)