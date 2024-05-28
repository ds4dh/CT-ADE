import os
import json
from config import Config as cfg
from functools import partial
from utils import (
    load_ct_ade_data,
    load_tokenizer,
    tokenize_features,
    process_labels,
    load_pretrained_model,
    compute_metrics,
    compute_metrics_comprehensive,
    get_path_base_name,
    get_unique_output_dir,
    save_model_state_to_cpu,
    delete_checkpoint_dirs,
    save_training_config,
    save_inference_config
)
from datasets import disable_progress_bars
from transformers import TrainingArguments, EarlyStoppingCallback, Trainer
from model import ADEModel
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

# Initialize Accelerator
script_accelerator = Accelerator()

if not script_accelerator.is_main_process:
    disable_progress_bars()
else:
    pass

# Load raw data and tokenizer(s)
used_features = [k for k, v in cfg.use_features.items() if v]
dataset = load_ct_ade_data(cfg.dataset_path, used_features)
text_tokenizer = load_tokenizer(
    cfg.text_llm_key,
    cfg.text_llm_max_length
)
smiles_tokenizer = load_tokenizer(
    cfg.smiles_llm_key,
    cfg.smiles_llm_max_length
)

# Tokenize dataset (text features)
text_features = [f for f in used_features if f != "smiles"]
text_used = len(text_features) > 0
if text_used:
    tokenization_func = partial(
        tokenize_features,
        tokenizer=text_tokenizer,
        features_to_tokenize=text_features,
    )
    dataset = dataset.map(tokenization_func, batched=True)

# Tokenize dataset (smiles)
smiles_used = "smiles" in used_features
if smiles_used:
    tokenization_func = partial(
        tokenize_features,
        tokenizer=smiles_tokenizer,
        features_to_tokenize=["smiles"],
    )
    dataset = dataset.map(tokenization_func, batched=True)

# Retrieve labels
label_names = [
    column.replace("label_", "")
    for column in dataset["train"].features
    if column.startswith("label_")
]
dataset = dataset.map(process_labels, batched=True)

model = ADEModel(
    num_labels=len(dataset["train"][0]["labels"]),
    text_llm_key=cfg.text_llm_key,
    smiles_llm_key=cfg.smiles_llm_key,
    text_features=text_features,
    smiles_used=smiles_used,
    negative_sampling_ratio=cfg.negative_sampling_ratio,
)

# Loading pre-trained encoders from other tasks
if cfg.path_to_pretrained_model:
    pretrained_model = load_pretrained_model(cfg.path_to_pretrained_model)

    # Transfer text_encoder if enabled in config
    if cfg.transfer_text_encoder and hasattr(model, 'text_encoder') and hasattr(pretrained_model, 'text_encoder'):
        if model.text_encoder is not None and pretrained_model.text_encoder is not None:
            model.text_encoder = pretrained_model.text_encoder
            if script_accelerator.is_main_process:
                print("Successfully transferred pretrained text_encoder.")
        else:
            if model.text_encoder is None and script_accelerator.is_main_process:
                print("Model's text_encoder is None. Could not transfer pretrained text_encoder.")
            if pretrained_model.text_encoder is None and script_accelerator.is_main_process:
                print("Pretrained model's text_encoder is None. Could not transfer text_encoder.")

    # Transfer smiles_encoder if enabled in config
    if cfg.transfer_smiles_encoder and hasattr(model, 'smiles_encoder') and hasattr(pretrained_model, 'smiles_encoder'):
        if model.smiles_encoder is not None and pretrained_model.smiles_encoder is not None:
            model.smiles_encoder = pretrained_model.smiles_encoder
            if script_accelerator.is_main_process:
                print("Successfully transferred pretrained smiles_encoder.")
        else:
            if model.smiles_encoder is None and script_accelerator.is_main_process:
                print("Model's smiles_encoder is None. Could not transfer pretrained smiles_encoder.")
            if pretrained_model.smiles_encoder is None and script_accelerator.is_main_process:
                print("Pretrained model's smiles_encoder is None. Could not transfer smiles_encoder.")

    # Transfer label_embeddings if enabled in config
    if cfg.transfer_label_embeddings and hasattr(model, 'label_embeddings') and hasattr(pretrained_model, 'label_embeddings'):
        if model.label_embeddings.num_embeddings == pretrained_model.label_embeddings.num_embeddings and \
           model.label_embeddings.embedding_dim == pretrained_model.label_embeddings.embedding_dim:
            model.label_embeddings = pretrained_model.label_embeddings
            if script_accelerator.is_main_process:
                print("Successfully transferred pretrained label_embeddings.")
        else:
            if script_accelerator.is_main_process:
                print("Pretrained model's label_embeddings dimensions do not match or are not set for transfer.")

    # Transfer MLP if enabled in config
    if cfg.transfer_mlp and hasattr(model, 'mlp') and hasattr(pretrained_model, 'mlp'):
        if all([
            model.mlp[i].weight.size() == pretrained_model.mlp[i].weight.size()
            for i in range(len(model.mlp))
        ]):
            model.mlp = pretrained_model.mlp
            if script_accelerator.is_main_process:
                print("Successfully transferred pretrained MLP.")
        else:
            if script_accelerator.is_main_process:
                print("Pretrained model's MLP dimensions do not match or are not set for transfer.")

    # Delete the pretrained model object
    del pretrained_model

    
def main() -> None:

    # Initialize an empty list to hold the output_dir
    output_dirs = [None]

    if script_accelerator.is_main_process:
        # Only the main process creates and configures the output directory
        output_dirs[0] = get_unique_output_dir(
            identifier=cfg.run_id
        )
        save_training_config(cfg, output_dirs[0])
        
        # Save inference configuration
        unified_config = {
            "num_labels": len(dataset["train"][0]["labels"]),
            "text_llm_key": cfg.text_llm_key,
            "smiles_llm_key": cfg.smiles_llm_key,
            "text_features": text_features,
            "smiles_used": smiles_used,
            "negative_sampling_ratio": cfg.negative_sampling_ratio,
            "label_names": label_names,
            "dataset_path": cfg.dataset_path,
            "use_features": cfg.use_features,
            "path_to_pretrained_model": cfg.path_to_pretrained_model,
            "transfer_text_encoder": cfg.transfer_text_encoder,
            "transfer_smiles_encoder": cfg.transfer_smiles_encoder,
            "transfer_label_embeddings": cfg.transfer_label_embeddings,
            "transfer_mlp": cfg.transfer_mlp,
            "metric_for_best_model": cfg.metric_for_best_model,
            "num_train_epochs": cfg.num_train_epochs,
            "early_stopping_patience": cfg.early_stopping_patience,
            "per_device_train_batch_size": cfg.per_device_train_batch_size,
            "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "max_grad_norm": cfg.max_grad_norm,
            "learning_rate": cfg.learning_rate,
            "text_llm_max_length": cfg.text_llm_max_length,
            "smiles_llm_max_length": cfg.smiles_llm_max_length,
            "run_id": cfg.run_id,
        }
        
        save_inference_config(unified_config, output_dirs[0])
    else:
        pass

    script_accelerator.wait_for_everyone()
    broadcast_object_list(output_dirs, from_process=0)
    output_dir = output_dirs[0]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_grad_norm=cfg.max_grad_norm,
        learning_rate=cfg.learning_rate,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        report_to="tensorboard",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best_model
    )

    # Create trainer object
    compute_metrics_label_names = partial(compute_metrics, label_names=label_names)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics_label_names,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)
        ],
    )

    trainer.train()
    script_accelerator.wait_for_everyone()

    # Create a new Trainer instance for evaluation (comprehensive metrics)
    compute_metrics_comprehensive_label_names = partial(compute_metrics_comprehensive, label_names=label_names)
    eval_trainer = Trainer(
        model=model,
        args=training_args,  # reuse the existing training args
        train_dataset=dataset["train"],  # even if not used
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics_comprehensive_label_names
    )
    
    test_metrics = eval_trainer.evaluate(dataset["test"])
    val_metrics = eval_trainer.evaluate(dataset["validation"])

    if script_accelerator.is_main_process:
        
        with open(f"{output_dir}/validation_metrics.json", "w") as f:
            json.dump(val_metrics, f, indent=4)

        with open(f"{output_dir}/test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=4)
            
        save_model_state_to_cpu(trainer.model, output_dir)
        delete_checkpoint_dirs(output_dir)
    else:
        pass


if __name__ == "__main__":
    main()