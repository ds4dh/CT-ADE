class Config:
    # HF
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    token = "..." # Put your HF token here

    # Training
    output_dir = "./Meta-Llama-3-8B-Instruct-SOC"
    num_train_epochs = 5
    per_device_train_batch_size = 2 # 2 for 7/8B, 1 for 70B
    per_device_eval_batch_size = 2 # 2 for 7/8B, 1 for 70B
    gradient_accumulation_steps = 2 # 2 for 7/8B, 4 for 70B
    gradient_checkpointing = False # False for 7/8B, True for 70B
    max_grad_norm = 1.0
    learning_rate = 5e-5
    max_seq_length = None

    # LoRa
    peft_lora_r = 16 # 16
    peft_lora_alpha = 32 # 8
    peft_lora_dropout = 0.05 # 0.05

    # Prediction
    max_new_tokens_for_generation = 315

    # Chat template
    chat_template = None
    on_completion_ids = None
