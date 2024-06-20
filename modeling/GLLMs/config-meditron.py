class Config:
    # HF
    model_name = "epfl-llm/meditron-7b"
    token = "..." # Put your HF token here

    # Training
    output_dir = "./meditron-7b-SOC"
    num_train_epochs = 5
    per_device_train_batch_size = 2 # 2 for 7/8B, 1 for 70B
    per_device_eval_batch_size = 2 # 2 for 7/8B, 1 for 70B
    gradient_accumulation_steps = 2 # 2 for 7/8B, 4 for 70B
    gradient_checkpointing = False
    max_grad_norm = 1.0
    learning_rate = 5e-5
    max_seq_length = 4096

    # LoRa
    peft_lora_r = 16 # 16
    peft_lora_alpha = 32 # 8
    peft_lora_dropout = 0.05 # 0.05

    # Prediction
    max_new_tokens_for_generation = 315

    # Chat template
    chat_template = "{% set loop_messages = messages %}\n{% for message in loop_messages %}\n{% if loop.index0 == 0 %}\n{{ message['content'] | trim }}\n{% elif loop.index == loop_messages|length %}\n### {{ message['role'].lower().capitalize() }}: {{ message['content'] | trim }}{% else %}\n### {{ message['role'].lower().capitalize() }}: {{ message['content'] | trim }}\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt %}{{ '\n### Assistant: ' }}{% endif %}\n{% if not add_generation_prompt %}{{ ' ' + eos_token }}{% endif %}"
    on_completion_ids = [13, 2277, 29937, 4007, 22137, 29901]
