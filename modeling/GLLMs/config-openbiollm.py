class Config:
    # HF
    model_name = "aaditya/Llama3-OpenBioLLM-8B"
    token = "..." # Put your HF token here

    # Training
    output_dir = "./Llama3-OpenBioLLM-8B-SOC"
    num_train_epochs = 5
    per_device_train_batch_size = 2 # 2 for 7/8B, 1 for 70B
    per_device_eval_batch_size = 2 # 2 for 7/8B, 1 for 70B
    gradient_accumulation_steps = 2 # 2 for 7/8B, 4 for 70B
    gradient_checkpointing = False
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
    chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|end_of_text|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    on_completion_ids = None
