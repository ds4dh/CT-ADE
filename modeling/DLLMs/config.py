from utils import create_run_id

class Config:
    # Data
    dataset_path = "../../data/ct_ade/soc"
    use_features = {
        "eligibility_criteria": True,
        "group_description": True,
        "smiles": True,
    }

    # Use pre-trained encoders
    path_to_pretrained_model = None
    transfer_text_encoder = True
    transfer_smiles_encoder = True
    transfer_label_embeddings = True
    transfer_mlp = True

    # Training
    metric_for_best_model = "f1_micro" # "ba_micro" or "f1_micro"
    negative_sampling_ratio = None

    num_train_epochs = 200
    early_stopping_patience = 10
    per_device_train_batch_size = 24
    per_device_eval_batch_size = 24
    gradient_accumulation_steps = 2
    max_grad_norm = 1.0
    learning_rate = 5e-05 / 2

    # Text model
    text_llm_key = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    text_llm_max_length = 512

    # SMILES model
    smiles_llm_key = "DeepChem/ChemBERTa-77M-MLM"
    smiles_llm_max_length = 512

    # ID
    run_id = create_run_id(dataset_path, use_features, path_to_pretrained_model, negative_sampling_ratio)
