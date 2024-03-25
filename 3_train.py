from src.modeling_helpers import *

path_to_data = './data/classification/smiles/train_augmented' #train_augmented or train_base
text_model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
smiles_model_name = 'DeepChem/ChemBERTa-77M-MLM'
text_max_len=512
smiles_max_len=512
batch_size = 16
num_epochs = 20
learning_rate=2e-5
device = 'cuda:0'
feature_use_config = {'group_desc': True, 'eligibility': True, 'smiles': True}
model_saving_path = './models/smiles/train_augmented/all/model.pt'

if __name__ == '__main__':

    # Extract the directory from model_saving_path
    directory = os.path.dirname(model_saving_path)

    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Load data
    train_df, val_df, test_df = load_trial_data(path_to_data)

    # Compute naive performance
    naive_perf = compute_baseline_test_performance(train_df, test_df)
    print(f'The naive micro F1-score on the test set is = {round(naive_perf*100, 2)}%.')

    # Instansiate tokenizers
    group_desc_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    eligibility_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    smiles_tokenizer = AutoTokenizer.from_pretrained(smiles_model_name)

    # Train dataset
    train_dataset = CustomDataset(train_df,
                                  group_desc_tokenizer,
                                  eligibility_tokenizer,
                                  smiles_tokenizer,
                                  text_max_len,
                                  smiles_max_len)
    # Validation datasets
    val_dataset = CustomDataset(val_df,
                                            group_desc_tokenizer,
                                            eligibility_tokenizer,
                                            smiles_tokenizer,
                                            text_max_len,
                                            smiles_max_len)

    # Test datasets
    test_dataset = CustomDataset(test_df,
                                             group_desc_tokenizer,
                                             eligibility_tokenizer,
                                             smiles_tokenizer,
                                             text_max_len,
                                             smiles_max_len)

    # Train dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=custom_collate_fn)

    # Validation dataloaders
    val_dataloader = DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            collate_fn=custom_collate_fn)

    # Test dataloaders
    test_dataloader = DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             collate_fn=custom_collate_fn)

    # Instansiate model
    NUM_LABELS = train_dataset.labels.shape[-1]
    smiles_tokenizer.add_tokens(['[PLACEBO]', '[NOSMILES]'])
    smiles_tokenizer_len = len(smiles_tokenizer)
    model = MultilabelModel(text_model_name,
                            smiles_model_name,
                            smiles_tokenizer_len,
                            NUM_LABELS,
                            feature_use_config=feature_use_config)

    # Train, evaluate, and save the best model
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val_f1 = 0.0

    for epoch in range(num_epochs):

        avg_train_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            loss_fn,
            device,
            epoch,
            num_epochs
        )

        # Validation steps
        val_metrics = validate(
            model,
            val_dataloader,
            device, epoch,
            num_epochs
        )
        val_f1_micro = val_metrics[0]

        # Save model if performance improved
        best_val_f1 = save_model_if_best_performance(
            val_f1_micro,
            best_val_f1,
            model,
            model_saving_path
        )

        # Display progress
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val F1: {val_f1_micro:.4f}.')

    # Load the best model after training
    model.load_state_dict(torch.load(model_saving_path))

    # testidation steps
    test_metrics = validate_full(
        model,
        test_dataloader,
        device, epoch,
        num_epochs
    )

    # Assuming test_metrics_non_placebo is obtained from the updated calculate_evaluation_metrics_ function
    f1_micro, overall_PPV, overall_NPV, overall_Recall, overall_Accuracy, overall_BalancedAccuracy, F1s, PPVs, NPVs, Recalls, Accuracies, BalancedAccuracies = test_metrics

    # Assuming test_non_placebo_df and train_df are defined elsewhere in your code
    res = pd.DataFrame([
        test_df.drop(columns=["eligibility_criteria", "smiles", "group_description"]).describe().iloc[1, :].values.tolist(),
        F1s,
        PPVs,
        NPVs,
        Recalls,
        Accuracies,
        BalancedAccuracies
    ])

    overall_prevalence = np.mean(test_df.drop(columns=["eligibility_criteria", "smiles", "group_description"]).values.flatten())

    res.insert(0, "Prevalence", [overall_prevalence, f1_micro, overall_PPV, overall_NPV, overall_Recall, overall_Accuracy, overall_BalancedAccuracy])
    res = res.T.reset_index(drop=True)
    res.columns = ["Prevalence", "F1", "PPV", "NPV", "Recall", "Accuracy", "Balanced Accuracy"]
    res.index = ["Overall"] + list(train_df.drop(columns=["eligibility_criteria", "smiles", "group_description"]).columns)
    res["Naive Accuracy"] = res.Prevalence.apply(lambda x: 1 - x if x <= 0.5 else x)
    res["Naive Balanced Accuracy"] = [0.5] * len(res)

    # Get the current date and time in a formatted string (e.g., '2024-01-02_15-30-00')
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a filename with the current date and time
    filename = f"results_{current_time}.csv"

    # Combine the directory and the filename
    full_path = os.path.join(directory, filename)

    # Save the DataFrame to a CSV file
    res.to_csv(full_path, index=True)

    print("Test results saved in provided path.")
