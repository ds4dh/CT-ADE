import os
import json
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np
import evaluate
from custom_metrics import BalancedAccuracy
import shutil
from model import ADEModel
from typing import List, Optional
import os
import datetime
import inspect

def extract_last_folder_name(path):
    """
    Extracts the last folder name from a given file path.

    Args:
    path (str): The file path from which to extract the last folder name.

    Returns:
    str or None: The name of the last folder if it exists, otherwise None.
    """
    # Normalize the path to handle different OS path separators and trailing slashes
    path = os.path.normpath(path)

    # Split the path into parts
    parts = path.split(os.sep)

    # Iterate over the parts from the end to the beginning and find the first non-empty part
    for part in reversed(parts):
        if part:  # Checking if the part is non-empty
            return part

    # If no non-empty part is found (unlikely unless the input path is empty or malformed)
    return ""

def evaluate_features(use_features):
    """
    Evaluates the dictionary of features and returns a string based on their boolean values.

    Args:
    use_features (dict): A dictionary with keys 'eligibility_criteria', 'group_description', and 'smiles',
                         and boolean values.

    Returns:
    str: A string indicating which features are True ('all', 'group_smiles', 'smiles').
    """
    # Default responses if keys are missing
    eligibility = use_features.get('eligibility_criteria', False)
    group_description = use_features.get('group_description', False)
    smiles = use_features.get('smiles', False)

    # Evaluate conditions based on the boolean values
    if eligibility and group_description and smiles:
        return "all"
    elif group_description and smiles:
        return "group_smiles"
    elif smiles and not group_description and not eligibility:
        return "smiles"
    return ""

def extract_model_name(filepath):
    """
    Extracts the model name from a given filepath that includes the model name followed by a timestamp.

    Args:
    filepath (str): The file path from which to extract the model name.

    Returns:
    str: The extracted model name or None if the format is not as expected.
    """
    # Splitting the file path on slashes to isolate the last part, the filename
    if filepath is None:
        return ""
        
    parts = filepath.split('/')
    if not parts:
        return None  # Return None if the split result is empty
    
    filename = parts[-1]  # Get the last part of the path, which should be the filename
    
    # Split the filename by underscore
    model_parts = filename.split('_')
    if len(model_parts) < 2:
        return None  # Return None if there aren't enough parts to match the expected pattern

    # Assuming the model name is the first part before the timestamp which starts with a date
    model_name = '_'.join(model_parts[:-2])  # Join all parts excluding the last three (date and time)

    return model_name

def create_run_id(dataset_path, use_features, path_to_pretrained_model, negative_sampling_ratio):
    """
    Creates a unique run identifier based on the dataset path, feature usage, and a pretrained model path.

    Args:
    dataset_path (str): Path to the dataset directory.
    use_features (dict): Dictionary of features with boolean values.
    path_to_pretrained_model (str): Path to the pretrained model file.

    Returns:
    str: A unique run identifier string.
    """
    # Extract components for the run ID
    last_folder_name = extract_last_folder_name(dataset_path) or "none"
    feature_evaluation = evaluate_features(use_features) or "none"
    model_name = extract_model_name(path_to_pretrained_model) or "none"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Determine the negative sampling status
    neg_sampling = "true" if negative_sampling_ratio is not None else "false"

    # Combine all parts to form the run ID
    run_id = f"{last_folder_name}_{feature_evaluation}_neg_samp_{neg_sampling}_pretrained_{model_name}_{timestamp}"
    return run_id


def load_ct_ade_data(path: str, columns: Optional[List[str]] = None) -> DatasetDict:
    """
    Load CT-ADE classification dataset from specified directory and return a combined Hugging Face DatasetDict.
    Automatically includes all columns starting with 'label_', ensuring these label columns are ordered to the right of any user-specified columns.

    Args:
        path (str): The directory path that contains 'train.csv', 'val.csv', and 'test.csv'.
        columns (Optional[List[str]]): List of column names to load from the CSV files, excluding 'label_' columns which are always included. If None, all columns are loaded.

    Returns:
        DatasetDict: A dictionary of datasets containing separate entries for training, validation, and test sets, each with the specified features and all 'label_' features.

    Raises:
        FileNotFoundError: If any of the CSV files are not found in the specified directory.
        ValueError: If specified columns are not found in the CSV files.
    """
    # Define the paths to the CSV files
    train_path = os.path.join(path, "train.csv")
    val_path = os.path.join(path, "val.csv")
    test_path = os.path.join(path, "test.csv")

    # Check if files exist before loading
    for file_path in [train_path, val_path, test_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")

    # Helper function to load CSV with optional columns
    def load_csv(file_path, columns):
        df = pd.read_csv(file_path)
        all_columns = df.columns.tolist()
        label_columns = [col for col in all_columns if col.startswith("label_")]

        if columns:
            # Combine columns specified by user and label columns, maintaining order
            specified_and_label_columns = [
                col for col in columns if col in all_columns
            ] + [col for col in label_columns if col not in columns]
            return df[specified_and_label_columns]
        else:
            return df

    # Load data
    try:
        train_df = load_csv(train_path, columns)
        val_df = load_csv(val_path, columns)
        test_df = load_csv(test_path, columns)
    except ValueError as e:
        raise ValueError(f"Error loading columns from CSV: {e}")

    # Convert dataframes to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Combine datasets into a DatasetDict
    dataset_dict = DatasetDict(
        {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    )

    return dataset_dict


def load_tokenizer(llm_key: str, max_length: int) -> PreTrainedTokenizer:
    """
    Load HF tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(llm_key)

    if max_length:
        tokenizer.model_max_length = max_length

    return tokenizer


def tokenize_features(
    batch: dict,
    tokenizer: PreTrainedTokenizer,
    features_to_tokenize: List[str],
) -> dict:
    """
    Tokenizes specified features in a batch of data using a given tokenizer and prioritizes tokenized data in the output.

    Args:
        batch (dict): A dictionary where keys are feature names and values are lists of data.
        tokenizer (PreTrainedTokenizer): An instance of a tokenizer.
        features_to_tokenize (List[str]): List of feature names to tokenize.

    Returns:
        dict: The dictionary with tokenized features first, followed by the rest of the batch.
    """
    tokenizer_kwargs = {"padding": "max_length", "truncation": True}

    tokenized_data = {}

    for feature in features_to_tokenize:
        if feature in batch:
            encoded = tokenizer(batch[feature], **tokenizer_kwargs)
            tokenized_data[f"{feature}_input_ids"] = encoded.input_ids
            tokenized_data[f"{feature}_attention_mask"] = encoded.attention_mask
            del batch[feature]
        else:
            raise ValueError(f"Feature '{feature}' not found in the batch")

    return {**tokenized_data, **batch}


def process_labels(batch: dict) -> dict:
    """Retrieve labels from raw data batch

    Args:
        batch (dict): _description_
    """
    label_data = []

    keys = list(batch.keys())
    for key in keys:
        if key.startswith("label_"):
            label_data.append(batch[key])
            del batch[key]

    label_data = {"labels": torch.tensor(label_data).transpose(0, 1)}

    return {**label_data, **batch}


def compute_metrics_comprehensive(eval_pred, label_names):

    logits, labels = eval_pred
    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
    predictions = np.where(probabilities > 0.5, 1, 0)
    flat_probabilities = probabilities.flatten()
    flat_predictions = predictions.flatten()
    flat_labels = labels.flatten()

    results = {}

    ba_score = BalancedAccuracy()
    f1_score = evaluate.load("f1")
    pr_score = evaluate.load("precision")
    re_score = evaluate.load("recall")
    ac_score = evaluate.load("accuracy")
    roc_auc_score = evaluate.load("roc_auc")

    ba_micro = ba_score.compute(
        predictions=flat_predictions,
        references=flat_labels
    )["balanced_accuracy"]
    f1_micro = f1_score.compute(
        predictions=flat_predictions,
        references=flat_labels,
        average="binary"
    )["f1"]
    pr_micro = pr_score.compute(
        predictions=flat_predictions,
        references=flat_labels,
        average="binary"
    )["precision"]
    re_micro = re_score.compute(
        predictions=flat_predictions,
        references=flat_labels,
        average="binary"
    )["recall"]
    ac_micro = ac_score.compute(
        predictions=flat_predictions,
        references=flat_labels
    )["accuracy"]
    roc_auc_micro = roc_auc_score.compute(
        prediction_scores=flat_probabilities,
        references=flat_labels
    )["roc_auc"]

    results.update(
        {
            "ba_micro": ba_micro,
            "f1_micro": f1_micro,
            "pr_micro": pr_micro,
            "re_micro": re_micro,
            "ac_micro": ac_micro,
            "roc_auc_micro": roc_auc_micro
        }
    )

    for i in range(labels.shape[1]):

        label_probs = probabilities[:, i]
        label_preds = predictions[:, i]
        label_true = labels[:, i]
        label_name = label_names[i]

        results[f"{label_name}_ba"] = ba_score.compute(
            predictions=label_preds,
            references=label_true
        )["balanced_accuracy"]
        results[f"{label_name}_f1"] = f1_score.compute(
            predictions=label_preds,
            references=label_true,
            average="binary"
        )["f1"]
        results[f"{label_name}_pr"] = pr_score.compute(
            predictions=label_preds,
            references=label_true,
            average="binary"
        )["precision"]
        results[f"{label_name}_re"] = re_score.compute(
            predictions=label_preds,
            references=label_true,
            average="binary"
        )["recall"]
        results[f"{label_name}_ac"] = ac_score.compute(
            predictions=label_preds,
            references=label_true
        )["accuracy"]
        if np.ptp(label_true) != 0:
            results[f"{label_name}_roc_auc"] = roc_auc_score.compute(
                prediction_scores=label_probs,
                references=label_true
            )["roc_auc"]
        else:
            results[f"{label_name}_roc_auc"] = 0

    return results


def compute_metrics(eval_pred, label_names):

    logits, labels = eval_pred
    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
    predictions = np.where(probabilities > 0.5, 1, 0)
    flat_probabilities = probabilities.flatten()
    flat_predictions = predictions.flatten()
    flat_labels = labels.flatten()

    results = {}

    ba_score = BalancedAccuracy()
    f1_score = evaluate.load("f1")
    pr_score = evaluate.load("precision")
    re_score = evaluate.load("recall")
    ac_score = evaluate.load("accuracy")
    roc_auc_score = evaluate.load("roc_auc")

    ba_micro = ba_score.compute(
        predictions=flat_predictions,
        references=flat_labels
    )["balanced_accuracy"]
    f1_micro = f1_score.compute(
        predictions=flat_predictions,
        references=flat_labels,
        average="binary"
    )["f1"]

    results.update(
        {
            "ba_micro": ba_micro,
            "f1_micro": f1_micro
        }
    )

    return results


def get_path_base_name(path):
    path = os.path.normpath(path)
    return os.path.basename(path)


def get_unique_output_dir(base_dir="models", identifier=None):
    unique_path = os.path.join(base_dir, identifier)
    os.makedirs(unique_path, exist_ok=True)
    return unique_path


def save_training_config(cfg, output_dir):
    config_dict = {
        key: getattr(cfg, key) for key in dir(cfg) if not key.startswith("__")
    }
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w") as json_file:
        json.dump(config_dict, json_file, indent=4)


def save_inference_config(config_dict, output_dir):
    config_path = os.path.join(output_dir, "inference_config.json")
    with open(config_path, "w") as json_file:
        json.dump(config_dict, json_file, indent=4)


def save_model_state_to_cpu(model, output_dir, filename="model_state.pt"):
    """
    Saves the model's state dictionary to a specified directory after moving it to the CPU.

    Args:
    model (torch.nn.Module): The model whose state_dict is to be saved.
    output_dir (str): The directory where the model state dictionary should be saved.
    filename (str): The name of the file to save the model state. Default is 'model_state.pt'.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Move the model to CPU and save its state dictionary
    model_cpu = model.to("cpu")
    save_path = os.path.join(output_dir, filename)
    torch.save(model_cpu.state_dict(), save_path)
    print(f"Model state dictionary saved to {save_path}")


def delete_checkpoint_dirs(output_dir):
    """
    Deletes directories starting with 'checkpoint-' in the specified output directory.

    Args:
    output_dir (str): The directory from which to delete the checkpoint directories.
    """
    # List all items in the output directory
    for item in os.listdir(output_dir):
        # Construct full path
        item_path = os.path.join(output_dir, item)
        # Check if it is a directory and starts with 'checkpoint-'
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            # Remove the directory and all its contents
            shutil.rmtree(item_path)
            print(f"Deleted checkpoint directory: {item_path}")


def load_pretrained_model(model_path: str):
    """
    Load a pre-trained model and its weights from the specified path.

    Args:
        model_path (str): Path to the directory containing the pre-trained model files.

    Returns:
        ADEModel: Loaded pre-trained model.
    """
    # Define paths to inference config and weights
    inference_config_path = os.path.join(model_path, "inference_config.json")
    weights_path = os.path.join(model_path, "model_state.pt")

    # Load inference config
    with open(inference_config_path, "r") as cf:
        inference_config = json.load(cf)

    # Get the constructor parameters of ADEModel
    model_constructor_params = inspect.signature(ADEModel).parameters

    # Filter the inference config to only include parameters that ADEModel accepts
    filtered_config = {
        k: v for k, v in inference_config.items() if k in model_constructor_params
    }

    # Initialize the model with the filtered config
    trained_model = ADEModel(**filtered_config)

    # Load weights into the model
    trained_model.load_state_dict(torch.load(weights_path))

    return trained_model