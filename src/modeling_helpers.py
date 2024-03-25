import os
import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.nn.modules.loss import _Loss
from sklearn.metrics import f1_score, balanced_accuracy_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer
from tqdm.auto import tqdm


def load_trial_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess trial data from CSV files.

    The function reads training, validation, and test data from CSV files,
    processes them by dropping specific columns, and segregates validation
    and test data into placebo and non-placebo groups.

    Args:
        path (str): The directory path where the train.csv, val.csv, and test.csv files are located.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        - train_df: Training data DataFrame.
        - val_placebo_df: Validation data DataFrame for placebo group.
        - val_non_placebo_df: Validation data DataFrame for non-placebo group.
        - test_placebo_df: Test data DataFrame for placebo group.
        - test_non_placebo_df: Test data DataFrame for non-placebo group.
    """

    # File paths for train, validation, and test datasets
    train_path = os.path.join(path, "train.csv")
    val_path = os.path.join(path, "val.csv")
    test_path = os.path.join(path, "test.csv")

    # Columns to be dropped from the datasets
    to_drop = ['nctid', 'group_id', 'title',
               'intervention_name', 'intervention_name_drugbank',
               'type', 'kingdom', 'atc_anatomical_main_group',
               'sequence','is_placebo', 'contain_placebo']

    # Load and process the training data
    train_df = pd.read_csv(train_path).drop(columns=to_drop)

    # Load and process the validation data
    val_df = pd.read_csv(val_path).drop(columns=to_drop)

    # Load and process the test data
    test_df = pd.read_csv(test_path).drop(columns=to_drop)

    return train_df, val_df, test_df


class MultilabelModel(nn.Module):
    """
    A multi-label classification model designed to handle text and SMILES (Simplified Molecular Input Line Entry System) data.
    It allows for dynamic feature selection, enabling the user to choose which inputs (group descriptions, eligibility criteria,
    and SMILES strings) should be included in the model's computations.

    Attributes:
        text_encoder (nn.Module): Encoder for text inputs.
        smiles_encoder (nn.Module): Encoder for SMILES strings.
        dropout (nn.Dropout): Dropout layer to mitigate overfitting.
        classifier (nn.Linear): Linear layer for classification.
        feature_use_config (dict): Configuration dict indicating which features are active.

    Args:
        text_model_name (str): Name of the pretrained model for text encoding.
        smiles_model_name (str): Name of the pretrained model for SMILES encoding.
        smiles_tokenizer_len (int): Length of the tokenizer for SMILES strings.
        num_labels (int): Number of labels for classification.
        feature_use_config (Optional[dict]): Specifies which of the features ('group_desc', 'eligibility', 'smiles') are used. Defaults to using all features.
    """
    def __init__(self, text_model_name: str, smiles_model_name: str, smiles_tokenizer_len: int, num_labels: int, feature_use_config=None):
        super(MultilabelModel, self).__init__()

        # Default configuration uses all features if none specified
        if feature_use_config is None:
            feature_use_config = {'group_desc': True, 'eligibility': True, 'smiles': True}
        self.feature_use_config = feature_use_config

        # Initialize text encoder if any text features are to be used
        self.text_encoder = AutoModel.from_pretrained(text_model_name) if feature_use_config.get('group_desc', False) or feature_use_config.get('eligibility', False) else None

        # Initialize SMILES encoder if it is to be used
        self.smiles_encoder = AutoModel.from_pretrained(smiles_model_name) if feature_use_config.get('smiles', False) else None
        if self.smiles_encoder:
            self.smiles_encoder.resize_token_embeddings(smiles_tokenizer_len)

        self.dropout = nn.Dropout(0.1)

        # Dynamically calculate input size for the classifier based on active features
        input_size = 0
        if self.text_encoder:
            input_size += self.text_encoder.config.hidden_size * sum(feature_use_config.get(k, False) for k in ['group_desc', 'eligibility'])
        if self.smiles_encoder and feature_use_config.get('smiles', False):
            input_size += self.smiles_encoder.config.hidden_size

        self.classifier = nn.Linear(input_size, num_labels)

    def forward(self, group_desc_input_ids: torch.Tensor, group_desc_attention_mask: torch.Tensor, eligibility_input_ids: torch.Tensor, eligibility_attention_mask: torch.Tensor, smiles_input_ids: torch.Tensor, smiles_attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model, processing inputs based on the configured active features.

        Args:
            group_desc_input_ids (torch.Tensor): Input IDs for group descriptions.
            group_desc_attention_mask (torch.Tensor): Attention masks for group descriptions.
            eligibility_input_ids (torch.Tensor): Input IDs for eligibility criteria.
            eligibility_attention_mask (torch.Tensor): Attention masks for eligibility criteria.
            smiles_input_ids (torch.Tensor): Input IDs for SMILES strings.
            smiles_attention_mask (torch.Tensor): Attention masks for SMILES strings.

        Returns:
            torch.Tensor: The logits resulting from the classifier.
        """
        outputs = []

        # Process group descriptions if enabled
        if self.text_encoder and self.feature_use_config.get('group_desc', False):
            group_desc_output = self.text_encoder(group_desc_input_ids, attention_mask=group_desc_attention_mask)[0][:, 0]
            outputs.append(group_desc_output)

        # Process eligibility criteria if enabled
        if self.text_encoder and self.feature_use_config.get('eligibility', False):
            eligibility_output = self.text_encoder(eligibility_input_ids, attention_mask=eligibility_attention_mask)[0][:, 0]
            outputs.append(eligibility_output)

        # Process SMILES strings if enabled
        if self.smiles_encoder and self.feature_use_config.get('smiles', False):
            smiles_output = self.smiles_encoder(smiles_input_ids, attention_mask=smiles_attention_mask)[0][:, 0]
            outputs.append(smiles_output)

        # Ensure that at least one feature is being processed
        if not outputs:
            raise ValueError("At least one feature must be selected for use.")

        # Concatenate the outputs from the active encoders
        concatenated = torch.cat(outputs, dim=1)

        # Apply dropout and classifier to obtain logits
        dropped = self.dropout(concatenated)
        logits = self.classifier(dropped)

        return logits


class CustomDataset(Dataset):
    """
    A custom dataset class for handling and tokenizing text and SMILES data.

    This class is designed to process datasets containing group descriptions,
    eligibility criteria, and SMILES (Simplified Molecular Input Line Entry System)
    strings, and prepare them for input into a neural network model.

    Attributes:
        data (pd.DataFrame): The entire dataframe.
        GroupDescription (pd.Series): Series containing group descriptions.
        EligibilityCriteria (pd.Series): Series containing eligibility criteria.
        smiles (pd.Series): Series containing SMILES strings.
        labels (np.ndarray): Array containing the labels.
        tokenizer_group_desc (PreTrainedTokenizer): Tokenizer for group descriptions.
        tokenizer_eligibility (PreTrainedTokenizer): Tokenizer for eligibility criteria.
        tokenizer_smiles (PreTrainedTokenizer): Tokenizer for SMILES strings.
        text_max_len (int, optional): Max length for text tokenization. Defaults to None.
        smiles_max_len (int, optional): Max length for SMILES tokenization. Defaults to None.

    Args:
        dataframe (pd.DataFrame): The dataframe containing all the necessary data.
        tokenizer_group_desc (PreTrainedTokenizer): Tokenizer for group descriptions.
        tokenizer_eligibility (PreTrainedTokenizer): Tokenizer for eligibility criteria.
        tokenizer_smiles (PreTrainedTokenizer): Tokenizer for SMILES strings.
        text_max_len (int, optional): Max length for text tokenization. Defaults to None.
        smiles_max_len (int, optional): Max length for SMILES tokenization. Defaults to None.
    """

    def __init__(self,
                 dataframe: pd.DataFrame,
                 tokenizer_group_desc: PreTrainedTokenizer,
                 tokenizer_eligibility: PreTrainedTokenizer,
                 tokenizer_smiles: PreTrainedTokenizer,
                 text_max_len: int = None,
                 smiles_max_len: int = None):

        self.data = dataframe
        self.GroupDescription = dataframe.group_description
        self.EligibilityCriteria = dataframe.eligibility_criteria
        self.smiles = dataframe.smiles
        self.labels = dataframe.drop(
            columns=["group_description", "eligibility_criteria", "smiles"]).values
        self.tokenizer_group_desc = tokenizer_group_desc
        self.tokenizer_eligibility = tokenizer_eligibility
        self.tokenizer_smiles = tokenizer_smiles
        self.text_max_len = text_max_len
        self.smiles_max_len = smiles_max_len

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.GroupDescription)

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a single sample from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing tokenized inputs, attention masks for each data type, and labels.
        """

        # Retrieve data for the specified index
        group_desc = str(self.GroupDescription[index])
        eligibility = str(self.EligibilityCriteria[index])
        smiles = str(self.smiles[index])

        # Tokenize the retrieved data
        group_desc_tokens = self.tokenizer_group_desc.encode_plus(
            group_desc, add_special_tokens=True, truncation=False)
        eligibility_tokens = self.tokenizer_eligibility.encode_plus(
            eligibility, add_special_tokens=True, truncation=False)
        smiles_tokens = self.tokenizer_smiles.encode_plus(
            smiles, add_special_tokens=True, truncation=False)

        # Extract labels for the current sample
        labels = torch.tensor(self.labels[index], dtype=torch.float)

        return {
            'group_desc_input_ids': group_desc_tokens['input_ids'],
            'group_desc_attention_mask': group_desc_tokens['attention_mask'],
            'eligibility_input_ids': eligibility_tokens['input_ids'],
            'eligibility_attention_mask': eligibility_tokens['attention_mask'],
            'smiles_input_ids': smiles_tokens['input_ids'],
            'smiles_attention_mask': smiles_tokens['attention_mask'],
            'labels': labels,
            'text_max_len': self.text_max_len,
            'smiles_max_len': self.smiles_max_len
        }


def custom_collate_fn(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for data loading.

    This function processes a batch of data by truncating and padding the sequences to create 
    uniform lengths across the batch. It handles sequences for group descriptions, eligibility 
    criteria, and SMILES strings, along with their respective attention masks.

    Args:
        batch (List[Dict[str, List[int]]]): A batch of data, where each item is a dictionary 
        containing tokenized information and labels.

    Returns:
        Dict[str, torch.Tensor]: A dictionary with keys corresponding to input IDs, attention masks, 
        and labels, each mapped to a tensor of appropriate shape.
    """

    # Truncate sequences if their lengths exceed the maximum allowed length
    for item in batch:
        if item['text_max_len']:
            item['group_desc_input_ids'] = item['group_desc_input_ids'][:item['text_max_len']]
            item['group_desc_attention_mask'] = item['group_desc_attention_mask'][:item['text_max_len']]
            item['eligibility_input_ids'] = item['eligibility_input_ids'][:item['text_max_len']]
            item['eligibility_attention_mask'] = item['eligibility_attention_mask'][:item['text_max_len']]

        if item['smiles_max_len']:
            item['smiles_input_ids'] = item['smiles_input_ids'][:item['smiles_max_len']]
            item['smiles_attention_mask'] = item['smiles_attention_mask'][:item['smiles_max_len']]

    # Pad the sequences in the batch to ensure uniform length
    group_desc_input_ids = pad_sequence(
        [torch.tensor(item['group_desc_input_ids']) for item in batch], batch_first=True)
    group_desc_attention_mask = pad_sequence([torch.tensor(
        item['group_desc_attention_mask']) for item in batch], batch_first=True)
    eligibility_input_ids = pad_sequence(
        [torch.tensor(item['eligibility_input_ids']) for item in batch], batch_first=True)
    eligibility_attention_mask = pad_sequence([torch.tensor(
        item['eligibility_attention_mask']) for item in batch], batch_first=True)
    smiles_input_ids = pad_sequence(
        [torch.tensor(item['smiles_input_ids']) for item in batch], batch_first=True)
    smiles_attention_mask = pad_sequence(
        [torch.tensor(item['smiles_attention_mask']) for item in batch], batch_first=True)

    # Stack all labels into a single tensor
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'group_desc_input_ids': group_desc_input_ids,
        'group_desc_attention_mask': group_desc_attention_mask,
        'eligibility_input_ids': eligibility_input_ids,
        'eligibility_attention_mask': eligibility_attention_mask,
        'smiles_input_ids': smiles_input_ids,
        'smiles_attention_mask': smiles_attention_mask,
        'labels': labels
    }


def calculate_evaluation_metrics(predictions: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float, float, List[float], List[float], List[float], List[float]]:
    """
    Calculates various evaluation metrics for multi-label classification.

    This function computes the overall and per-label evaluation metrics including F1-scores, 
    Positive Predictive Value (PPV), Negative Predictive Value (NPV), and Recall.

    Args:
        predictions (np.ndarray): An array of predicted labels.
        labels (np.ndarray): An array of actual labels.

    Returns:
        Tuple[float, float, float, float, List[float], List[float], List[float], List[float]]: A tuple containing:
        - Overall micro F1-score (float)
        - Overall Positive Predictive Value (PPV) (float)
        - Overall Negative Predictive Value (NPV) (float)
        - Overall Recall (float)
        - List of per-label F1-scores (List[float])
        - List of per-label Positive Predictive Values (PPVs) (List[float])
        - List of per-label Negative Predictive Values (NPVs) (List[float])
        - List of per-label Recalls (List[float])
    """
    C = predictions.shape[1]  # Number of classes

    PPVs, NPVs, F1s, Recalls = [], [], [], []

    # Initialize overall counts
    total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0

    # Calculate metrics for each class
    for i in range(C):
        TP = np.sum((predictions[:, i] == 1) & (labels[:, i] == 1))
        FP = np.sum((predictions[:, i] == 1) & (labels[:, i] == 0))
        TN = np.sum((predictions[:, i] == 0) & (labels[:, i] == 0))
        FN = np.sum((predictions[:, i] == 0) & (labels[:, i] == 1))

        total_TP += TP
        total_FP += FP
        total_TN += TN
        total_FN += FN

        PPV = TP / (TP + FP) if TP + FP > 0 else 0
        NPV = TN / (TN + FN) if TN + FN > 0 else 0
        Recall = TP / (TP + FN) if TP + FN > 0 else 0

        PPVs.append(PPV)
        NPVs.append(NPV)
        Recalls.append(Recall)

        f1_label = f1_score(labels[:, i], predictions[:, i], average="binary")
        F1s.append(f1_label)

    # Calculate overall metrics
    overall_PPV = total_TP / \
        (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    overall_NPV = total_TN / \
        (total_TN + total_FN) if total_TN + total_FN > 0 else 0
    overall_Recall = total_TP / \
        (total_TP + total_FN) if total_TP + total_FN > 0 else 0

    f1_micro = f1_score(labels, predictions, average="micro")

    return f1_micro, overall_PPV, overall_NPV, overall_Recall, F1s, PPVs, NPVs, Recalls


def compute_baseline_test_performance(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """
    Computes the baseline test performance using the majority class from the training set.

    This function calculates the baseline performance of a classifier on the test set by
    assuming the majority class from the training set as the prediction for all test instances.

    Args:
        train_df (pd.DataFrame): The training dataset containing labels.
        test_df (pd.DataFrame): The test dataset containing labels.

    Returns:
        float: The micro F1 score of the baseline classifier.
    """
    # Extract actual labels from the test set
    actuals = test_df.drop(
        columns=["group_description", "eligibility_criteria", "smiles"]).values

    # Calculate the majority class from the training set
    train_df_label = train_df.drop(
        columns=["group_description", "eligibility_criteria", "smiles"])
    majority_classes = (train_df_label.mean() > 0.5).astype(int).values

    # Create predictions using majority classes
    predictions_majority_class = np.tile(
        majority_classes, (actuals.shape[0], 1))

    # Compute the micro F1 score
    f1_score_majority_class = f1_score(
        actuals, predictions_majority_class, average='micro')

    return f1_score_majority_class


def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer,
                    loss_fn: _Loss, device: torch.device, epoch: int, num_epochs: int) -> float:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to be trained.
        dataloader (DataLoader): Dataloader for the training data.
        optimizer (Optimizer): Optimizer for training.
        loss_fn (_Loss): Loss function.
        device (torch.device): Device to run the training (e.g., 'cuda' or 'cpu').
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(
        dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()

        # Prepare data and perform a forward pass
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        outputs = model(**inputs)
        labels = batch['labels'].to(device)
        loss = loss_fn(outputs, labels)

        # Perform a backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(
            {'Train Loss': total_loss / (progress_bar.n + 1)})

    return total_loss / len(dataloader)


def validate(model: nn.Module, dataloader: DataLoader, device: torch.device, epoch: int, num_epochs: int) -> Tuple[float, float, float, float, List[float], List[float], List[float], List[float]]:
    """
    Validate the model on a given dataset.

    Args:
        model (nn.Module): The model to be evaluated.
        dataloader (DataLoader): Dataloader for the validation data.
        device (torch.device): Device to run the validation (e.g., 'cuda' or 'cpu').
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.

    Returns:
        Tuple[float, float, float, float, List[float], List[float], List[float], List[float]]: 
        Evaluation metrics including overall and per-label F1-score, PPV, NPV, and Recall.
    """
    model.eval()
    predictions, actuals = [], []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            inputs = {k: v.to(device)
                      for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)

            logits = torch.sigmoid(outputs).cpu().numpy()
            labels = labels.cpu().numpy()

            predictions.append(logits > 0.5)
            actuals.append(labels)

    return calculate_evaluation_metrics(np.vstack(predictions), np.vstack(actuals))


def save_model_if_best_performance(current_perf: float, best_perf: float, model: nn.Module, path: str) -> float:
    """
    Save the model if the current performance is better than the best recorded performance.

    This function checks if the current performance metric (e.g., F1 score) of the model is greater 
    than the best performance metric recorded so far. If it is, the model's state dictionary is saved 
    to the specified path. This function is typically used during the training loop to save the best 
    model.

    Args:
        current_perf (float): The current performance metric of the model.
        best_perf (float): The best performance metric recorded so far.
        model (Module): The PyTorch model to be saved.
        path (str): File path where the model's state dictionary should be saved.

    Returns:
        float: The updated best performance metric. This will be either the current performance 
        (if it was better) or the previous best performance.
    """
    if current_perf > best_perf:
        torch.save(model.state_dict(), path)
        return current_perf
    else:
        return best_perf

def calculate_full_evaluation_metrics(predictions: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float, float, float, float, List[float], List[float], List[float], List[float], List[float], List[float]]:
    C = predictions.shape[1]  # Number of classes

    PPVs, NPVs, F1s, Recalls, Accuracies, BalancedAccuracies = [], [], [], [], [], []

    # Initialize overall counts
    total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0

    # Calculate metrics for each class
    for i in range(C):
        TP = np.sum((predictions[:, i] == 1) & (labels[:, i] == 1))
        FP = np.sum((predictions[:, i] == 1) & (labels[:, i] == 0))
        TN = np.sum((predictions[:, i] == 0) & (labels[:, i] == 0))
        FN = np.sum((predictions[:, i] == 0) & (labels[:, i] == 1))

        total_TP += TP
        total_FP += FP
        total_TN += TN
        total_FN += FN

        PPV = TP / (TP + FP) if TP + FP > 0 else 0
        NPV = TN / (TN + FN) if TN + FN > 0 else 0
        Recall = TP / (TP + FN) if TP + FN > 0 else 0
        Accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0

        # Calculating balanced accuracy for each label
        label_true = labels[:, i]
        label_pred = predictions[:, i]
        BalancedAccuracy = balanced_accuracy_score(label_true, label_pred)

        PPVs.append(PPV)
        NPVs.append(NPV)
        Recalls.append(Recall)
        Accuracies.append(Accuracy)
        BalancedAccuracies.append(BalancedAccuracy)

        f1_label = f1_score(label_true, label_pred, average="binary")
        F1s.append(f1_label)

    # Calculate overall metrics
    overall_PPV = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    overall_NPV = total_TN / (total_TN + total_FN) if total_TN + total_FN > 0 else 0
    overall_Recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
    overall_Accuracy = (total_TP + total_TN) / (total_TP + total_FP + total_TN + total_FN) if (total_TP + total_FP + total_TN + total_FN) > 0 else 0
    overall_BalancedAccuracy = balanced_accuracy_score(labels.ravel(), predictions.ravel())

    f1_micro = f1_score(labels, predictions, average="micro")

    return f1_micro, overall_PPV, overall_NPV, overall_Recall, overall_Accuracy, overall_BalancedAccuracy, F1s, PPVs, NPVs, Recalls, Accuracies, BalancedAccuracies


def validate_full(model: nn.Module, dataloader: DataLoader, device: torch.device, epoch: int, num_epochs: int) -> Tuple[float, float, float, float, List[float], List[float], List[float], List[float]]:
    """
    Validate the model on a given dataset.

    Args:
        model (nn.Module): The model to be evaluated.
        dataloader (DataLoader): Dataloader for the validation data.
        device (torch.device): Device to run the validation (e.g., 'cuda' or 'cpu').
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.

    Returns:
        Tuple[float, float, float, float, List[float], List[float], List[float], List[float]]: 
        Evaluation metrics including overall and per-label F1-score, PPV, NPV, and Recall.
    """
    model.eval()
    predictions, actuals = [], []
    progress_bar = tqdm(
        dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            inputs = {k: v.to(device)
                      for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)

            logits = torch.sigmoid(outputs).cpu().numpy()
            labels = labels.cpu().numpy()

            predictions.append(logits > 0.5)
            actuals.append(labels)

    return calculate_full_evaluation_metrics(np.vstack(predictions), np.vstack(actuals))