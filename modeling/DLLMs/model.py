import torch
from torch import Tensor
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from typing import List, Tuple, Optional, Dict


class ADEModel(nn.Module):
    def __init__(
        self,
        num_labels: int,
        text_llm_key: str,
        smiles_llm_key: str,
        text_features: List[str],
        smiles_used: bool,
        negative_sampling_ratio: Optional[float] = None,
    ):
        """
        Initializes the ADEModel class which integrates text and SMILES data encoding 
        to perform multi-label classification.

        Parameters:
            num_labels (int): Number of labels excluding the pad token.
            text_llm_key (str): Pre-trained model identifier for the text encoder.
            smiles_llm_key (str): Pre-trained model identifier for the SMILES encoder.
            text_features (List[str]): List of text features to be used.
            smiles_used (bool): Flag indicating if SMILES features should be used.
            negative_sampling_ratio (Optional[float]): Ratio of negative samples to generate
                relative to positive samples for training. If None, no negative sampling is applied.
        """
        super(ADEModel, self).__init__()

        # Initialize text encoder if text features are provided
        self.text_encoder = (
            AutoModel.from_pretrained(text_llm_key) if text_features else None
        )

        # Initialize SMILES encoder if SMILES features are provided
        self.smiles_encoder = (
            AutoModel.from_pretrained(smiles_llm_key) if smiles_used else None
        )

        # Define input size calculation based on the encoder's configuration
        input_size = 0
        if self.text_encoder is not None:
            input_size += self.text_encoder.config.hidden_size * len(text_features)
        if self.smiles_encoder is not None:
            input_size += self.smiles_encoder.config.hidden_size

        # Initialize label embeddings with a padding index
        total_num_labels = (num_labels + 1)  # Add one for "padding token" (in negative sampling)
        self.label_embeddings = nn.Embedding(
            num_embeddings=total_num_labels,
            embedding_dim=input_size,
            padding_idx=0
        )
        self.num_labels = num_labels
        self.negative_sampling_ratio = negative_sampling_ratio
        self.default_label_indices = torch.tensor(range(1, self.num_labels + 1))

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.1)

        # MLP with 1 hidden layers and CELU activation
        self.mlp = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.CELU(),
            nn.Linear(input_size, input_size)
        )

        # Loss function for multilabel classification
        self.loss_fn = nn.BCEWithLogitsLoss(
            reduction="none"
        )  # Apply reduction 'none' for custom handling

    def forward(
        self,
        eligibility_criteria_input_ids: Optional[Tensor] = None,
        eligibility_criteria_attention_mask: Optional[Tensor] = None,
        group_description_input_ids: Optional[Tensor] = None,
        group_description_attention_mask: Optional[Tensor] = None,
        smiles_input_ids: Optional[Tensor] = None,
        smiles_attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Optional[Tensor]]:
        """
        Performs the forward pass of the model, processing inputs through their respective encoders,
        combining features, and computing logits and loss if labels are provided.

        Parameters:
            eligibility_criteria_input_ids (Optional[Tensor]): Input IDs for eligibility criteria text.
            eligibility_criteria_attention_mask (Optional[Tensor]): Attention mask for eligibility criteria text.
            group_description_input_ids (Optional[Tensor]): Input IDs for group description text.
            group_description_attention_mask (Optional[Tensor]): Attention mask for group description text.
            smiles_input_ids (Optional[Tensor]): Input IDs for SMILES features.
            smiles_attention_mask (Optional[Tensor]): Attention mask for SMILES features.
            labels (Optional[Tensor]): True labels for computing loss during training.

        Returns:
            Dict[str, Optional[Tensor]]: A dictionary containing logits and optionally loss if labels are provided.
        """
        # Initialization
        outputs = []

        # Process text and SMILES features through their respective encoders
        if self.text_encoder is not None:

            if eligibility_criteria_input_ids is not None:
                features = self.encoder_forward(
                    self.text_encoder,
                    eligibility_criteria_input_ids,
                    eligibility_criteria_attention_mask,
                )
                outputs.append(features)

            if group_description_input_ids is not None:
                features = self.encoder_forward(
                    self.text_encoder,
                    group_description_input_ids,
                    group_description_attention_mask,
                )
                outputs.append(features)

        if self.smiles_encoder is not None:
            features = self.encoder_forward(
                self.smiles_encoder, smiles_input_ids, smiles_attention_mask
            )
            outputs.append(features)

        # Combine features from all modalities
        combined_features = self.dropout(torch.cat(outputs, dim=-1))

        # Apply the MLP
        combined_features = self.mlp(combined_features)

        # Negative sampling handling
        label_indices, do_sampling = self.get_label_indices(labels=labels, device=combined_features.device)

        # Fetch selected label embeddings based on the indices
        selected_label_embeddings = self.label_embeddings(label_indices)

        # Compute the logits using matrix multiplication
        if do_sampling:
            logits = torch.einsum(
                "bi,bji->bj",
                combined_features,
                selected_label_embeddings
            )
            loss = None
            if labels is not None:
                selected_labels = torch.gather(
                    labels, 1,
                    (label_indices - 1).clamp(min=0)
                )
                losses = self.loss_fn(logits, selected_labels).flatten()
                label_mask = (label_indices != 0).flatten()
                loss = losses[label_mask].mean()
        else:
            logits = torch.matmul(
                combined_features,
                selected_label_embeddings.transpose(-1, -2)
            )
            loss = None
            if labels is not None:
                loss = self.loss_fn(logits, labels).mean()

        return {"logits": (None if self.training else logits), "loss": loss}

    def get_label_indices(self, labels: Tensor, device: torch.device) -> Tuple[Tensor, bool]:
        """
        Determines label indices for sampling or returns default indices if negative sampling is not used.

        Parameters:
            labels (Tensor): Tensor containing label information for each batch.

        Returns:
            Tuple[Tensor, bool]: A tuple where the first element is the Tensor of label indices,
                and the second element is a boolean indicating if sampling was performed.
        """
        indices = self.default_label_indices.to(device)
        
        if labels is None:
            return indices, False

        if self.negative_sampling_ratio is not None and self.training:
            batch_size, _ = labels.size()
            indices = []

            for i in range(batch_size):
                positive = labels[i].nonzero(as_tuple=False).squeeze(1)
                negative = (labels[i] == 0).nonzero(as_tuple=False).squeeze(1)

                # Randomly sample negative indices to match the number of positives
                num_neg = max(
                    1, int(positive.size(0) * self.negative_sampling_ratio * 2)
                )
                negative_sample = negative[torch.randperm(negative.size(0))[:num_neg]]
                # Append indices for this sample to the list
                indices.append(
                    torch.cat([positive, negative_sample]) + 1
                )  # "+ 1" to account for "0 = label padding index"
                
            # Concatenate all indices per batch for a flat index tensor
            return (
                nn.utils.rnn.pad_sequence(indices, batch_first=True, padding_value=0),
                True,
            )

        return indices, False

    def encoder_forward(self, encoder: PreTrainedModel, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Processes input through the provided encoder model and extracts relevant features.

        Parameters:
            encoder (PreTrainedModel): The encoder model to process the inputs.
            input_ids (Tensor): The input IDs for the encoder.
            attention_mask (Tensor): The attention mask to specify which parts of the input are relevant.

        Returns:
            Tensor: Extracted features from the encoder.
        """
        input_ids, attention_mask = self._remove_excess_padding(
            input_ids, attention_mask
        )
        features = encoder(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
        return features

    @staticmethod
    def _remove_excess_padding(input_ids: Tensor, attention_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Removes columns in the input tensors that are completely padded (i.e., all zeros across the batch).

        Parameters:
            input_ids (Tensor): Original input IDs.
            attention_mask (Tensor): Original attention mask indicating active regions of the input.

        Returns:
            Tuple[Tensor, Tensor]: Tensors of input IDs and attention masks with excess padding removed.
        """
        """Identify columns that have non-zero elements"""
        non_zero_columns = attention_mask.sum(dim=0) != 0
        return input_ids[:, non_zero_columns], attention_mask[:, non_zero_columns]