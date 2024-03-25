import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
from captum.attr import LayerIntegratedGradients
import os
import pickle

# Params
device = "cuda:0"
text_model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
smiles_model_name = "DeepChem/ChemBERTa-77M-MLM"
trained_model_path = './models/smiles/train_augmented/all/model.pt'
test_set = pd.read_csv("./data/classification/smiles/train_augmented/test.csv")
output_filename = "IG_test_set_output_clean.pkl"

if __name__ == '__main__':

    class MultilabelModel(nn.Module):
        def __init__(self, text_model_name, smiles_model_name, smiles_tokenizer_len, num_labels):
            super(MultilabelModel, self).__init__()
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            self.smiles_encoder = AutoModel.from_pretrained(smiles_model_name)
            self.smiles_encoder.resize_token_embeddings(smiles_tokenizer_len)
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(
                self.text_encoder.config.hidden_size +
                self.text_encoder.config.hidden_size +
                self.smiles_encoder.config.hidden_size,
                num_labels
            )

        def forward(self, group_desc_input_ids, group_desc_attention_mask, eligibility_input_ids, eligibility_attention_mask, smiles_input_ids, smiles_attention_mask):
            group_desc_output = self.text_encoder(group_desc_input_ids, attention_mask=group_desc_attention_mask)
            eligibility_output = self.text_encoder(eligibility_input_ids, attention_mask=eligibility_attention_mask)
            smiles_output = self.smiles_encoder(smiles_input_ids, attention_mask=smiles_attention_mask)
            concatenated = torch.cat(
                (group_desc_output[0][:, 0], eligibility_output[0][:, 0], smiles_output[0][:, 0]),
                dim=1
            )
            dropped = self.dropout(concatenated)
            logits = self.classifier(dropped)
            return logits


    class NonSharedMultilabelModel(nn.Module):
        def __init__(self,
                     text_model_name,
                     smiles_model_name,
                     smiles_tokenizer_len,
                     num_labels):
            super(NonSharedMultilabelModel, self).__init__()
            self.text_encoder_a = AutoModel.from_pretrained(text_model_name)
            self.text_encoder_b = AutoModel.from_pretrained(text_model_name)
            self.smiles_encoder = AutoModel.from_pretrained(smiles_model_name)
            self.smiles_encoder.resize_token_embeddings(smiles_tokenizer_len)
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(
                self.text_encoder_a.config.hidden_size +
                self.text_encoder_a.config.hidden_size +
                self.smiles_encoder.config.hidden_size,
                num_labels
            )

        def forward(self,
                    group_desc_input_ids,
                    group_desc_attention_mask,
                    eligibility_input_ids,
                    eligibility_attention_mask,
                    smiles_input_ids,
                    smiles_attention_mask):
            group_desc_output = self.text_encoder_a(group_desc_input_ids, attention_mask=group_desc_attention_mask)
            eligibility_output = self.text_encoder_b(eligibility_input_ids, attention_mask=eligibility_attention_mask)
            smiles_output = self.smiles_encoder(smiles_input_ids, attention_mask=smiles_attention_mask)
            concatenated = torch.cat(
                (group_desc_output[0][:, 0], eligibility_output[0][:, 0], smiles_output[0][:, 0]),
                dim=1
            )
            dropped = self.dropout(concatenated)
            logits = self.classifier(dropped)
            return logits

    # Tokenizers
    group_desc_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    eligibility_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    smiles_tokenizer = AutoTokenizer.from_pretrained(smiles_model_name)
    smiles_tokenizer.add_tokens(['[PLACEBO]', '[NOSMILES]'])
    smiles_tokenizer_len = len(smiles_tokenizer)

    # Model
    shared_model = MultilabelModel(text_model_name, smiles_model_name, smiles_tokenizer_len, 27)
    shared_model.load_state_dict(torch.load(trained_model_path))
    model = NonSharedMultilabelModel(text_model_name, smiles_model_name, smiles_tokenizer_len, 27)
    model.text_encoder_a.load_state_dict(shared_model.text_encoder.state_dict())
    model.text_encoder_b.load_state_dict(shared_model.text_encoder.state_dict())
    model.smiles_encoder.load_state_dict(shared_model.smiles_encoder.state_dict())
    model.classifier.load_state_dict(shared_model.classifier.state_dict())

    del shared_model
    model = model.to(device)
    model.eval();

    def create_baseline_ids(input_ids, tokenizer):
        # Get special token IDs
        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        pad_token_id = tokenizer.pad_token_id

        # Create a baseline tensor filled with the pad token ID
        baseline_ids = torch.full_like(input_ids, pad_token_id)

        # Set the first token to [CLS] and the last token to [SEP]
        baseline_ids[:, 0] = cls_token_id
        baseline_ids[torch.arange(baseline_ids.size(0)), input_ids.ne(pad_token_id).sum(dim=1) - 1] = sep_token_id

        return baseline_ids

    def save_with_pickle(data, filename):
        with open(filename, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    max_length = 512
    overall_output = {}
    label_names = list(test_set.iloc()[:, -27:].columns)

    for row_id, row in tqdm(test_set.iterrows(), total=len(test_set)):

        output = {}

        group_id = row.group_id
        eligibility_criteria = row.eligibility_criteria
        group_description = row.group_description
        smiles = row.smiles
        true_labels = list(row.iloc()[-27:])

        eligibility_ = eligibility_tokenizer(
            eligibility_criteria,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        group_desc_ = group_desc_tokenizer(
            group_description,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )
        smiles_ = smiles_tokenizer(
            smiles,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )

        eligibility_input_ids = eligibility_.input_ids.to(device)
        eligibility_attention_mask = eligibility_.attention_mask.to(device)
        baseline_ids_eligibility = create_baseline_ids(
            eligibility_input_ids, eligibility_tokenizer
        )
        baseline_mask_eligibility = torch.ones_like(eligibility_attention_mask)

        group_desc_input_ids = group_desc_.input_ids.to(device)
        group_desc_attention_mask = group_desc_.attention_mask.to(device)
        baseline_ids_group_desc = create_baseline_ids(
            group_desc_input_ids, group_desc_tokenizer
        )
        baseline_mask_group_desc = torch.ones_like(group_desc_attention_mask)

        smiles_input_ids = smiles_.input_ids.to(device)
        smiles_attention_mask = smiles_.attention_mask.to(device)
        baseline_ids_smiles = create_baseline_ids(smiles_input_ids, smiles_tokenizer)
        baseline_mask_smiles = torch.ones_like(smiles_attention_mask)

        # Compute model's prediction
        model_prediction = model(
            group_desc_input_ids,
            group_desc_attention_mask,
            eligibility_input_ids,
            eligibility_attention_mask,
            smiles_input_ids,
            smiles_attention_mask,
        )
        probabilities = torch.sigmoid(model_prediction).cpu().detach().numpy()

        for label, label_name in enumerate(label_names):

            # Forward function for the model focusing on a specific class's logit
            def forward_func(ids, mask, a, b, c, d):
                outputs = model(a, b, ids, mask, c, d)
                return torch.sigmoid(outputs)

            # Define LayerIntegratedGradients using forward_func
            lig = LayerIntegratedGradients(forward_func, model.text_encoder_b.embeddings)

            # Attributions for group description embeddings
            attributions_eligibility, delta_eligibility = lig.attribute(
                internal_batch_size=10,
                inputs=(eligibility_input_ids, eligibility_attention_mask),
                baselines=(baseline_ids_eligibility, baseline_mask_eligibility),
                additional_forward_args=(
                    group_desc_input_ids,
                    group_desc_attention_mask,
                    smiles_input_ids,
                    smiles_attention_mask,
                ),
                target=label,
                return_convergence_delta=True,
            )

            # Forward function for the model focusing on a specific class's logit
            def forward_func(ids, mask, a, b, c, d):
                outputs = model(ids, mask, a, b, c, d)
                return torch.sigmoid(outputs)

            # Define LayerIntegratedGradients using forward_func
            lig = LayerIntegratedGradients(forward_func, model.text_encoder_a.embeddings)

            # Baseline tensors
            baseline_ids = create_baseline_ids(group_desc_input_ids, group_desc_tokenizer)
            baseline_mask = torch.ones_like(group_desc_attention_mask)

            # Attributions for group description embeddings
            attributions_group_desc, delta_group_desc = lig.attribute(
                internal_batch_size=10,
                inputs=(group_desc_input_ids, group_desc_attention_mask),
                baselines=(baseline_ids_group_desc, baseline_mask_group_desc),
                additional_forward_args=(
                    eligibility_input_ids,
                    eligibility_attention_mask,
                    smiles_input_ids,
                    smiles_attention_mask,
                ),
                target=label,
                return_convergence_delta=True,
            )

            # Forward function for the model focusing on a specific class's logit
            def forward_func(ids, mask, a, b, c, d):
                outputs = model(a, b, c, d, ids, mask)
                return torch.sigmoid(outputs)

            # Define LayerIntegratedGradients using forward_func
            lig = LayerIntegratedGradients(forward_func, model.smiles_encoder.embeddings)

            # Attributions for group description embeddings
            attributions_smiles, delta_smiles = lig.attribute(
                internal_batch_size=10,
                inputs=(smiles_input_ids, smiles_attention_mask),
                baselines=(baseline_ids_smiles, baseline_mask_smiles),
                additional_forward_args=(
                    group_desc_input_ids,
                    group_desc_attention_mask,
                    eligibility_input_ids,
                    eligibility_attention_mask,
                ),
                target=label,
                return_convergence_delta=True,
            )

            probability = probabilities[0][label]

            output[label_name] = {
                "true_label": float(true_labels[label]),
                "predicted_prob": float(probability),
                "attributions_eligibility": attributions_eligibility.cpu().detach().numpy(),
                "attributions_group_desc": attributions_group_desc.cpu().detach().numpy(),
                "attributions_smiles": attributions_smiles.cpu().detach().numpy(),
            }

            torch.cuda.empty_cache()

        overall_output[group_id] = output

        save_with_pickle(
            overall_output,
            filename=os.path.join(
                "/".join(trained_model_path.split("/")[:-1]), output_filename
            ),
        )
