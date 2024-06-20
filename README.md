# CT-ADE
CT-ADE: An Evaluation Benchmark for Adverse Drug Event Prediction from Clinical Trial Results

## Developed with

- Operating System: Ubuntu 22.04.3 LTS
    - Kernel: Linux 4.18.0-513.18.1.el8_9.x86_64
    - Architecture: x86_64
- Python:
    - 3.10.12

## Prerequisites

### Installation of Required Python Libraries

1. Set up your environment and install the necessary Python libraries.
2. Put your unzipped MedDRA files in `./data/MedDRA_25_0_English` and your DrugBank xml database in `./data/drugbank`.

## Repository Structure

```plaintext
.
├── a0_download_clinical_trials.py
├── a1_extract_completed_or_terminated_interventional_results_clinical_trials.py
├── a2_extract_and_preprocess_monopharmacy_clinical_trials.py
├── b0_download_pubchem_cids.py
├── b1_download_pubchem_cid_details.py
├── c0_extract_drugbank_dbid_details.py
├── d0_extract_chembl_approved_CHEMBL_details.py
├── data
│   ├── MedDRA_25_0_English
│   │   └── empty.null
│   └── drugbank
│       └── empty.null
├── e0_extract_chembl_usan_CHEMBL_details.py
├── f0_create_unified_chemical_database.py
├── g0_create_ct_ade_raw.py
├── g1_create_ct_ade_meddra.py
├── g2_create_ct_ade_classification_datasets.py
├── modeling
│   ├── DLLMs
│   │   ├── config.py
│   │   ├── custom_metrics.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── utils.py
│   └── GLLMs
│       ├── config-llama3.py
│       ├── config-meditron.py
│       ├── config-openbiollm.py
│       ├── config.py
│       ├── train_S.py
│       ├── train_SG.py
│       └── train_SGE.py
├── requirements.txt
└── src
    └── meddra_graph.py
```

## Typical Pipeline For an Up to Date CT-ADE Dataset

The typical pipeline for generating the CT-ADE dataset and running the models involves several steps. Here follows a step-by-step guide to creating an updated version of CT-ADE. To recreate the original dataset, you can switch to the next section "Typical Pipeline from Checkpoint".

### 1. Download Clinical Trials Data

Download clinical trials data from ClinicalTrials.gov using the `a0_download_clinical_trials.py` script.

```bash
python a0_download_clinical_trials.py
```

### 2. Extract Completed or Terminated Interventional Clinical Trials

Extract only the completed or terminated interventional clinical trials.

```bash
python a1_extract_completed_or_terminated_interventional_results_clinical_trials.py
```

### 3. Extract and Preprocess Monopharmacy Clinical Trials

Filter out and preprocess the monopharmacy clinical trials.

```bash
python a2_extract_and_preprocess_monopharmacy_clinical_trials.py
```

### 4. Download PubChem CIDs

Download PubChem CIDs for the drugs used in the clinical trials.

```bash
python b0_download_pubchem_cids.py
```

### 5. Download PubChem CID Details

Download details for the PubChem CIDs.

```bash
python b1_download_pubchem_cid_details.py
```

### 6. Extract DrugBank DBID Details

Extract drug details from the DrugBank database.

```bash
python c0_extract_drugbank_dbid_details.py
```

### 7. Extract ChEMBL Approved Details

Extract details of approved drugs from the ChEMBL database.

```bash
python d0_extract_chembl_approved_CHEMBL_details.py
```

### 8. Extract ChEMBL USAN Details

Extract details of USAN drugs from the ChEMBL database.

```bash
python e0_extract_chembl_usan_CHEMBL_details.py
```

### 9. Create Unified Chemical Database

Create a unified database combining information from PubChem, DrugBank, and ChEMBL.

```bash
python f0_create_unified_chemical_database.py
```

### 10. Create Raw CT-ADE Dataset

Generate the raw CT-ADE dataset from the processed clinical trials data.

```bash
python g0_create_ct_ade_raw.py
```

### 11. Create MedDRA Annotations

Annotate the CT-ADE dataset with MedDRA terms.

```bash
python g1_create_ct_ade_meddra.py
```

### 12. Create Classification Datasets

Generate the final classification datasets for modeling.

```bash
python g2_create_ct_ade_classification_datasets.py
```

### 13. Training Models

#### Discriminative Models (DLLMs)

Navigate to the `modeling/DLLMs` directory and run the training scripts with the desired configuration.

```bash
cd modeling/DLLMs
...
```

#### Generative Models (GLLMs)

Navigate to the `modeling/GLLMs` directory and run the training scripts for different configurations.

For example, to train a model using SMILES only:

```bash
cd modeling/GLLMs
...
```

## Typical Pipeline from Checkpoint

If you want to recreate the original dataset starting from a checkpoint, follow these steps. Ensure you have the intermediate data files saved...

## Citation

No citation is available yet.
