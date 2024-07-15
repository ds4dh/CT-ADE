# CT-ADE
CT-ADE: An Evaluation Benchmark for Adverse Drug Event Prediction from Clinical Trial Results

## Citation

No citation is available yet.

## Developed with

- Operating System: Ubuntu 22.04.3 LTS
    - Kernel: Linux 4.18.0-513.18.1.el8_9.x86_64
    - Architecture: x86_64
- Python:
    - 3.10.12

## Prerequisites

1. Set up your environment and install the necessary Python libraries as specified in `requirements.txt`. Note that you will need to install the development versions of certain libraries from their respective Git repositories.
2. Place your unzipped MedDRA files in the directory `./data/MedDRA_25_0_English` and your DrugBank XML database in the directory `./data/drugbank`.

Ensure you clone and install the following libraries directly from their Git repositories for the development versions:

- [`transformers`](https://github.com/huggingface/transformers)
- [`trl`](https://github.com/huggingface/trl)

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
│   ├── chembl_approved
│   │   └── empty.null
│   ├── chembl_usan
│   │   └── empty.null
│   ├── clinicaltrials_gov
│   │   └── empty.null
│   ├── drugbank
│   │   └── empty.null
│   └── pubchem
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

## Download Publically Available CT-ADE-SOC and CT-ADE-PT Versions

You can download the publicly available CT-ADE-SOC and CT-ADE-PT versions from HuggingFace. These datasets contain standardized annotations from ClinicalTrials.gov:

- [`CT-ADE-SOC`](https://huggingface.co/datasets/anthonyyazdaniml/CT-ADE-SOC)
- [`CT-ADE-PT`](https://huggingface.co/datasets/anthonyyazdaniml/CT-ADE-PT)

The above datasets are identical to the SOC and PT versions you will produce in the `Typical Pipeline from Checkpoint` section.

## Typical Pipeline from Checkpoint

Follow this procedure if you aim to recreate the dataset detailed in our paper for all levels (SOC, HLGT, HLT, and PT).

### 1. Place your licensed data
Place your unzipped MedDRA files in the directory `./data/MedDRA_25_0_English` and your DrugBank XML database in the directory `./data/drugbank`.

### 2. Download checkpoint from HuggingFace
Download [`chembl_approved, chembl_usan, clinicaltrials_gov, pubchem`](https://huggingface.co/datasets/anthonyyazdaniml/CTADE_v1_initial_release_checkpoint) files and place them accordingly.

### 3. Extract DrugBank DBID Details

Extract drug details from the DrugBank database.

```bash
python c0_extract_drugbank_dbid_details.py
```

### 4. Create Unified Chemical Database

Create a unified database combining information from PubChem, DrugBank, and ChEMBL.

```bash
python f0_create_unified_chemical_database.py
```

### 5. Create Raw CT-ADE Dataset

Generate the raw CT-ADE dataset from the processed clinical trials data.

```bash
python g0_create_ct_ade_raw.py
```

### 6. Create MedDRA Annotations

Annotate the CT-ADE dataset with MedDRA terms.

```bash
python g1_create_ct_ade_meddra.py
```

### 7. Create Classification Datasets

Generate the final classification datasets for modeling.

```bash
python g2_create_ct_ade_classification_datasets.py
```

## Training Models

### Discriminative Models (DLLMs)

Navigate to the `modeling/DLLMs` directory and run the training scripts with the desired configuration.

```bash
cd modeling/DLLMs
```

For single-GPU training, use this command:

```bash
export CUDA_VISIBLE_DEVICES="0"; \
export MIXED_PRECISION="bf16"; \
FIRST_GPU=$(echo $CUDA_VISIBLE_DEVICES | cut -d ',' -f 1); \
BASE_PORT=29500; \
PORT=$(( $BASE_PORT + $FIRST_GPU )); \
accelerate launch \
--mixed_precision=$MIXED_PRECISION \
--num_processes=$(( $(echo $CUDA_VISIBLE_DEVICES | grep -o "," | wc -l) + 1 )) \
--num_machines=1 \
--dynamo_backend=no \
--main_process_port=$PORT \
train.py
```

For multi-GPU training, use this command:

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"; \
export MIXED_PRECISION="bf16"; \
FIRST_GPU=$(echo $CUDA_VISIBLE_DEVICES | cut -d ',' -f 1); \
BASE_PORT=29500; \
PORT=$(( $BASE_PORT + $FIRST_GPU )); \
accelerate launch \
--mixed_precision=$MIXED_PRECISION \
--num_processes=$(( $(echo $CUDA_VISIBLE_DEVICES | grep -o "," | wc -l) + 1 )) \
--num_machines=1 \
--dynamo_backend=no \
--main_process_port=$PORT \
train.py
```

### Generative Models (GLLMs)

Navigate to the `modeling/GLLMs` directory and run the training scripts for different configurations.

```bash
cd modeling/GLLMs
```

Example configurations for LLama3, OpenBioLLM, and Meditron are provided in the folder. You can copy the desired configuration into `config.py` and adjust it to your convenience. Next, you can execute the following for the SGE configuration:

```bash
python train_SGE.py
```
