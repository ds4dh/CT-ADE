# CTADE
CTADE: An Evaluation Benchmark for Adverse Drug Event Prediction from Clinical Trial Protocols

----------------
# Developed with

- Operating System: Ubuntu 22.04.3 LTS
    - Kernel: Linux 4.18.0-513.18.1.el8_9.x86_64
    - Architecture: x86_64
- Python:
    - 3.10.12

----------------

## Prerequisites

### Installation of Required Python Libraries

Set up your environment and install the necessary Python libraries with the following commands:

```bash
# Create a new Conda environment.
conda create -n CTADE python=3.10.12

# Activate the environment.
conda activate CTADE

# Navigate to the project root directory.
cd <path_to_project_root>

# Install the required libraries.
pip install -r requirements.txt

# Follow the steps in the "Typical Pipeline" section for further instructions.

# When finished, deactivate the Conda environment.
conda deactivate

# Optionally, remove the Conda environment if no longer needed.
conda env remove --name CTADE
```

----------------
## Repository Structure

```plaintext
├── data (Place clinical trial data here)
│
├── drugbank_data (Place the DrugBank XML database here)
│
├── models
│   └── smiles (Contains metric tables for models; model weights are not included due to size)
│
├── notebooks (Jupyter notebooks for EDA, visualization, etc.)
│
├── src (Source code for utilities and functions)
│
├── 0a_download_data.py (Downloads the up-to-date clinical trial database)
│
├── 0b_extract_drugbank.py (Extracts data from the DrugBank database)
│
├── 1_extract_completed_interventional_results_ades.py (Extracts relevant clinical trials)
│
├── 2_preprocess_data.py (Create the CTADE final dataset; ready for model training)
│
├── 3_train.py (Trains the ADE prediction model)
│
├── 4_compute_integrated_gradients.py (Computes Integrated Gradients for model interpretability)
│
├── LICENSE (Project license)
│
├── README.md (This file)
│
└── requirements.txt (Required dependencies)
```

----------------
## Typical Pipeline

### Option 1: Using Data from HuggingFace

Skip the dataset creation step and train your model using our dataset available on [HuggingFace](https://huggingface.co/datasets/anthonyyazdaniml/CTADE).

Download the splits and organize them in your project folder as follows:

```plaintext
├── data
│   └── classification
│       └── smiles
│           ├── train_base
│           │   ├── train.csv (rename train_base.csv to train.csv)
│           │   ├── val.csv (same for both base and augmented scenarios)
│           │   └── test.csv (same for both base and augmented scenarios)
│           └── train_augmented
│               ├── train.csv (rename train_augmented.csv to train.csv)
│               ├── val.csv (same for both base and augmented scenarios)
│               └── test.csv (same for both base and augmented scenarios)
```

If you opt for HuggingFace data, proceed directly to Step 3.

### Option 2: Creating Data from Scratch

#### Clinical Trials (CTs)

For the most up-to-date CTs, execute `0a_download_data.py`.

#### DrugBank

After obtaining access to [DrugBank](https://go.drugbank.com/), place the XML database in `./drugbank_data/`.

### Step 0: Parse the DrugBank Database

```bash
python 0b_extract_drugbank.py
```

### Step 1: Parse the Clinical Trials

Keep only the eligible trials:

```bash
python 1_extract_completed_interventional_results_ades.py
```

### Step 2: Pre-process the Data

Prepare the data for model training:

```bash
python 2_preprocess_data.py
```

This step creates two folders with data splits inside: `./data/classification/smiles/train_base` and `./data/classification/smiles/train_augmented`.

### Step 3: Train the Model

Train your desired model:

```bash
python 3_train.py
```

Ensure you adjust the parameters at the beginning of the script as needed.

### Step 4: Compute Integrated Gradients

For model interpretability, compute integrated gradients:

```bash
python 4_compute_integrated_gradients.py
```

Adjust the parameters at the beginning of the script accordingly.
