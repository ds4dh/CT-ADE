import os
from src.drugbank_helpers import extract_drug_data
import json

# Main path of the current file
main_path: str = os.path.dirname(os.path.realpath(__file__))

# Path of the data to be processed
data_path: str = "drugbank_data/database.xml"

# Path where the processed data will be saved
save_path: str = main_path + "/drugbank_data"

# Extract the drug data from `data_path`
drug_dict, aliases = extract_drug_data(data_path)

# Convert `drug_dict` to JSON and write to file
with open(f"{save_path}/drug_dict.json", "w") as f:
    f.write(json.dumps(drug_dict, indent=4, ensure_ascii=False))

# Convert `aliases` to JSON and write to file
with open(f"{save_path}/aliases.json", "w") as f:
    f.write(json.dumps(aliases, indent=4, ensure_ascii=False))