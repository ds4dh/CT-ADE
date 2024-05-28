from pathlib import Path
import json
import pandas as pd

def extract_chembl_data(csv_file_path: str) -> dict:
    """
    Extract ChEMBL drug data from a CSV file.

    Args:
        csv_file_path (str): The path of the CSV file to parse.

    Returns:
        dict: Dictionary of drugs where each drug is mapped to its properties.
    """
    # Load the CSV file
    chembl_data = pd.read_csv(csv_file_path, delimiter=';')
    
    # Dictionary to store drug information
    chembl_info_dict = {}
    
    # Process each row in the dataframe
    for _, row in chembl_data.iterrows():
        chembl_id = row['Parent Molecule']
        smiles = row['Smiles']
        if smiles and pd.notna(row['Smiles']):
            drug_info = {
                'title': row['Name'] if (row['Name'] and pd.notna(row['Name'])) else None,
                'synonyms': row['Synonyms'].split('|') if (row['Synonyms'] and pd.notna(row['Synonyms'])) else None,
                'smiles': smiles,
                'atc_code': row['ATC Codes'] if (row['ATC Codes'] and pd.notna(row['ATC Codes'])) else None,
            }
            
            chembl_info_dict[f"{chembl_id}_usan"] = drug_info

    return chembl_info_dict

def main():
    csv_file_path = "./data/chembl_usan/chembl_usan_drugs.csv"
    chembl_info = extract_chembl_data(csv_file_path)

    output_path = Path("./data/chembl_usan/chembl_usan_details.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(chembl_info, f, indent=4)

if __name__ == "__main__":
    main()