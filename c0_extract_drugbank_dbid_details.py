import xml.etree.ElementTree as ET
from typing import Dict
from tqdm.auto import tqdm
import json
from pathlib import Path


def extract_drug_data(xml_file_path: str) -> Dict[str, Dict[str, list]]:
    """
    Extract drug data from Drugbank database XML file.

    Args:
    - xml_file_path (str): The path of the XML file to parse.

    Returns:
    - Dict[str, Dict[str, list]]: Dictionary of drugs where each drug is mapped to its properties.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    ns = {'db': 'http://www.drugbank.ca'}
    dbid_info_dict = {}
    
    for drug in tqdm(root.findall('./db:drug', ns), leave=True):
        dbid_info = {}
    
        # Extract dbid
        dbid = drug.findtext('./db:drugbank-id', namespaces=ns)
        if not dbid:
            continue
    
        # Extract title
        title = drug.findtext('./db:name', namespaces=ns)
        dbid_info['title'] = title or None
    
        # Extract synonyms and brand names, collecting all under the 'synonyms' key
        synonyms = [syn.text for syn in drug.findall('./db:synonyms/db:synonym', ns)]
        brand_names = [brand.text for brand in drug.findall('./db:international-brands/db:international-brand/db:name', ns)]
        all_names = (synonyms + brand_names) if (synonyms + brand_names) else None
        dbid_info['synonyms'] = all_names

        # Extract SMILES
        smiles = drug.findtext('./db:calculated-properties/db:property[db:kind="SMILES"]/db:value', namespaces=ns)
        if not smiles:
            continue
        dbid_info['smiles'] = smiles
    
        # Extract all ATC codes and join them with " | "
        atc_codes = [element.get('code') for element in drug.findall('./db:atc-codes/db:atc-code', namespaces=ns)]
        dbid_info['atc_code'] = " | ".join(atc_codes) if atc_codes else None
    
        dbid_info_dict[dbid] = dbid_info

    return dbid_info_dict


def main():
    xml_file_path = "data/drugbank/full database.xml"
    dbid_info = extract_drug_data(xml_file_path)

    output_path = Path("data/drugbank/dbid_details.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dbid_info, f, indent=4)


if __name__ == "__main__":
    main()