import xml.etree.ElementTree as ET
from typing import Dict, Tuple
from tqdm.auto import tqdm


def extract_drug_data(xml_file: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """
    Extract drug data from Drugbank database xml file.

    Args:
    - xml_file (str): The path of the xml file to parse.

    Returns:
    - Tuple[Dict[str, Dict[str, str]], Dict[str, str]]: Tuple of two dictionaries: drug_dict and aliases.
    drug_dict is a dictionary that maps drug names to a dictionary of drug properties.
    aliases is a dictionary that maps drug names and synonyms to the actual drug name.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = {'db': 'http://www.drugbank.ca'}
    drug_dict = {}
    aliases = {}

    for drug in tqdm(root.findall('./db:drug', ns), leave=True):

        # Extract name
        name = drug.findtext('./db:name', namespaces=ns)
        drug_dict[name] = {}

        # Extract synonymes and brand names to create an alias dictionary
        for synonym in drug.findall('./db:synonyms/db:synonym', ns):
            aliases[synonym.text] = name
        for brand_name in drug.findall('./db:international-brands/db:international-brand/db:name', ns):
            aliases[brand_name.text] = name
        aliases[name] = name

        # Extract drug type
        drug_type = drug.get('type')
        drug_dict[name]["type"] = drug_type.strip().lower() if drug_type else "[NOTYPE]"

        # Extract classification kingdom
        kingdom = drug.findtext(
            './db:classification/db:kingdom', namespaces=ns)
        drug_dict[name]["kingdom"] = kingdom.strip().lower() if kingdom else "[NOKINGDOM]"

        # Extract ATC anatomical main group
        try:
            atc_code_element = drug.findall('./db:atc-codes/db:atc-code', namespaces=ns)[0]
            atc = [atc_code_element.get('code')] + [level.get('code') for level in atc_code_element.findall('./db:level', namespaces=ns)]
            drug_dict[name]["atc_anatomical_main_group"] = next((s.strip().lower() for s in atc if len(s) == 1), "[NOATCM]")
        except IndexError:
            drug_dict[name]["atc_anatomical_main_group"] = "[NOATCM]"

        # Extract SMILES
        smiles = drug.findtext(
            './db:calculated-properties/db:property[db:kind="SMILES"]/db:value', namespaces=ns)
        drug_dict[name]["smiles"] = smiles if smiles else "[NOSMILES]"

        # Extract sequence
        sequence = drug.findtext('./db:sequences/db:sequence', namespaces=ns)
        drug_dict[name]["sequence"] = ''.join(
            sequence.split("\n")[1:]) if sequence else "[NOSEQUENCE]"

    return drug_dict, aliases