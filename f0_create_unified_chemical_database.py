import json
from typing import Optional, Dict, Any, Tuple, List
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
from tqdm.auto import tqdm
from copy import deepcopy
from pathlib import Path


def read_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Reads a JSON file and returns its content as a Python dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        Optional[Dict[str, Any]]: The content of the JSON file as a dictionary,
        or None if the file is not found or cannot be decoded.
    """
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} could not be decoded.")
    return None


def canonical_smiles(smiles: str) -> Optional[str]:
    """
    Generates a canonical SMILES string from the given SMILES input.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)  # Disable auto-sanitization
    if mol:
        try:
            Chem.SanitizeMol(mol)  # Manual sanitization to control error handling
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            logging.warning(f"Error processing SMILES '{smiles}': {e}")
    else:
        logging.warning(f"Failed to parse SMILES '{smiles}'")
    return None


def canonicalize_and_update_synonyms(
    db: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Canonicalizes SMILES strings in the chemical database and updates the synonyms list
    by appending the title as a synonym.

    Args:
        db: A dictionary with compound identifiers as keys and dictionaries as values,
            which include 'smiles' and 'title' keys, and possibly a 'synonyms' key.

    Returns:
        A dictionary containing entries with successfully canonicalized SMILES strings.
        Each entry's 'synonyms' list is updated to include the compound's title.

    Note:
        Entries with non-canonicalizable SMILES strings are omitted.
        If the 'synonyms' key is missing or None, it is created with the title as the only synonym.
    """
    standardized_db = {}
    for key, value in db.items():
        standardized_smiles = canonical_smiles(value["smiles"])
        if standardized_smiles:
            value["smiles"] = standardized_smiles
            if "synonyms" not in value or value["synonyms"] is None:
                value["synonyms"] = [value["title"]]
            else:
                if value["title"] not in value["synonyms"]:
                    value["synonyms"].append(value["title"])
            standardized_db[key] = value
    return standardized_db


def filter_usan_by_approved(
    usan_db: Dict[str, Dict[str, Any]], approved_db: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Filters the USAN database by removing entries that exist in the approved database.

    Args:
        usan_db: Dictionary of USAN compounds with compound identifiers as keys.
        approved_db: Dictionary of approved compounds with compound identifiers as keys.

    Returns:
        A filtered dictionary of USAN compounds that do not exist in the approved database.
    """
    # Create a set of SMILES from the approved database for fast comparison
    approved_smiles = {details["smiles"] for details in approved_db.values()}

    # Filter USAN entries, keeping only those whose SMILES are not in the approved set
    filtered_usan_db = {
        key: value
        for key, value in usan_db.items()
        if value["smiles"] not in approved_smiles
    }

    return filtered_usan_db


class UnionFind:
    """
    Class implementing the Union-Find data structure.
    """

    def __init__(self) -> None:
        """
        Initializes the UnionFind object.
        """
        self.parent: Dict[Any, Any] = {}

    def find(self, item: Any) -> Any:
        """
        Finds the root of the set containing the given item.

        Args:
            item (Any): The item whose root needs to be found.

        Returns:
            Any: The root of the set containing the given item.
        """
        if item not in self.parent:
            self.parent[item] = item
        elif self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, set1: Any, set2: Any) -> None:
        """
        Unions two sets represented by set1 and set2.

        Args:
            set1 (Any): Representative element of the first set.
            set2 (Any): Representative element of the second set.

        Returns:
            None
        """
        root1 = self.find(set1)
        root2 = self.find(set2)
        if root1 != root2:
            self.parent[root2] = root1


def create_unified_database(
    *dbs: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Combines multiple databases into a unified database, grouping compounds based on their SMILES and titles.

    Args:
        *dbs (Dict[str, Dict[str, Any]]): Variable number of dictionaries representing databases,
            each with compound identifiers as keys and dictionaries as values.

    Returns:
        Dict[str, Dict[str, Any]]: A unified database with compound identifiers as keys
            and dictionaries as values, where each dictionary represents a compound.
    """
    uf = UnionFind()
    key_to_details = {}
    smiles_to_key = {}
    title_to_key = {}

    # First pass: Assign each compound a group based on SMILES and title.
    for db in dbs:
        for key, details_ in db.items():
            details = deepcopy(details_)
            details.update({"key": key})  # Store the original key for reference
            smiles = details["smiles"]
            title = details["title"].lower().strip()

            # Ensure each compound has a unique entry in key_to_details
            key_to_details[key] = details

            # Union by SMILES
            if smiles not in smiles_to_key:
                smiles_to_key[smiles] = key
            uf.union(key, smiles_to_key[smiles])

            # Union by title
            if title not in title_to_key:
                title_to_key[title] = key
            uf.union(key, title_to_key[title])

    # Second pass: Aggregate compounds by the root of their set.
    grouped = {}
    for key in key_to_details:
        root = uf.find(key)
        if root not in grouped:
            grouped[root] = []
        grouped[root].append(key_to_details[key])

    # Format the final output
    final_db = {}
    counter = 1
    for entries in grouped.values():
        group_id = f"ID_{counter}"
        final_db[group_id] = {
            entry["key"]: entry for entry in entries
        }  # Use comprehension to collect all entries
        counter += 1

    return final_db


def clean_duplicate_synonyms(
    unified_db: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Identifies synonyms that are associated with more than one unique SMILES string and removes them
    based on the least database agreement for those synonyms, case-insensitively. Removes synonyms from all entries
    if their association frequency across groups is equal. The input database is not modified in place.

    Args:
        unified_db: The unified database with entries grouped by canonical SMILES strings.

    Returns:
        Dict[str, Dict[str, Dict[str, Any]]]: A new version of the unified database with necessary modifications.
    """
    # Create a deep copy of the unified database to modify
    new_unified_db = deepcopy(unified_db)
    synonym_usage = {}

    # Track the usage of each synonym across different SMILES strings and count occurrences
    for group_id, compounds in new_unified_db.items():
        for comp_id, details in compounds.items():
            synonyms = details.get("synonyms", [])
            for synonym in synonyms:
                synonym_lower = synonym.lower().strip()
                if synonym_lower not in synonym_usage:
                    synonym_usage[synonym_lower] = {}
                if group_id not in synonym_usage[synonym_lower]:
                    synonym_usage[synonym_lower][group_id] = []
                synonym_usage[synonym_lower][group_id].append(comp_id)

    # Determine which synonyms to remove from each group
    for synonym, groups in synonym_usage.items():
        if len(groups) > 1:
            # Sort groups by the number of associated database IDs (ascending)
            sorted_groups = sorted(groups.items(), key=lambda item: len(item[1]))
            # Find if the smallest and largest groups are equal in size
            if len(sorted_groups[0][1]) == len(sorted_groups[-1][1]):
                groups_to_remove = (
                    groups.keys()
                )  # Remove from all groups if frequency is equal
            else:
                groups_to_remove = [group[0] for group in sorted_groups[:-1]]

            # Remove the synonym from the determined groups
            for group_id in groups_to_remove:
                compounds = new_unified_db[group_id]
                for comp_id in compounds:
                    original_synonyms = compounds[comp_id]["synonyms"]
                    # Remove the synonym if it matches the current one (case insensitive)
                    compounds[comp_id]["synonyms"] = [
                        syn
                        for syn in original_synonyms
                        if syn.lower().strip() != synonym
                    ]

    return new_unified_db


def add_frequency_ranking(
    unified_db: Dict[str, Dict[str, Dict[str, str]]]
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Add frequency ranking to each entry in the JSON data.

    Args:
        unified_db: The JSON data to process.

    Returns:
        The updated JSON data with frequency ranking.
    """
    json_data = deepcopy(unified_db)
    for group_id, group_data in json_data.items():
        smiles_strings: List[str] = [
            entry_data["smiles"] for entry_data in group_data.values()
        ]
        frequency_ranking: Dict[str, int] = {}
        for smiles in set(smiles_strings):
            frequency: int = smiles_strings.count(smiles)
            frequency_ranking[smiles] = frequency
        sorted_smiles: List[Tuple[str, int]] = sorted(
            frequency_ranking.items(), key=lambda x: (-x[1], -len(x[0]), x[0])
        )
        rankings: Dict[str, int] = {}
        for i, (smiles, _) in enumerate(sorted_smiles):
            rankings[smiles] = i + 1
        for entry_id, entry_data in group_data.items():
            json_data[group_id][entry_id]["smiles_frequency_ranking"] = rankings[
                entry_data["smiles"]
            ]
    return json_data


def create_unique_clean_compounds(
    unified_db: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Processes a unified database of compounds to extract unique, clean compound information, including ATC codes.
    The simplest title is chosen from titles only, based on the highest frequency, shortest length, and alphabetical order for ties.
    Synonyms include a unique list of all titles and synonyms combined.
    """
    db_copy = deepcopy(unified_db)
    unique_clean_compounds = {}

    for group_id, compounds in db_copy.items():
        all_titles = set()
        all_synonyms = set()
        all_atc_code = set()
        selected_smiles = None

        # Aggregate all titles and synonyms separately
        for comp_id, details in compounds.items():
            all_titles.add(details["title"])
            all_synonyms.update(details.get("synonyms", []))
            if "atc_code" in details and details["atc_code"]:
                all_atc_code.update(details["atc_code"].split(" | "))

        # Compute frequency of each title
        title_frequencies = {title: 0 for title in all_titles}
        for title in all_titles:
            for comp_id, details in compounds.items():
                if title == details["title"]:
                    title_frequencies[title] += 1

        # Select the simplest title based on frequency, then length, then alphabetically
        sorted_titles = sorted(
            title_frequencies.items(), key=lambda x: (-x[1], len(x[0]), x[0])
        )
        simplest_title = sorted_titles[0][0]

        # Select the SMILES for the highest frequency ranking (existing logic)
        for comp_id, details in compounds.items():
            if details.get("smiles_frequency_ranking", 0) == 1:
                selected_smiles = details["smiles"]

        # Sort and join ATC codes
        sorted_atc_code = " | ".join(sorted(all_atc_code))

        # Combine all titles and synonyms for a unique list of terms
        combined_terms = all_titles.union(all_synonyms)

        # Organize the processed data into a new dictionary entry
        unique_clean_compounds[group_id] = {
            "title": simplest_title,
            "synonyms": list(combined_terms),
            "atc_code": sorted_atc_code,
            "smiles": selected_smiles,
        }

    return unique_clean_compounds


def save_unified_database(
    unified_db: Dict[str, Dict[str, Dict[str, Any]]], file_path: str
) -> None:
    """
    Saves the unified database to a JSON file.

    Args:
        unified_db: The unified database to be saved.
        file_path: The path to the file where the database should be saved.

    Raises:
        IOError: If the file could not be written.
        JSONEncodeError: If the database contains objects that are not serializable to JSON.
    """
    try:
        with open(file_path, "w") as file:
            json.dump(unified_db, file, indent=4)
        print(f"Database successfully saved to {file_path}")
    except IOError as e:
        print(f"Failed to write to file: {e}")
    except TypeError as e:
        print(f"Error serializing the database to JSON: {e}")


def main() -> None:

    # Load databases
    dbid_details = read_json_file("./data/drugbank/dbid_details.json")
    chembl_approved_details = read_json_file("./data/chembl_approved/chembl_approved_details.json")
    chembl_usan_details = read_json_file("./data/chembl_usan/chembl_usan_details.json")
    cid_details = read_json_file("./data/pubchem/cid_details.json")

    # Standardize SMILES and update synonyms with title in each database
    dbid_details = canonicalize_and_update_synonyms(dbid_details)
    chembl_approved_details = canonicalize_and_update_synonyms(chembl_approved_details)
    chembl_usan_details = canonicalize_and_update_synonyms(chembl_usan_details)
    cid_details = canonicalize_and_update_synonyms(cid_details)

    # Filter USAN entries that are already in the Approved database
    chembl_usan_details = filter_usan_by_approved(
        chembl_usan_details, chembl_approved_details
    )

    # Create unified database
    unified_database = create_unified_database(
        dbid_details, chembl_approved_details, chembl_usan_details, cid_details
    )

    # Clean overlapping synonyms
    unified_database = clean_duplicate_synonyms(unified_database)

    # Calculate frequency ranking for each SMILES within a compound
    unified_database = add_frequency_ranking(unified_database)

    # Keep only one representation per compound
    unified_database = create_unique_clean_compounds(unified_database)

    output_file_path = Path("./data/unified_chemical_database/unified_chemical_database.json")
    output_folder = output_file_path.parent
    output_folder.mkdir(parents=True, exist_ok=True)

    save_unified_database(unified_database, output_file_path)


if __name__ == "__main__":
    main()
