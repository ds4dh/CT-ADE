import json
from pathlib import Path
from nltk.corpus import stopwords
import regex
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count, Manager
import itertools
from copy import deepcopy
from collections import Counter
import logging
from rdkit import Chem
import pandas as pd
from typing import Optional, Dict, Any, Set, List, Tuple, Iterator, TypeVar, Callable

T = TypeVar("T")
stop_words = set(stopwords.words("english"))
global drug_id_details_global
drug_id_details_global = {}


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


def count_study_groups(trial_data: Dict[str, Any]) -> int:
    """
    Counts the number of study groups in the trial data.

    Args:
        trial_data (Dict[str, Any]): A dictionary containing trial data, with each trial
        having one or more study groups.

    Returns:
        int: The total count of study groups across all trials.
    """
    study_group_counter = 0
    for nctid, trial_value in trial_data.items():
        for study_group in trial_value["study_groups"]:
            study_group_counter += 1
    return study_group_counter


def sanitize_drug_name(drug: str) -> str:
    """
    Sanitizes the drug name by removing units, routes, forms, special characters, and stop words.

    Args:
        drug (str): The original drug name string to be sanitized.

    Returns:
        str: The sanitized drug name.
    """

    # List of common drug units, routes, and forms
    units = ["mg", "ml", "g", "l", "mcg", "ug", "iu", "u", "mmol", "nmol", "pmol", "kg"]
    routes = [
        "oral",
        "injection",
        "intravenous",
        "topical",
        "nasal",
        "subcutaneous",
        "intramuscular",
        "inhalation",
        "rectal",
        "sublingual",
        "transdermal",
        "intradermal",
        "ophthalmic",
        "otic",
        "vaginal",
        "buccal",
        "enteral",
        "parenteral",
        "intravitreal",
    ]
    forms = [
        "tablet",
        "capsule",
        "cream",
        "ointment",
        "gel",
        "solution",
        "suspension",
        "syrup",
        "lozenge",
        "patch",
        "powder",
        "aerosol",
        "emulsion",
        "spray",
        "drop",
        "film",
        "foam",
        "implant",
        "inhaler",
        "insert",
        "lotion",
        "pessary",
        "suppository",
    ]

    # Create regular expression patterns
    unit_pattern = (
        r"\b\d*\.?\d+\s*(?:"
        + "|".join(units)
        + ")(?:\s*[-/]\s*(?:"
        + "|".join(units)
        + "))?"
    )
    percentage_pattern = r"\b\d+(\.\d+)?%"
    route_pattern = r"\b(?:" + "|".join(routes) + r")(?:\s|$)"
    form_pattern = r"\b(?:" + "|".join(forms) + r")(?:\s|$)"
    special_char_pattern = r"\p{S}"  # Matches any symbol

    # Apply regular expressions
    drug = regex.sub(r"\[.*?\]|\(.*?\)", "", drug)
    drug = regex.sub(unit_pattern, "", drug, flags=regex.IGNORECASE)
    drug = regex.sub(percentage_pattern, "", drug)
    drug = regex.sub(route_pattern, "", drug, flags=regex.IGNORECASE)
    drug = regex.sub(form_pattern, "", drug, flags=regex.IGNORECASE)
    drug = regex.sub(special_char_pattern, "", drug)

    drug = " ".join([word for word in drug.split() if word.lower() not in stop_words])

    # Replace multiple white spaces with a single space
    drug = regex.sub(r"\s+", " ", drug)

    drug = drug.strip()

    return drug


def normalize_synonyms(synonyms: Set[str], sanitize: bool = False) -> Set[str]:
    """
    Normalizes a set of drug synonyms by optionally sanitizing them and converting to lowercase.

    Args:
        synonyms (Set[str]): A set of synonyms to normalize.
        sanitize (bool): Whether to sanitize the synonyms. Defaults to False.

    Returns:
        Set[str]: A set of normalized synonyms.
    """
    if sanitize:
        return {sanitize_drug_name(synonym).lower().strip() for synonym in synonyms}
    else:
        return {synonym.lower().strip() for synonym in synonyms}


def find_exact_match(
    candidate_synonyms: Set[str], intervention_synonyms: Set[str]
) -> Optional[str]:
    """
    Identifies the exact matching synonym between two sets of synonyms.

    Args:
        candidate_synonyms (Set[str]): The first set of synonyms.
        intervention_synonyms (Set[str]): The second set of synonyms.

    Returns:
        Optional[str]: The exact matching synonym if found, None otherwise.
    """
    match = candidate_synonyms.intersection(intervention_synonyms)
    if match:
        # Sort the match set to ensure deterministic behavior
        return sorted(match)[0]
    return None


def find_partial_match(
    candidate_synonyms: Set[str], intervention_synonyms: Set[str]
) -> Optional[str]:
    """
    Identifies the first partial matching synonym between two sets of synonyms.

    Args:
        candidate_synonyms (Set[str]): The first set of synonyms.
        intervention_synonyms (Set[str]): The second set of synonyms.

    Returns:
        Optional[str]: The first partial matching synonym if found, None otherwise.
    """
    # Convert sets to sorted lists to maintain a consistent order
    sorted_candidates = sorted(candidate_synonyms)
    sorted_interventions = sorted(intervention_synonyms)
    for candidate in sorted_candidates:
        for intervention in sorted_interventions:
            if intervention in candidate:
                return candidate
    return None


def get_combined_synonyms(data: Dict[str, Any], drug_id: str) -> List[str]:
    """
    Retrieves combined synonyms and the title for a given drug ID.

    Args:
        title_synonym_atc_data (Dict[str, Any]): Dictionary containing titles, synonyms, ATC codes, and SMILES.
        drug_id (str): The drug ID for which synonyms are required.

    Returns:
        List[str]: List of synonyms including the drug title.
    """
    candidate_data = data.get(str(drug_id), {}) or {}
    candidate_synonyms = candidate_data.get("synonyms", []) or []
    candidate_title = candidate_data.get("title")
    if candidate_title:
        candidate_synonyms.append(candidate_title)
    return candidate_synonyms or []


def get_title(data: Dict[str, Any], drug_id: str) -> Optional[str]:
    """
    Retrieves the title for a given drug ID.

    Args:
        title_synonym_atc_data (Dict[str, Any]): Dictionary containing titles, synonyms, ATC codes, and SMILES.
        drug_id (str): The drug ID for which the title is required.

    Returns:
        Optional[str]: The title if available, None otherwise.
    """
    candidate_data = data.get(str(drug_id), {}) or {}

    candidate_title = candidate_data.get("title")
    return candidate_title or None


def get_atc_code(data: Dict[str, Any], drug_id: str) -> Optional[str]:
    """
    Retrieves the ATC code for a given drug ID.

    Args:
        title_synonym_atc_data (Dict[str, Any]): Dictionary containing titles, synonyms, ATC codes, and SMILES.
        drug_id (str): The drug ID for which the ATC code is required.

    Returns:
        Optional[str]: The ATC code if available, None otherwise.
    """
    candidate_data = data.get(str(drug_id), {}) or {}

    candidate_atc_code = candidate_data.get("atc_code")
    return candidate_atc_code or None


def chunk_dict(data: Dict[str, T], num_chunks: int) -> Iterator[Dict[str, T]]:
    """
    Splits a dictionary into `num_chunks` approximately equal parts, yielding each part as a new dictionary.

    Args:
        data (Dict[str, T]): The dictionary to be split.
        num_chunks (int): The number of chunks to split the dictionary into.

    Returns:
        Iterator[Dict[str, T]]: An iterator that yields parts of the original dictionary.
    """
    it = iter(data)
    for i in range(num_chunks):
        yield {
            k: data[k]
            for k in itertools.islice(
                it, len(data) // num_chunks + (i < len(data) % num_chunks)
            )
        }


def process_trial_group(
    candidate_data: Tuple[List[str], List[Set[str]], List[str], List[str]],
    trial_chunk: Dict[str, Any],
    match_function: Callable[[Set[str], Set[str]], Any],
    sanitize: bool,
) -> Tuple[Dict[str, Any], Set[str], Set[str]]:
    """
    Processes a chunk of trial data to map drug IDs to intervention details using predefined synonyms and titles.

    Args:
        candidate_data (Tuple[List[str], List[Set[str]], List[str], List[str]]): Precomputed data consisting of drug IDs, their synonyms, titles, and ATC codes.
        trial_chunk (Dict[str, Any]): A chunk of trial data containing information about clinical trials.
        match_function (Callable[[Set[str], Set[str]], Any]): The matching function used to find corresponding synonyms.
        sanitize (bool): Flag indicating whether drug names should be sanitized before matching.

    Returns:
        Tuple[Dict[str, Any], Set[str], Set[str]]: Modified trial data with updated intervention details, set of mapped study group codes, and set of unique SMILES identifiers.
    """
    global drug_id_details_global
    # Unpack precomputed candidate data
    (
        all_drug_ids,
        candidate_drug_ids_synonyms,
        candidate_drug_ids_titles,
        candidate_drug_ids_atc_codes,
    ) = candidate_data
    mapped_study_group_codes = set()
    unique_smiles_mapped = set()
    modified_trial_data = {}

    # Process each trial in the chunk
    for nctid, trial in trial_chunk.items():
        for study_group in trial["study_groups"]:
            if "smiles" in study_group["intervention_details"]:
                continue
            intervention_details = study_group["intervention_details"]
            raw_intervention_name = (
                intervention_details["name"][0].lower().replace("drug:", "").strip()
            )
            intervention_synonyms = normalize_synonyms(
                [raw_intervention_name] + intervention_details.get("synonyms", []),
                sanitize=sanitize,
            )

            for drug_id, synonyms, title, atc_code in zip(
                all_drug_ids,
                candidate_drug_ids_synonyms,
                candidate_drug_ids_titles,
                candidate_drug_ids_atc_codes,
            ):
                matched_name = match_function(synonyms, intervention_synonyms)
                if matched_name:
                    canonical_name = title or matched_name.lower().strip()
                    smiles = drug_id_details_global.get(drug_id, {}).get("smiles")
                    if canonical_name and smiles:
                        unique_smiles_mapped.add(smiles)
                        mapped_study_group_codes.add(study_group["group_code"])
                        intervention_details.update(
                            {
                                "canonical_name": canonical_name,
                                "drug_id": drug_id,
                                "smiles": smiles,
                                "atc_code": atc_code,
                            }
                        )
                        break
        modified_trial_data[nctid] = trial

    return modified_trial_data, mapped_study_group_codes, unique_smiles_mapped


def process_matching_multiprocessing(
    trial_data: Dict[str, Any],
    drug_id_details: Dict[str, Any],
    match_function: Callable[[Set[str], Set[str]], Optional[str]],
    sanitize: bool = False,
    num_processes: Optional[int] = None,
) -> Tuple[Dict[str, Any], Set[str], Set[str]]:
    """
    Processes trial data in parallel to match drug synonyms and maps additional data such as SMILES and ATC codes.

    Args:
        trial_data (Dict[str, Any]): Dictionary containing trial details.
        drug_id_details (Dict[str, Any]): Dictionary containing additional drug details.
        match_function (Callable[[Set[str], Set[str]], Optional[str]]): Function used to match synonyms.
        sanitize (bool): Whether to sanitize synonyms during processing.
        num_processes (Optional[int]): Number of parallel processes to use. Uses all available CPUs if None.

    Returns:
        Tuple[Dict[str, Any], Set[str], Set[str]]: The updated trial data, set of mapped study group codes, and set of unique SMILES strings.
    """
    global drug_id_details_global
    drug_id_details_global = deepcopy(drug_id_details)

    if num_processes is None:
        num_processes = cpu_count()

    # Precompute data for all drug_ids
    all_drug_ids = list(drug_id_details.keys())
    candidate_data = (
        all_drug_ids,
        [
            normalize_synonyms(
                get_combined_synonyms(drug_id_details, drug_id),
                sanitize=sanitize,
            )
            for drug_id in all_drug_ids
        ],
        [get_title(drug_id_details, drug_id) for drug_id in all_drug_ids],
        [get_atc_code(drug_id_details, drug_id) for drug_id in all_drug_ids],
    )

    # Split trial data into chunks for multiprocessing
    trial_data_chunks = list(chunk_dict(trial_data, num_processes))

    with Pool(processes=num_processes) as pool:
        jobs = [
            pool.apply_async(
                process_trial_group,
                (candidate_data, dict(chunk), match_function, sanitize),
            )
            for chunk in trial_data_chunks
        ]
        results = [
            job.get()
            for job in tqdm(
                jobs,
                total=len(jobs),
                desc=f"Processing {num_processes} trial chunks",
                leave=False,
            )
        ]

    # Reintegrate modified trial data chunks back into the original trial_data dictionary
    updated_trial_data = {}
    all_mapped_study_group_codes = set()
    all_unique_smiles_mapped = set()
    for modified_chunk, mapped_codes, smiles in results:
        updated_trial_data.update(modified_chunk)
        all_mapped_study_group_codes.update(mapped_codes)
        all_unique_smiles_mapped.update(smiles)

    return updated_trial_data, all_mapped_study_group_codes, all_unique_smiles_mapped


def print_summary(title: str, codes: Set[str], smiles: Set[str]) -> None:
    """
    Prints a summary of the mapping process, including the number of mapped study groups and unique drugs.

    Args:
        title (str): The title for the summary.
        codes (Set[str]): The set of study group codes that have been mapped.
        smiles (Set[str]): The set of unique SMILES strings that have been mapped.
    """
    print(f"\n{title}:")
    print(f"Study groups mapped: {len(codes)}")
    print(f"Unique drugs mapped: {len(smiles)}\n")


def collect_mapped_study_groups(
    preprocessed_trials: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Collects and combines data from trials and their study groups where study groups have been mapped.

    Args:
        preprocessed_trials (Dict[str, Dict[str, Any]]): preprocessed trials from `process_matching_multiprocessing` output.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each combining data from a trial and its mapped study groups.
    """
    mapped_study_groups = []
    for nct_id, trial in preprocessed_trials.items():
        for study_group in trial["study_groups"]:
            if "smiles" in study_group["intervention_details"]:
                combined_data = {
                    "nctid": trial["nctid"],
                    "title": trial["title"],
                    "status": trial["status"],
                    "sponsor": trial["sponsor"],
                    "collaborators": trial["collaborators"],
                    "healthy_volunteers": trial["healthy_volunteers"],
                    "gender": trial["gender"],
                    "age": trial["age"],
                    "phase": trial["phase"],
                    "enrollment_count": trial["enrollment_count"],
                    "eligibility_criteria": trial["eligibility_criteria"],
                    **study_group,
                }
                mapped_study_groups.append(combined_data)
    return mapped_study_groups


def tabularize_study_groups(study_groups: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Processes a list of clean and mapped study groups, extracting relevant data and adverse events,
    then compiling this information into a DataFrame.

    Args:
        study_groups (List[Dict[str, Any]]): A list of study groups with comprehensive data including
                                             trial information and adverse event details.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data from each study group, including details
                      on adverse events.
    """
    # Initialize an empty list to store row data
    data_rows = []

    # Iterate over each study group
    for group in study_groups:

        # Base information from the study group
        base_info = {
            "nctid": group["nctid"],
            "title": group["title"],
            "status": group["status"],
            "sponsor": group["sponsor"],
            "collaborators": group["collaborators"],
            "healthy_volunteers": group["healthy_volunteers"],
            "gender": group["gender"],
            "age": group["age"],
            "phase": group["phase"],
            "enrollment_count": group["enrollment_count"],
            "eligibility_criteria": group["eligibility_criteria"],
            "group_id": group["group_code"],
            "group_description": group["intervention_details"]["description"],
            "ct_intervention_name": group["intervention_details"]["name"][0],
            "canonical_name": group["intervention_details"]["canonical_name"],
            "drug_id": group["intervention_details"]["drug_id"],
            "smiles": group["intervention_details"]["smiles"],
            "atc_code": group["intervention_details"]["atc_code"],
        }

        # Function to add adverse events to the list
        def add_adverse_events(events: List[Dict[str, Any]], event_type: str) -> int:
            for event in events:
                row = base_info.copy()  # Start with the base info
                row.update(
                    {
                        "event_type": event_type,
                        "ade_vocabulary": event["ade_vocabulary"],
                        "ade_organ_system": event["ade_organ_system"],
                        "ade_term": event["ade_term"],
                        "ade_num_affected": event["ade_num_affected"],
                        "ade_num_at_risk": event["ade_num_at_risk"],
                    }
                )
                data_rows.append(row)
            return len(events)

        # Track if any events are added
        events_added = 0
        events_added += add_adverse_events(
            group["adverse_events"]["serious_events"], "Serious"
        )
        events_added += add_adverse_events(
            group["adverse_events"]["other_events"], "Other"
        )

        # If no events added, append a base_info row with default values for event fields
        if events_added == 0:
            no_event_row = base_info.copy()
            no_event_row.update(
                {
                    "event_type": "No Event",
                    "ade_vocabulary": None,
                    "ade_organ_system": None,
                    "ade_term": None,
                    "ade_num_affected": 0,
                    "ade_num_at_risk": base_info["enrollment_count"],
                }
            )
            data_rows.append(no_event_row)

    # Create a DataFrame from the collected data rows
    return pd.DataFrame(data_rows)


def main() -> None:
    """
    Main function that orchestrates the execution of the ade data mapping pipeline.
    """
    # Load data from JSON files
    preprocessed_monopharmacy_cts_mapped = read_json_file("./data/clinicaltrials_gov/preprocessed_monopharmacy_cts.json")
    loaded_compound_details = read_json_file("./data/unified_chemical_database/unified_chemical_database.json")

    # Count the number of unique study groups
    unique_study_group_count = count_study_groups(preprocessed_monopharmacy_cts_mapped)
    print(f"Loaded data has {unique_study_group_count} unique study groups")

    # Initialize sets to store combined codes and smiles
    all_mapped_codes = set()
    all_unique_smiles = set()

    # Exact matching
    (
        preprocessed_monopharmacy_cts_mapped,
        codes_exact,
        smiles_exact,
    ) = process_matching_multiprocessing(
        preprocessed_monopharmacy_cts_mapped,
        loaded_compound_details,
        find_exact_match,
        sanitize=False,
    )
    all_mapped_codes.update(codes_exact)
    all_unique_smiles.update(smiles_exact)

    # Partial matching
    (
        preprocessed_monopharmacy_cts_mapped,
        codes_partial,
        smiles_partial,
    ) = process_matching_multiprocessing(
        preprocessed_monopharmacy_cts_mapped,
        loaded_compound_details,
        find_partial_match,
        sanitize=False,
    )
    all_mapped_codes.update(codes_partial)
    all_unique_smiles.update(smiles_partial)

    # Pre-processed exact matching
    (
        preprocessed_monopharmacy_cts_mapped,
        codes_pre_exact,
        smiles_pre_exact,
    ) = process_matching_multiprocessing(
        preprocessed_monopharmacy_cts_mapped,
        loaded_compound_details,
        find_exact_match,
        sanitize=True,
    )
    all_mapped_codes.update(codes_pre_exact)
    all_unique_smiles.update(smiles_pre_exact)

    # Pre-processed partial matching
    (
        preprocessed_monopharmacy_cts_mapped,
        codes_pre_partial,
        smiles_pre_partial,
    ) = process_matching_multiprocessing(
        preprocessed_monopharmacy_cts_mapped,
        loaded_compound_details,
        find_partial_match,
        sanitize=True,
    )
    all_mapped_codes.update(codes_pre_partial)
    all_unique_smiles.update(smiles_pre_partial)

    # Print the final combined summary
    print_summary("Final Combined Results", all_mapped_codes, all_unique_smiles)

    # Compute the final mapping percentage
    mapped_percentage = (len(all_mapped_codes) / unique_study_group_count) * 100
    print(f"Among the {unique_study_group_count} unique study groups, {len(all_mapped_codes)} were mapped ({mapped_percentage:.2f}%)\n")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Collect mapped study groups
    mapped_study_groups = collect_mapped_study_groups(
        preprocessed_monopharmacy_cts_mapped
    )

    # Create final tabular data
    final_pt_level_dataset = tabularize_study_groups(mapped_study_groups)

    # And save it
    output_file_path = Path("./data/ct_ade/ct_ade_raw.csv")
    output_folder = output_file_path.parent
    output_folder.mkdir(parents=True, exist_ok=True)
    final_pt_level_dataset.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    main()