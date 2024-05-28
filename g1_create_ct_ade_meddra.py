from typing import Dict, List, Tuple, Set, Optional, Callable, Any
from src.meddra_graph import MedDRA, Node
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from itertools import chain
from pathlib import Path


def clean_term(input_string: str) -> str:
    """
    Cleans an input string by removing non-alphanumeric characters, converting to lower case, and stripping whitespace.

    Args:
    input_string (str): The string to be cleaned.

    Returns:
    str: A cleaned version of the input string.
    """
    cleaned_string = re.sub(r"[^a-zA-Z0-9 ]", "", input_string)
    cleaned_string = cleaned_string.lower()
    cleaned_string = cleaned_string.strip()
    return cleaned_string


def process_codes(
    codes_dict: Dict[str, List[Tuple[str, str, str]]]
) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Processes the codes dictionary to remove empty entries.

    Args:
    codes_dict (Dict[str, List[Tuple[str, str, str]]]): The dictionary containing terms mapped to their codes.

    Returns:
    Dict[str, List[Tuple[str, str, str]]]: A processed dictionary with empty entries removed.
    """
    processed_codes = {}
    for term, codes in codes_dict.items():
        if codes:
            processed_codes[term] = [(str(code), term, level) for code, term, level in codes]

    return processed_codes


def init_globals_terms_to_codes(*args):
    """
    Initializes global variables in the multiprocessing environment.

    Args:
    *args: Tuple containing the MedDRA instance and the preprocess function.
    """
    global meddra, preprocess
    meddra, preprocess = args


def map_term_to_code(term: str) -> Tuple[str, List[Tuple[str, str, str]]]:
    """
    Maps a term to its corresponding codes using the MedDRA instance.

    Args:
    term (str): The term to map.

    Returns:
    Tuple[str, List[Tuple[str, str, str]]]: A tuple containing the term and a list of tuples (code, term, level).
    """
    nodes = meddra.find_node_by_term(term, preprocess=preprocess)
    codes = [(str(node.code), node.term, node.level) for node in nodes]
    return term, codes


def run_multiprocessing_terms_to_codes(
    terms: List[str],
    meddra_instance: MedDRA,
    preprocess_func: Optional[Callable[[str], str]] = None,
    num_cpus: Optional[int] = None,
) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Maps a list of terms to their MedDRA codes using multiprocessing for efficiency.

    Args:
    terms (List[str]): List of terms to map.
    meddra_instance (MedDRA): An instance of the MedDRA class.
    preprocess_func (Callable[[str], str], optional): A function to preprocess terms.
    num_cpus (int, optional): Number of CPUs to use; defaults to the number of CPUs available on the system.

    Returns:
    Dict[str, List[Tuple[str, str, str]]]: Dictionary mapping terms to a list of their codes.
    """
    if num_cpus is None:
        num_cpus = cpu_count()

    pbar = tqdm(total=len(terms), desc="Mapping raw terms to MedDRA")

    with Pool(
        num_cpus,
        initializer=init_globals_terms_to_codes,
        initargs=(meddra_instance, preprocess_func),
    ) as pool:
        results = {}
        for result in pool.imap_unordered(map_term_to_code, terms):
            results[result[0]] = result[1]
            pbar.update()

    pbar.close()
    return results


def init_globals_cache_and_chunk(meddra_instance: MedDRA) -> None:
    """
    Initializes global variables for use within a multiprocessing environment.

    Args:
        meddra_instance (MedDRA): An instance of the MedDRA class used to access medical terms and their relationships.
    """
    global meddra
    meddra = meddra_instance


def find_and_cache_paths(
    code_level_tuple: Tuple[str, str]
) -> Tuple[Tuple[str, str], Any]:
    """
    Finds and caches paths for a given medical code and level using the global MedDRA instance.

    Args:
        code_level_tuple (Tuple[str, str]): A tuple containing the level and code of the medical term.

    Returns:
        Tuple[Tuple[str, str], Any]: A tuple of the code_level_tuple and the paths found from that code and level.
    """
    level, code = code_level_tuple
    code = str(code)
    paths = meddra.find_paths(code, level, pad_levels=False)
    return (level, code), paths


def cache_meddra_paths_multiprocessing(
    df: pd.DataFrame, meddra_instance: MedDRA
) -> Dict[Tuple[str, str], Any]:
    """
    Caches paths for all unique medical codes in the DataFrame across specified MedDRA levels using multiprocessing.

    Args:
        df (pd.DataFrame): The DataFrame containing medical codes.
        meddra_instance (MedDRA): An instance of the MedDRA class.

    Returns:
        Dict[Tuple[str, str], Any]: A dictionary where keys are tuples of (level, code) and values are the cached paths.
    """
    meddra_levels = ["LLT", "PT", "HLT", "HLGT"]  # Levels to process
    tasks = []

    # Collect all unique codes and levels that need path computation
    for level in meddra_levels:
        list_of_code_lists = (
            df[f"ade_mapped_code_{level}"].dropna().apply(lambda x: x.split(" | "))
        )
        unique_codes = set(chain.from_iterable(list_of_code_lists))
        tasks.extend([(level, str(code)) for code in unique_codes])

    # Setup tqdm progress bar for multiprocessing
    pbar = tqdm(total=len(tasks), desc="Caching paths")

    # Use multiprocessing to compute paths and cache them
    with Pool(
        cpu_count(),
        initializer=init_globals_cache_and_chunk,
        initargs=(meddra_instance,),
    ) as pool:
        path_cache = {}
        for result in pool.imap_unordered(find_and_cache_paths, tasks):
            path_cache[result[0]] = result[1]
            pbar.update()  # Update the progress bar as each task completes

    pbar.close()
    return path_cache

def map_to_consistent_level(cached_paths, target_level):
    """
    Filter the cached_paths dictionary to include only key-value pairs where all paths
    lead to the same target level code, and map each key to this consistent target level code.

    Parameters:
    cached_paths (dict): Dictionary containing paths.
    target_level (str): The target level to filter and map to (e.g., 'SOC', 'HLGT').

    Returns:
    dict: Filtered dictionary with key-value pairs where each key is mapped to a single consistent target level code.
    """
    filtered_paths = {}

    # Function to extract the target level code from a path
    def get_level_code(path, target_level):
        for code in path:
            if code.endswith(f'@{target_level}'):
                return code
        return None

    # Iterate through the cached_paths dictionary
    for key, paths in cached_paths.items():
        # Extract the target level codes for each path
        level_codes = [get_level_code(path, target_level) for path in paths]
        level_codes = [code for code in level_codes if code is not None]

        # Check if all target level codes are the same
        if level_codes:
            if all(code == level_codes[0] for code in level_codes):
                code, level = level_codes[0].split("@")
                filtered_paths[key] = (level, str(code))

    return filtered_paths

def fill_na_with_target_via_merge(to_process_ct_ade_positive, meddra, consistent_mappings, target_level):
    """
    Fill NaN values in the to_process_ct_ade_positive DataFrame based on consistent mappings for the specified target level.

    Parameters:
    to_process_ct_ade_positive (DataFrame): The DataFrame to process.
    meddra (object): The MedDRA object containing nodes.
    consistent_mappings (dict): Dictionary containing consistent mappings.
    target_level (str): The target level to fill NaN values for (e.g., 'SOC', 'HLGT').

    Returns:
    DataFrame: The updated DataFrame with NaN values filled for the specified target level.
    """
    # Step 1: Create a DataFrame from the mappings
    mappings = []
    for k, v in consistent_mappings.items():
        query_node = meddra.nodes[k]
        target_node = meddra.nodes[v]
        mappings.append({
            'query_level': k[0],   # Extract the level (e.g., 'LLT')
            'query_code': str(query_node.code),
            'target_code': str(target_node.code),
            'target_term': target_node.term
        })
    
    mappings_df = pd.DataFrame(mappings)

    # Step 2: Iterate over unique query levels in the mappings
    for level in mappings_df['query_level'].unique():
        # Filter mappings for the current level
        mappings_df_level = mappings_df[mappings_df['query_level'] == level]
        
        # Step 3: Perform the merge
        merged_df = to_process_ct_ade_positive.merge(
            mappings_df_level,
            how='left',   # Keep all rows from the left DataFrame
            left_on=f'ade_mapped_code_{level}',   # Match on the appropriate level code
            right_on='query_code',
            suffixes=('', '_map')   # Suffix to distinguish overlapping column names
        )
        
        # Step 4: Fill NaN values in the target level code and term columns
        target_code_column = f'ade_mapped_code_{target_level}'
        target_term_column = f'ade_mapped_term_{target_level}'

        merged_df[target_code_column] = merged_df[target_code_column].fillna(merged_df['target_code'])
        merged_df[target_term_column] = merged_df[target_term_column].fillna(merged_df['target_term'])
        
        # Drop the extra columns added by the merge
        to_process_ct_ade_positive = merged_df.drop(columns=['query_level', 'query_code', 'target_code', 'target_term'])
    
    return to_process_ct_ade_positive

def get_lowest_level_code(row):
    levels = ['LLT', 'PT', 'HLT', 'HLGT', 'SOC']
    columns = [f'ade_mapped_code_{level}' for level in levels]

    for level, column in zip(levels, columns):
        if pd.notna(row[column]):
            return (level, row[column]), row[columns]
    return (None,None), None

def find_and_fill_missing_codes(row, meddra):
    # Step 1: Get the lowest level code and columns
    (lowest_level, lowest_code), row_columns = get_lowest_level_code(row)
    
    if lowest_level is None or lowest_code is None:
        return row  # No code found, return the original row
    
    # Step 2: Handle multiple codes
    lowest_codes = lowest_code.split(' | ')
    all_paths = []
    
    for code in lowest_codes:
        # Step 3: Find all paths using meddra.find_paths
        paths = meddra.find_paths(meddra.nodes[(lowest_level, code)].code, lowest_level, pad_levels=False)
        all_paths.append(paths)
    
    # Step 4: Ensure paths agree across all codes
    def paths_agree(paths_list):
        if not paths_list:
            return []
        reference = set(map(tuple, paths_list[0]))
        for paths in paths_list[1:]:
            current_set = set(map(tuple, paths))
            reference = reference.intersection(current_set)
        return list(reference)
    
    agreed_paths = paths_agree(all_paths)
    
    # Step 5: Determine missing levels
    levels = ['LLT', 'PT', 'HLT', 'HLGT', 'SOC']
    code_columns = [f'ade_mapped_code_{level}' for level in levels]
    term_columns = [f'ade_mapped_term_{level}' for level in levels]
    missing_levels = [level for level, column in zip(levels, code_columns) if pd.isna(row[column])]
    
    if not missing_levels:
        return row  # No missing levels, return the original row
    
    # Step 6: Filter paths using mapped codes
    def path_respects_mapped_codes(path):
        path_dict = {code.split('@')[1]: code.split('@')[0] for code in path}
        for level, column in zip(levels, code_columns):
            if pd.notna(row[column]) and row[column] != path_dict.get(level, row[column]):
                return False
        return True

    filtered_paths = [path for path in agreed_paths if path_respects_mapped_codes(path)]
    
    # Step 7: Check for unique path to populate missing values
    possible_values = {level: set() for level in missing_levels}
    
    for path in filtered_paths:
        for code in path:
            level = code.split('@')[1]
            if level in missing_levels:
                possible_values[level].add(code.split('@')[0])
    
    # Step 8: Update the row with found values if unique
    for level in missing_levels:
        if len(possible_values[level]) == 1:
            code = list(possible_values[level])[0]
            row[f'ade_mapped_code_{level}'] = str(code)
            row[f'ade_mapped_term_{level}'] = meddra.nodes[(level, str(code))].term
    
    return row


def main() -> None:
    ct_ade_raw = pd.read_csv("./data/ct_ade/ct_ade_raw.csv")
    
    meddra = MedDRA()
    meddra.load_data("./data/MedDRA_25_0_English/MedAscii")
    
    ct_ade_meddra = ct_ade_raw.copy()
    
    unique_positive_ade_terms = [i for i in ct_ade_raw.ade_term.unique() if not pd.isna(i)]
    
    unique_positive_ade_codes = run_multiprocessing_terms_to_codes(
        unique_positive_ade_terms, meddra, clean_term
    )
    unique_positive_codes_clean = process_codes(unique_positive_ade_codes)
    
    unique_positive_organ_terms = [i for i in ct_ade_raw.ade_organ_system.unique() if not pd.isna(i)]
    unique_positive_organ_codes = run_multiprocessing_terms_to_codes(
        unique_positive_organ_terms, meddra, clean_term
    )
    unique_positive_organ_codes_clean = process_codes(unique_positive_organ_codes)
    
    meddra_levels = ["LLT", "PT", "HLT", "HLGT", "SOC"]
    
    # Prepare dictionaries to hold mapped terms and codes for each MedDRA level
    column_data = {f"ade_mapped_term_{level}": [] for level in meddra_levels}
    column_data.update({f"ade_mapped_code_{level}": [] for level in meddra_levels})
    
    # Aggregate mappings for each term to the corresponding MedDRA levels
    for term in ct_ade_meddra["ade_term"]:
        mappings = unique_positive_codes_clean.get(term, [])
        terms_by_level = {level: [] for level in meddra_levels}
        codes_by_level = {level: [] for level in meddra_levels}
    
        for code, mapped_term, level in mappings:
            terms_by_level[level].append(str(mapped_term))
            codes_by_level[level].append(str(code))  # Ensure code is treated as a string
    
        for level in meddra_levels:
            if level == "LLT": # One to many PT -> LLTs (no need to consider all LLTs)
                column_data[f"ade_mapped_term_{level}"].append(
                    str(terms_by_level[level][0]) if terms_by_level[level] else np.nan
                )
                column_data[f"ade_mapped_code_{level}"].append(
                    str(codes_by_level[level][0]) if codes_by_level[level] else np.nan
                )
            else:
                column_data[f"ade_mapped_term_{level}"].append(
                    " | ".join(terms_by_level[level]) if terms_by_level[level] else np.nan
                )
                column_data[f"ade_mapped_code_{level}"].append(
                    " | ".join(codes_by_level[level]) if codes_by_level[level] else np.nan
                )
    
    # Insert new columns for mapped terms and codes into the DataFrame
    insert_index = ct_ade_meddra.columns.get_loc("ade_term") + 1
    for level in reversed(meddra_levels):
        # Convert the list elements to string type explicitly to handle any discrepancies
        terms_to_insert = [
            str(x) if pd.notna(x) else np.nan
            for x in column_data[f"ade_mapped_term_{level}"]
        ]
        codes_to_insert = [
            str(x) if pd.notna(x) else np.nan
            for x in column_data[f"ade_mapped_code_{level}"]
        ]
        ct_ade_meddra.insert(
            insert_index, f"ade_mapped_code_{level}", codes_to_insert
        )
        ct_ade_meddra.insert(
            insert_index, f"ade_mapped_term_{level}", terms_to_insert
        )
    
    # Identify rows that lack SOC mappings
    missing_indicator = (
        ct_ade_meddra["ade_mapped_code_SOC"]
        .isna()
    )
    
    # Prepare a dictionary to map ade_organ_system to SOC codes and terms
    organ_to_soc = {}
    for organ, mappings in unique_positive_organ_codes_clean.items():
        soc_codes = [str(code) for code, term, level in mappings if level == "SOC"]
        soc_terms = [term for code, term, level in mappings if level == "SOC"]
        if soc_codes and soc_terms:
            organ_to_soc[organ] = {
                "SOC Codes": " | ".join(set(soc_codes)),
                "SOC Terms": " | ".join(set(soc_terms)),
            }
    
    # Use ade_organ_system mappings to fill missing SOC codes and terms for rows with no mappings
    ct_ade_meddra.loc[missing_indicator, "temp_soc_codes"] = ct_ade_meddra.loc[
        missing_indicator, "ade_organ_system"
    ].apply(lambda x: organ_to_soc.get(x, {}).get("SOC Codes", np.nan))
    ct_ade_meddra.loc[missing_indicator, "temp_soc_terms"] = ct_ade_meddra.loc[
        missing_indicator, "ade_organ_system"
    ].apply(lambda x: organ_to_soc.get(x, {}).get("SOC Terms", np.nan))
    
    # Fill missing SOC code and term columns and clean up temporary columns
    ct_ade_meddra["ade_mapped_code_SOC"] = ct_ade_meddra[
        "ade_mapped_code_SOC"
    ].fillna(ct_ade_meddra["temp_soc_codes"])
    ct_ade_meddra["ade_mapped_term_SOC"] = ct_ade_meddra[
        "ade_mapped_term_SOC"
    ].fillna(ct_ade_meddra["temp_soc_terms"])
    ct_ade_meddra.drop(["temp_soc_codes", "temp_soc_terms"], axis=1, inplace=True)
    
    # Process only the rows that don't meet the exclude condition
    cached_paths = cache_meddra_paths_multiprocessing(
        ct_ade_meddra, meddra
    )
    
    ct_ade_meddra_ = ct_ade_meddra.copy()
    for target_level in reversed(meddra_levels):
        consistent_mappings = map_to_consistent_level(cached_paths, target_level)
        ct_ade_meddra_ = fill_na_with_target_via_merge(ct_ade_meddra_, meddra, consistent_mappings, target_level)
    
    tqdm.pandas()
    ct_ade_meddra__ = ct_ade_meddra_.progress_apply(lambda row: find_and_fill_missing_codes(row, meddra), axis=1)
    
    ct_ade_meddra__ = ct_ade_meddra__.drop(columns=["ade_vocabulary"])
    ct_ade_meddra__ = ct_ade_meddra__.reset_index(drop=True)
    output_file_path = Path("./data/ct_ade/ct_ade_meddra.csv")
    output_folder = output_file_path.parent
    output_folder.mkdir(parents=True, exist_ok=True)
    ct_ade_meddra__.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    main()