import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_confint
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
import warnings
from typing import Dict, List, Tuple, Any
from src.meddra_graph import MedDRA, Node
from sklearn.model_selection import train_test_split
from copy import deepcopy
from pathlib import Path


warnings.filterwarnings("ignore", category=FutureWarning)


def do_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_nodes_by_level(nodes: Dict[Tuple[str, str], Node], level: str) -> List[Node]:
    """
    Retrieve nodes from a dictionary that match a specified level.

    Args:
        nodes (Dict[Tuple[str, str], Node]): Dictionary of nodes keyed by (level, code).
        level (str): Level to filter nodes by, such as "SOC", "PT", etc.

    Returns:
        List[Node]: Nodes that match the specified level.
    """
    return [node for (node_level, _), node in nodes.items() if node_level == level]


def apply_wilson_lower_bound(row: pd.Series) -> float:
    """
    Calculate the lower bound of the Wilson score interval for binomial proportion confidence.

    Args:
        row (pd.Series): A row of a DataFrame, expected to contain 'ade_num_affected' and 'ade_num_at_risk'.

    Returns:
        float: The lower bound of the Wilson score interval, or NaN if conditions are not met.
    """
    if row["ade_num_affected"] >= 0 and row["ade_num_at_risk"] > 0:
        ci_lower, _ = proportion_confint(
            count=row["ade_num_affected"],
            nobs=row["ade_num_at_risk"],
            alpha=0.1,  # One-sided 95% confidence
            method="wilson",
        )
        return ci_lower
    else:
        return np.nan


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the Wilson lower bound calculation to each row in a DataFrame chunk and mark significant events
    with True if >= 0.01, False if < 0.01, and NaN if NaN.

    Args:
        chunk (pd.DataFrame): The DataFrame chunk to process.

    Returns:
        pd.DataFrame: The chunk with additional columns for the confidence interval lower bound and significance.
    """
    chunk["ci_lower_bound"] = chunk.apply(apply_wilson_lower_bound, axis=1)
    chunk["is_significant"] = chunk["ci_lower_bound"].apply(
        lambda x: x >= 0.01 if not pd.isna(x) else np.nan
    )
    return chunk


def event_type_classification(group_df: pd.DataFrame) -> pd.Series:
    """
    Classify the event type based on significance and the type of event.

    Args:
        group_df (pd.DataFrame): DataFrame containing event data.

    Returns:
        pd.Series: A series with labels indicating the presence of serious, other, or no significant events.
    """
    events = group_df["event_type"]
    significance = group_df["is_significant"]

    has_serious_event = float(any((events == "Serious") & significance))
    has_other_event = float(any((events == "Other") & significance))
    has_no_event = float(all(events == "No Event") or (not any(significance) and not any(pd.isna(significance))))
    assert not (has_serious_event == has_other_event == has_no_event == 1)

    return pd.Series(
        {
            "label_serious_event": has_serious_event,
            "label_other_event": has_other_event,
            "label_no_event": has_no_event,
        }
    )


def init_globals(ct_ade_meddra_instance: pd.DataFrame) -> None:
    """
    Initializes global variables for use within a multiprocessing environment.

    Args:
        ct_ade_meddra_instance (pd.DataFrame): Loaded ct_ade_meddra data.
    """
    global ct_ade_meddra
    ct_ade_meddra = ct_ade_meddra_instance


def process_group(group_id: str) -> Dict[str, Any]:
    """
    Process data for a specific group ID.

    Args:
        group_id (str): The group ID to process.

    Returns:
        Dict[str, Any]: A dictionary of aggregated data for the group.
    """
    group_df = ct_ade_meddra[ct_ade_meddra["group_id"] == group_id]
    pass_condition = len(group_df[group_df.is_significant.notna()]) == len(group_df)

    if not pass_condition:
        return None

    event_labels = event_type_classification(group_df)

    # Check if all labels are zero
    if event_labels.eq(0).all():
        return None

    result = {
        "nctid": group_df["nctid"].iloc[0],
        "group_id": group_df["group_id"].iloc[0],
        "healthy_volunteers": int(group_df["healthy_volunteers"].iloc[0] != "No"),
        "gender": group_df["gender"].iloc[0],
        "age": group_df["age"].iloc[0],
        "phase": group_df["phase"].iloc[0],
        "ade_num_at_risk": group_df["ade_num_at_risk"].iloc[0],
        "eligibility_criteria": group_df["eligibility_criteria"].iloc[0],
        "group_description": group_df["group_description"].iloc[0],
        "intervention_name": group_df["canonical_name"].iloc[0],
        "smiles": group_df["smiles"].iloc[0],
        "atc_code": group_df["atc_code"].iloc[0],
        **event_labels,
    }
    return result


def process_group_data(args: Tuple[pd.DataFrame, List[str], str]) -> pd.DataFrame:
    """
    Process data for a group, handling the application of dummy variable encoding and aggregation.

    Args:
        args (Tuple[pd.DataFrame, List[str], str]): Tuple containing the group DataFrame, all SOC codes, and the target level.

    Returns:
        pd.DataFrame: Aggregated group data.
    """
    bool_map = {True: 1, False: 0, np.nan: np.nan}

    group_df, all_codes, level = args

    # Extract relevant columns and copy the DataFrame slice
    group_label_info = group_df[[f"ade_mapped_code_{level}", "is_significant"]].copy()

    # Check if all significant f"ade_mapped_code_{level}" are not NaN
    all_significant_non_nan = group_df[group_df["is_significant"] == True][f"ade_mapped_code_{level}"].notna().all()

    # Check if all is_significant values are not NaN
    all_is_significant_non_nan = group_df["is_significant"].notna().all()

    # Combine both conditions
    pass_condition = all_significant_non_nan and all_is_significant_non_nan

    # Only proceed if the pass_condition is met
    if pass_condition:
        # Map boolean values to integers, respecting NaN
        group_label_info["is_significant"] = group_label_info["is_significant"].map(bool_map)

        # Pivot and reindex DataFrame to match the expected column structure
        all_dummies = group_label_info.pivot(columns=f"ade_mapped_code_{level}", values="is_significant")
        all_dummies = all_dummies.reindex(columns=all_codes, fill_value=0.0)

        # Concatenate dummies with original group DataFrame
        group_df = pd.concat([group_df, all_dummies], axis=1)

        # Define aggregation dictionary for grouped data
        agg_dict = {
            "nctid": "first",
            "group_id": "first",
            "healthy_volunteers": "first",
            "gender": "first",
            "age": "first",
            "phase": "first",
            "ade_num_at_risk": "first",
            "eligibility_criteria": "first",
            "group_description": "first",
            "canonical_name": "first",
            "smiles": "first",
            "atc_code": "first",
            **{col: "max" for col in all_codes},  # Aggregate dummies with max to ensure binary presence flags
        }

        # Perform aggregation
        group_df_agg = group_df.groupby("group_id", as_index=False).agg(agg_dict)
        group_df_agg.rename(columns={"canonical_name": "intervention_name"}, inplace=True)

        return group_df_agg.to_dict("records")
    else:
        # Return an empty list if not all significant entries are mapped
        return []


def split_dataframe_by_smiles(
    df: pd.DataFrame,
    train_smiles: List[str],
    val_smiles: List[str],
    test_smiles: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into train, validation, and test sets based on lists of SMILES.

    Args:
        df (pd.DataFrame): The original DataFrame to split.
        train_smiles (List[str]): List of SMILES strings for the training set.
        val_smiles (List[str]): List of SMILES strings for the validation set.
        test_smiles (List[str]): List of SMILES strings for the test set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Three DataFrames corresponding to the
                                                        training, validation, and test datasets.
    """
    # Assign samples to splits based on SMILES
    train_df = df[df["smiles"].isin(train_smiles)].reset_index(drop=True)
    val_df = df[df["smiles"].isin(val_smiles)].reset_index(drop=True)
    test_df = df[df["smiles"].isin(test_smiles)].reset_index(drop=True)

    return train_df, val_df, test_df


def main() -> None:
    ct_ade_meddra = pd.read_csv(
        "./data/ct_ade/ct_ade_meddra.csv", #"./data/ct_ade/ct_ade_meddra.csv",
        dtype={
            "ade_mapped_code_SOC": str,
            "ade_mapped_code_HLGT": str,
            "ade_mapped_code_HLT": str,
            "ade_mapped_code_PT": str,
            "ade_mapped_code_LLT": str,
        },
    )

    meddra = MedDRA()
    meddra.load_data("./data/MedDRA_25_0_English/MedAscii")

    # Splitting the DataFrame into chunks
    chunks = np.array_split(ct_ade_meddra, cpu_count())

    # Pool initialization and process execution with progress bar
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks)))

    # Combine all processed chunks
    ct_ade_meddra = pd.concat(results, ignore_index=True)

    #### SOC classification ####

    # Retrieve all existing SOC codes
    SOC_codes = [i.code for i in get_nodes_by_level(meddra.nodes, "SOC")]

    # Prepare data tuples for parallel processing
    group_data = [
        (group, SOC_codes, "SOC") for _, group in ct_ade_meddra.groupby("group_id")
    ]

    # Using multiprocessing Pool with imap for better integration with tqdm
    with Pool(
        cpu_count(),
        initializer=init_globals,
        initargs=(deepcopy(ct_ade_meddra),),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(process_group_data, group_data),
                total=len(group_data),
                desc="Creating CT-ADE SOC",
            )
        )

    # Create a DataFrame from the results in chunks to visualize progress
    all_records = []
    for record_batch in results:
        all_records.extend(record_batch)

    # Process chunks of records into DataFrames and concatenate them to monitor progress
    chunk_size = 100  # Adjust based on memory capacity and list size
    SOC_classification_df = pd.DataFrame()  # Initialize an empty DataFrame

    for chunk in tqdm(
        do_chunks(all_records, chunk_size),
        total=(len(all_records) // chunk_size) + 1,
        desc="Building DataFrame",
    ):
        df_chunk = pd.DataFrame(chunk)
        SOC_classification_df = pd.concat(
            [SOC_classification_df, df_chunk], ignore_index=True
        )

    # Create a DataFrame from all records
    SOC_classification_df = SOC_classification_df.sort_values(by="group_id")
    SOC_classification_df = SOC_classification_df.reset_index(drop=True)
    SOC_classification_df.columns = [
        f"label_{col}" if col in SOC_codes else col
        for col in SOC_classification_df.columns
    ]
    print("SOC_classification_df", f"{len(SOC_classification_df)} study groups", f"{SOC_classification_df.smiles.nunique()} unique drugs")

    # Split the data
    train_df, val_df, test_df = split_dataframe_by_smiles(
        SOC_classification_df, train_smiles, val_smiles, test_val_smiles
    )

    # Save data
    output_folder = Path("./data/ct_ade/soc")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save each split separately to ensure data is not mixed in subsequent analyses
    train_df.to_csv(output_folder / "train.csv", index=False)
    val_df.to_csv(output_folder / "val.csv", index=False)
    test_df.to_csv(output_folder / "test.csv", index=False)

    #### HLGT classification ####

    # Retrieve all existing HLGT codes
    HLGT_codes = [i.code for i in get_nodes_by_level(meddra.nodes, "HLGT")]

    # Prepare data tuples for parallel processing
    group_data = [
        (group, HLGT_codes, "HLGT") for _, group in ct_ade_meddra.groupby("group_id")
    ]

    # Using multiprocessing Pool with imap for better integration with tqdm
    with Pool(
        cpu_count(),
        initializer=init_globals,
        initargs=(deepcopy(ct_ade_meddra),),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(process_group_data, group_data),
                total=len(group_data),
                desc="Creating CT-ADE HLGT",
            )
        )

    # Create a DataFrame from the results in chunks to visualize progress
    all_records = []
    for record_batch in results:
        all_records.extend(record_batch)

    # Process chunks of records into DataFrames and concatenate them to monitor progress
    chunk_size = 100  # Adjust based on memory capacity and list size
    HLGT_classification_df = pd.DataFrame()  # Initialize an empty DataFrame

    for chunk in tqdm(
        do_chunks(all_records, chunk_size),
        total=(len(all_records) // chunk_size) + 1,
        desc="Building DataFrame",
    ):
        df_chunk = pd.DataFrame(chunk)
        HLGT_classification_df = pd.concat(
            [HLGT_classification_df, df_chunk], ignore_index=True
        )

    # Create a DataFrame from all records
    HLGT_classification_df = HLGT_classification_df.sort_values(by="group_id")
    HLGT_classification_df = HLGT_classification_df.reset_index(drop=True)
    HLGT_classification_df.columns = [
        f"label_{col}" if col in HLGT_codes else col
        for col in HLGT_classification_df.columns
    ]
    print("HLGT_classification_df", f"{len(HLGT_classification_df)} study groups", f"{HLGT_classification_df.smiles.nunique()} unique drugs")

    # Split the data
    train_df, val_df, test_df = split_dataframe_by_smiles(
        HLGT_classification_df, train_smiles, val_smiles, test_val_smiles
    )

    # Save data
    output_folder = Path("./data/ct_ade/hlgt")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save each split separately to ensure data is not mixed in subsequent analyses
    train_df.to_csv(output_folder / "train.csv", index=False)
    val_df.to_csv(output_folder / "val.csv", index=False)
    test_df.to_csv(output_folder / "test.csv", index=False)

    #### HLT classification ####

    # Retrieve all existing HLT codes
    HLT_codes = [i.code for i in get_nodes_by_level(meddra.nodes, "HLT")]

    # Prepare data tuples for parallel processing
    group_data = [
        (group, HLT_codes, "HLT") for _, group in ct_ade_meddra.groupby("group_id")
    ]

    # Using multiprocessing Pool with imap for better integration with tqdm
    with Pool(
        cpu_count(),
        initializer=init_globals,
        initargs=(deepcopy(ct_ade_meddra),),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(process_group_data, group_data),
                total=len(group_data),
                desc="Creating CT-ADE HLT",
            )
        )

    # Create a DataFrame from the results in chunks to visualize progress
    all_records = []
    for record_batch in results:
        all_records.extend(record_batch)

    # Process chunks of records into DataFrames and concatenate them to monitor progress
    chunk_size = 100  # Adjust based on memory capacity and list size
    HLT_classification_df = pd.DataFrame()  # Initialize an empty DataFrame

    for chunk in tqdm(
        do_chunks(all_records, chunk_size),
        total=(len(all_records) // chunk_size) + 1,
        desc="Building DataFrame",
    ):
        df_chunk = pd.DataFrame(chunk)
        HLT_classification_df = pd.concat(
            [HLT_classification_df, df_chunk], ignore_index=True
        )

    # Create a DataFrame from all records
    HLT_classification_df = HLT_classification_df.sort_values(by="group_id")
    HLT_classification_df = HLT_classification_df.reset_index(drop=True)
    HLT_classification_df.columns = [
        f"label_{col}" if col in HLT_codes else col
        for col in HLT_classification_df.columns
    ]
    print("HLT_classification_df", f"{len(HLT_classification_df)} study groups", f"{HLT_classification_df.smiles.nunique()} unique drugs")

    # Split the data
    train_df, val_df, test_df = split_dataframe_by_smiles(
        HLT_classification_df, train_smiles, val_smiles, test_val_smiles
    )

    # Save data
    output_folder = Path("./data/ct_ade/hlt")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save each split separately to ensure data is not mixed in subsequent analyses
    train_df.to_csv(output_folder / "train.csv", index=False)
    val_df.to_csv(output_folder / "val.csv", index=False)
    test_df.to_csv(output_folder / "test.csv", index=False)

    #### PT classification ####

    # Retrieve all existing PT codes
    PT_codes = [i.code for i in get_nodes_by_level(meddra.nodes, "PT")]

    # Prepare data tuples for parallel processing
    group_data = [
        (group, PT_codes, "PT") for _, group in ct_ade_meddra.groupby("group_id")
    ]

    # Using multiprocessing Pool with imap for better integration with tqdm
    with Pool(
        cpu_count(),
        initializer=init_globals,
        initargs=(deepcopy(ct_ade_meddra),),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(process_group_data, group_data),
                total=len(group_data),
                desc="Creating CT-ADE PT",
            )
        )

    # Create a DataFrame from the results in chunks to visualize progress
    all_records = []
    for record_batch in results:
        all_records.extend(record_batch)

    # Process chunks of records into DataFrames and concatenate them to monitor progress
    chunk_size = 100  # Adjust based on memory capacity and list size
    PT_classification_df = pd.DataFrame()  # Initialize an empty DataFrame

    for chunk in tqdm(
        do_chunks(all_records, chunk_size),
        total=(len(all_records) // chunk_size) + 1,
        desc="Building DataFrame",
    ):
        df_chunk = pd.DataFrame(chunk)
        PT_classification_df = pd.concat(
            [PT_classification_df, df_chunk], ignore_index=True
        )

    # Create a DataFrame from all records
    PT_classification_df = PT_classification_df.sort_values(by="group_id")
    PT_classification_df = PT_classification_df.reset_index(drop=True)
    PT_classification_df.columns = [
        f"label_{col}" if col in PT_codes else col
        for col in PT_classification_df.columns
    ]
    print("PT_classification_df", f"{len(PT_classification_df)} study groups", f"{PT_classification_df.smiles.nunique()} unique drugs")

    # Split the data
    train_df, val_df, test_df = split_dataframe_by_smiles(
        PT_classification_df, train_smiles, val_smiles, test_val_smiles
    )

    # Save data
    output_folder = Path("./data/ct_ade/pt")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save each split separately to ensure data is not mixed in subsequent analyses
    train_df.to_csv(output_folder / "train.csv", index=False)
    val_df.to_csv(output_folder / "val.csv", index=False)
    test_df.to_csv(output_folder / "test.csv", index=False)


if __name__ == "__main__":
    main()
