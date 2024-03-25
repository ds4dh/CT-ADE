from src.preprocessing_helpers import *

if __name__ == "__main__":

    # Load data
    with open("./drugbank_data/aliases.json", "r") as file:
        aliases = json.load(file)

    with open("./drugbank_data/drug_dict.json", "r") as file:
        drug_dict = json.load(file)

    folder_path = "./data/completed_interventional_results_ades"
    json_files = json_file_paths(folder_path)
    results = parallel_process_json_files(json_files)

    # Unpack the results
    ade_data = []
    for data in results:
        ade_data.extend(data)

    # Keep only full instances
    ade_data_df = pd.DataFrame(ade_data)
    ade_data_df = ade_data_df.dropna().reset_index(drop=True)

    # Delete instances that are marked as non placebo or not contain any placebo,
    # but have the term 'placebo' in the title or group description
    serious_nopla = ade_data_df[
        (ade_data_df.is_placebo != 1) & (ade_data_df.contain_placebo != 1)
    ]
    rows_to_delete = serious_nopla[
        (serious_nopla.title.apply(lambda x: "placebo" in x.strip().lower()))
        | (
            serious_nopla.group_description.apply(
                lambda x: "placebo" in x.strip().lower()
            )
        )
    ].index
    ade_data_df = ade_data_df.drop(rows_to_delete).reset_index(drop=True)

    # Keep instances that have only drugs in intervention types
    rows_to_keep = ade_data_df.intervention_name.apply(
        lambda x: all("drug" in j for j in [i.split(":")[0].strip().lower() for i in x])
    )
    ade_data_df = ade_data_df[rows_to_keep].reset_index(drop=True)

    # keep only monopharmacy study groups
    rows_to_keep = ade_data_df.intervention_name.apply(lambda x: len(x) > 1)
    polypharmacy_ade_data_df = ade_data_df[rows_to_keep].reset_index(drop=True)
    rows_to_keep = ade_data_df.intervention_name.apply(lambda x: len(x) == 1)
    monopharmacy_ade_data_df = ade_data_df[rows_to_keep].reset_index(drop=True)
    ade_data_df = monopharmacy_ade_data_df.copy()

    # Sanitize intervention_name
    ade_data_df.intervention_name = ade_data_df.intervention_name.apply(
        sanitize_intervention_name
    )

    # Since we have monopharmacy instances, we should have is_placebo always equal to contain_placebo
    assert all(ade_data_df.is_placebo == ade_data_df.contain_placebo)

    # Filter out instances where ade_num_at_risk is 0
    # and sanitize occurence values
    ade_data_df = ade_data_df[ade_data_df.ade_num_at_risk > 0].reset_index(drop=True)
    ade_data_df = ade_data_df[ade_data_df.ade_num_affected >= 0].reset_index(drop=True)

    # We only focus on the SOC level in MedDRA, i.e. ade_organ_system.
    # ADEs are reported at lower levels.
    # So we need to merge instances by ade_organ_system based on group_id
    # taking the maximum ade_num_affected/ade_num_at_risk ratio.
    # The maximum ade_num_affected/ade_num_at_risk ratio is the safest upper bound of the ADE occurrence.
    ade_data_df["ade_ratio"] = (
        ade_data_df["ade_num_affected"] / ade_data_df["ade_num_at_risk"]
    )
    groups = ade_data_df.groupby(["group_id", "ade_organ_system"])
    max_ratio_df = ade_data_df.loc[
        ade_data_df.groupby(["group_id", "ade_organ_system"])["ade_ratio"].idxmax()
    ]
    ade_data_df = max_ratio_df.drop(columns=["ade_ratio"]).reset_index(drop=True)

    # Map and process drugs
    def process_name(name, fuzzy_aliases):
        try:
            return name, fuzzy_aliases[name]
        except KeyError:
            return name, None

    def worker_init(q):
        """Initialize the worker with a queue to send progress updates"""
        global queue
        queue = q

    def worker_main(name):
        """The main function that each worker will execute"""
        result = process_name(name, FuzzyDict(aliases, fuzzy_threshold=100))
        queue.put(1)  # Signal that a task is complete
        return result

    unique_names = list(ade_data_df.intervention_name.unique())
    num_processes = multiprocessing.cpu_count()

    # Create a multiprocessing queue
    manager = multiprocessing.Manager()
    queue = manager.Queue()

    # Create a pool of processes
    pool = multiprocessing.Pool(
        num_processes, initializer=worker_init, initargs=(queue,)
    )

    # Process the data
    result_objects = [
        pool.apply_async(worker_main, args=(name,)) for name in unique_names
    ]

    # Close the pool and wait for the work to finish
    pool.close()

    # Set up tqdm for progress updates
    pbar = tqdm(total=len(unique_names), desc="Mapping drugs")
    results_count = 0
    while True:
        queue.get()
        results_count += 1
        pbar.update()
        if results_count == len(unique_names):
            break
    pbar.close()

    # Retrieve the results
    results = [r.get() for r in result_objects]

    # Combine the results into a dictionary
    data_aliases = {name: alias for name, alias in results}

    pool.join()

    ade_data_df.insert(
        5,
        "intervention_name_drugbank",
        pd.DataFrame(ade_data_df.intervention_name.apply(lambda x: data_aliases[x])),
    )

    ade_data_df.loc[
        ade_data_df.is_placebo == 1, "intervention_name_drugbank"
    ] = "[PLACEBO]"

    def get_drug_attribute(drug, attribute):
        try:
            return drug_dict[drug][attribute]
        except KeyError:
            return "[PLACEBO]" if drug == "[PLACEBO]" else None

    attributes = ["type", "kingdom", "atc_anatomical_main_group", "smiles", "sequence"]

    for i, attr in enumerate(attributes, start=6):
        ade_data_df.insert(
            i,
            attr,
            pd.DataFrame(
                ade_data_df.intervention_name_drugbank.apply(
                    get_drug_attribute, attribute=attr
                )
            ),
        )

    ade_data_df = ade_data_df.dropna()

    # Pivoting the data for classification
    ade_data_df_pivot = pivot_ade_frame(input_data_df=ade_data_df)

    # Cleaning anomalies: Each combination of drug, criteria and group description should be unique.
    # If not, the original cts had non clean data.
    ade_data_df_pivot["combination"] = ade_data_df_pivot.apply(
        lambda x: "".join(
            [x["smiles"], x["eligibility_criteria"], x["group_description"]]
        ),
        axis=1,
    )
    duplicates = ade_data_df_pivot.duplicated("combination", keep=False)
    ade_data_df_pivot = ade_data_df_pivot[~duplicates]
    ade_data_df_pivot = ade_data_df_pivot.drop(columns=["combination"]).reset_index(
        drop=True
    )

    # Split the data
    train_ade_data_df, val_ade_data_df, test_ade_data_df = split_dataframe(
        ade_data_df_pivot,
        val_size=0.1,
        test_size=0.1,
        seed=42,
        unique_col="smiles",
        special_value="[PLACEBO]",
    )

    # and save the base version (all instances without smiles are deleted using `clean_ade_data`).
    path = os.path.join(".", "data", "classification", "smiles", "train_base")
    if not os.path.exists(path):
        os.makedirs(path)

    clean_ade_data(train_ade_data_df).to_csv(
        os.path.join(path, "train.csv"), index=False
    )
    clean_ade_data(val_ade_data_df).to_csv(
        os.path.join(path, "val.csv"), index=False
    )
    clean_ade_data(test_ade_data_df).to_csv(
        os.path.join(path, "test.csv"), index=False
    )

    # and save the augmented version (all instances without smiles are appended to the training set using `augment_train_df`).
    path = os.path.join(".", "data", "classification", "smiles", "train_augmented")
    if not os.path.exists(path):
        os.makedirs(path)

    train_ade_data_df, val_ade_data_df, test_ade_data_df = augment_train_df(
        train_ade_data_df, val_ade_data_df, test_ade_data_df
    )

    train_ade_data_df.to_csv(
        os.path.join(path, "train.csv"), index=False
    )
    val_ade_data_df.to_csv(
        os.path.join(path, "val.csv"), index=False
    )
    test_ade_data_df.to_csv(
        os.path.join(path, "test.csv"), index=False
    )
