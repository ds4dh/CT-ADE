import os
import json
from tqdm.auto import tqdm
import re
import pandas as pd
from werkzeug.utils import secure_filename
import scipy.stats as st
import math
import multiprocessing
from src.FuzzyDict import FuzzyDict
import regex
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def json_file_paths(folder_path):
    '''
    Gather all JSON file paths in the given folder.

    :param folder_path: The path to the folder containing JSON files.
    :return: A list of paths to JSON files.
    '''
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.json')]


def sanitize_number(s):
    # Remove non-numeric characters except for the decimal point and minus sign
    cleaned_number = re.sub(r'[^\d.-]', '', str(s))
    try:
        # Attempt to convert the cleaned string to an integer
        return int(cleaned_number)
    except ValueError:
        # If conversion fails, return None or raise an appropriate error
        return None


def get_arm_groups_summary(data):
    arm_groups_summary = []
    try:
        for arm_group in data['FullStudy']['Study']['ProtocolSection']['ArmsInterventionsModule']['ArmGroupList']['ArmGroup']:
            arm_groups_summary.append(
                {
                    'matching_values': [arm_group.get('ArmGroupLabel'), arm_group.get('ArmGroupDescription')],
                    'intervention_name': arm_group.get('ArmGroupInterventionList', {}).get('ArmGroupInterventionName')
                }
            )
    except KeyError:
        # Arm groups not properly defined
        arm_groups_summary.append(
            {
                'matching_values': [None, None],
                'intervention_name': None
            }
        )

    return arm_groups_summary


def check_for_match(list1, list2):
    # Create sets from the lists for efficient lookup, excluding None
    set1 = set(str(item).strip().lower() for item in list1 if item is not None)
    set2 = set(str(item).strip().lower() for item in list2 if item is not None)

    # Check for any common element in both sets
    for item in set1:
        if item in set2:
            return True

    return False


def check_strict_match(list1, list2):
    set1 = set(str(item).strip().lower() for item in list1 if item is not None)
    set2 = set(str(item).strip().lower() for item in list2 if item is not None)

    if len(set1) != len(set2) or len(set1) != len(list1):
        # If the lengths of the sets or the length of set1 and list1 differ,
        # it means there are duplicates or None values, so strict matching is not possible
        return False

    matched = set()
    for item1 in set1:
        match_found = False
        for item2 in set2:
            if item1 == item2 and item2 not in matched:
                match_found = True
                matched.add(item2)
                break
        if not match_found:
            return False

    return True


def find_intervention_name(ADE, arm_groups_summary):

    intervention_name = None
    got_match = 0

    ade_matching_values = [ADE['title'], ADE['group_description']]

    # First pass with regular matching
    for arm_group in arm_groups_summary:
        if check_for_match(ade_matching_values, arm_group['matching_values']):
            intervention_name = arm_group['intervention_name']
            got_match += 1

    # If more than one match found, apply stricter matching criteria
    if got_match > 1:
        intervention_name = None
        got_match = 0
        for arm_group in arm_groups_summary:
            if check_strict_match(ade_matching_values, arm_group['matching_values']):
                intervention_name = arm_group['intervention_name']
                got_match += 1

    if intervention_name and got_match <= 1:
        intervention_name = [str(name) for name in intervention_name]
        return intervention_name
    else:
        return None


def get_AdverseEvents(data):

    ade_groups = {}

    try:
        event_groups = data['FullStudy']['Study']['ResultsSection']['AdverseEventsModule']['EventGroupList']['EventGroup']
    except KeyError:
        # Handle missing EventGroupList or EventGroup
        return ade_groups

    for event_group in event_groups:

        # Summary
        group_key = event_group.get('EventGroupId')
        ade_groups[group_key] = {
            'title': event_group.get('EventGroupTitle'),
            'group_description': event_group.get('EventGroupDescription'),
            'serious_population': sanitize_number(event_group.get('EventGroupSeriousNumAtRisk')),
            'other_population': sanitize_number(event_group.get('EventGroupOtherNumAtRisk'))
        }

        # Serious ADEs
        serious_events = []
        try:
            serious_event_list = data['FullStudy']['Study']['ResultsSection'][
                'AdverseEventsModule']['SeriousEventList']['SeriousEvent']
            for serious_event in serious_event_list:
                for stat in serious_event['SeriousEventStatsList']['SeriousEventStats']:
                    if stat['SeriousEventStatsGroupId'] == group_key:
                        serious_events.append({
                            'ade_organ_system': serious_event.get('SeriousEventOrganSystem'),
                            'ade_num_affected': sanitize_number(stat.get('SeriousEventStatsNumAffected')),
                            'ade_num_at_risk': sanitize_number(stat.get('SeriousEventStatsNumAtRisk'))
                        })
        except KeyError:
            pass

        ade_groups[group_key]['serious_events'] = serious_events

        # Other ADEs
        other_events = []
        try:
            other_event_list = data['FullStudy']['Study']['ResultsSection']['AdverseEventsModule']['OtherEventList']['OtherEvent']
            for other_event in other_event_list:
                for stat in other_event['OtherEventStatsList']['OtherEventStats']:
                    if stat['OtherEventStatsGroupId'] == group_key:
                        other_events.append({
                            'ade_organ_system': other_event.get('OtherEventOrganSystem'),
                            'ade_num_affected': sanitize_number(stat.get('OtherEventStatsNumAffected')),
                            'ade_num_at_risk': sanitize_number(stat.get('OtherEventStatsNumAtRisk'))
                        })
        except KeyError:
            pass

        ade_groups[group_key]['other_events'] = other_events

    return ade_groups


def get_EligibilityCriteria(data):
    try:
        return data['FullStudy']['Study']['ProtocolSection']['EligibilityModule']['EligibilityCriteria']
    except KeyError:
        return None


def get_NCTId(data):
    try:
        return data['FullStudy']['Study']['ProtocolSection']['IdentificationModule']['NCTId']
    except KeyError:
        return None


def check_if_all_placebos(intervention_names):
    return int(all('placebo' in name.strip().lower() for name in intervention_names))


def check_at_least_placebos(intervention_names):
    return int(any('placebo' in name.strip().lower() for name in intervention_names))


def parse_trial(data):

    NCTId = get_NCTId(data)
    EligibilityCriteria = get_EligibilityCriteria(data)

    ADEs = []  # Combined list for all ADEs

    adverse_events = get_AdverseEvents(data)

    arm_groups_summary = get_arm_groups_summary(data)

    for group_code, ADE in adverse_events.items():

        intervention_name = find_intervention_name(ADE, arm_groups_summary)
        if intervention_name is None:
            continue

        is_placebo = check_if_all_placebos(intervention_name)
        contain_placebo = check_at_least_placebos(intervention_name)

        # Removed serious_population and other_population as no placeholders are required

        common_data = {
            'nctid': NCTId,
            'group_id': NCTId + '_' + group_code,
            'eligibility_criteria': EligibilityCriteria,
            'title': ADE['title'],
            'intervention_name': intervention_name,
            'group_description': ADE['group_description'],
            'is_placebo': is_placebo,
            'contain_placebo': contain_placebo
        }

        # Process serious events
        if ADE['serious_events']:
            for event in ADE['serious_events']:
                row = {**common_data, **event}
                ADEs.append(row)

        # Process other events
        if ADE['other_events']:
            for event in ADE['other_events']:
                row = {**common_data, **event}
                ADEs.append(row)

    return ADEs


def process_json_file(file_path):
    with open(file_path, 'r') as file:
        data = parse_trial(json.load(file))
    return data


def parallel_process_json_files(json_files):
    num_cores = multiprocessing.cpu_count()  # Get the number of CPU cores
    # Create a multiprocessing pool
    pool = multiprocessing.Pool(processes=num_cores)
    results = list(tqdm(pool.imap(process_json_file, json_files),
                   total=len(json_files), desc='Parsing JSON Files'))
    pool.close()
    pool.join()
    return results


def wilson_score_lower_bound(pos, n, confidence):
    if n == 0:
        return 0, 0, 0

    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n

    denominator = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)

    lower_bound = (center - margin) / denominator

    return lower_bound


def sanitize_intervention_name(drug):
    drug = drug[0].split(":")[-1].strip()

    # List of common drug units, routes, and forms
    units = ['mg', 'ml', 'g', 'l', 'mcg', 'ug',
             'iu', 'u', 'mmol', 'nmol', 'pmol', 'kg']
    routes = ['oral', 'injection', 'intravenous', 'topical', 'nasal', 'subcutaneous',
              'intramuscular', 'inhalation', 'rectal', 'sublingual', 'transdermal',
              'intradermal', 'ophthalmic', 'otic', 'vaginal', 'buccal', 'enteral', 'parenteral',
              'intravitreal']
    forms = ['tablet', 'capsule', 'cream', 'ointment', 'gel', 'solution', 'suspension',
             'syrup', 'lozenge', 'patch', 'powder', 'aerosol', 'emulsion', 'spray',
             'drop', 'film', 'foam', 'implant', 'inhaler', 'insert', 'lotion', 'pessary', 'suppository']

    # Create regular expression patterns
    unit_pattern = r'\b\d*\.?\d+\s*(?:' + '|'.join(units) + \
        ')(?:\s*[-/]\s*(?:' + '|'.join(units) + '))?'
    percentage_pattern = r'\b\d+(\.\d+)?%'
    route_pattern = r'\b(?:' + '|'.join(routes) + r')(?:\s|$)'
    form_pattern = r'\b(?:' + '|'.join(forms) + r')(?:\s|$)'
    special_char_pattern = r'\p{S}'  # Matches any symbol

    # Apply regular expressions
    drug = regex.sub(r'\[.*?\]|\(.*?\)', '', drug)
    drug = regex.sub(unit_pattern, '', drug, flags=regex.IGNORECASE)
    drug = regex.sub(percentage_pattern, '', drug)
    drug = regex.sub(route_pattern, '', drug, flags=regex.IGNORECASE)
    drug = regex.sub(form_pattern, '', drug, flags=regex.IGNORECASE)
    drug = regex.sub(special_char_pattern, '', drug)

    drug = ' '.join([word for word in drug.split()
                    if word.lower() not in stop_words])

    # Replace multiple white spaces with a single space
    drug = regex.sub(r'\s+', ' ', drug)

    drug = drug.strip()
    return drug


def pivot_ade_frame(input_data_df):

    label_names = ['Blood and lymphatic system disorders',
                   'Cardiac disorders',
                   'Congenital, familial and genetic disorders',
                   'Ear and labyrinth disorders',
                   'Endocrine disorders',
                   'Eye disorders',
                   'Gastrointestinal disorders',
                   'General disorders and administration site conditions',
                   'Hepatobiliary disorders',
                   'Immune system disorders',
                   'Infections and infestations',
                   'Injury, poisoning and procedural complications',
                   'Investigations',
                   'Metabolism and nutrition disorders',
                   'Musculoskeletal and connective tissue disorders',
                   'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                   'Nervous system disorders',
                   'Pregnancy, puerperium and perinatal conditions',
                   'Psychiatric disorders',
                   'Renal and urinary disorders',
                   'Reproductive system and breast disorders',
                   'Respiratory, thoracic and mediastinal disorders',
                   'Skin and subcutaneous tissue disorders',
                   'Social circumstances',
                   'Surgical and medical procedures',
                   'Vascular disorders',
                   'Product issues']

    data_df = input_data_df.copy()

    label_names_columns = [secure_filename(
        label_name.strip().lower()) for label_name in label_names]
    data_df.ade_organ_system = data_df.ade_organ_system.apply(
        lambda x: secure_filename(x.strip().lower()))
    data_df_meddra = data_df[data_df.ade_organ_system.apply(
        lambda x: x in label_names_columns)]

    # Group the data by 'group_id'
    groups = data_df_meddra.groupby('group_id')
    part_a_list = []  # List to store each 'part_a' DataFrame

    # Iterate over each group
    for group_id, group_data in tqdm(groups, desc='Pivoting ADE table'):
        # Extract part_a by dropping specific columns and selecting the first row
        part_a = group_data.drop(
            columns=['ade_organ_system', 'ade_num_affected', 'ade_num_at_risk']).iloc[0, :]
        part_a = part_a.to_frame().T

        # Extract part_b with specific columns
        part_b = group_data[['ade_organ_system',
                             'ade_num_affected', 'ade_num_at_risk']]

        # Calculate values using the wilson_score_lower_bound function
        values = [float(wilson_score_lower_bound(pos=i, n=j, confidence=0.99) > 0)
                  for i, j in zip(part_b['ade_num_affected'], part_b['ade_num_at_risk'])]

        # Create a dictionary of true occurrences
        true_occurences = {k: v for k, v in zip(
            part_b['ade_organ_system'], values)}

        # Initialize columns in part_a with zeros
        part_a[label_names_columns] = [0.0] * len(label_names_columns)

        # Update part_a columns based on true occurrences
        for col in part_a:
            if col in true_occurences:
                part_a[col] = true_occurences[col]

        # Append the processed part_a DataFrame to the list
        part_a_list.append(part_a)

    return pd.concat(part_a_list, ignore_index=True)


def split_dataframe(df, val_size, test_size, seed, unique_col=None, special_value=None):
    """
    Splits a DataFrame into train, validation, and test sets, ensuring a specific value is distributed 
    across all splits in the same proportions as the main DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to split.
    val_size (float): The size of the validation set (as a fraction).
    test_size (float): The size of the test set (as a fraction).
    seed (int): Random seed.
    unique_col (str, optional): Column name to enforce unique values across splits. Defaults to None.
    special_value (any, optional): A specific value in the unique_col to be distributed in all sets. Defaults to None.

    Returns:
    train_df (DataFrame): The training set.
    val_df (DataFrame): The validation set.
    test_df (DataFrame): The test set.
    """

    if special_value is not None and unique_col in df.columns:
        # Separate out the special rows
        special_df = df[df[unique_col] == special_value]
        df = df[df[unique_col] != special_value]

        # Split the special rows
        train_val_special, test_special = train_test_split(
            special_df, test_size=test_size, random_state=seed)
        train_special, val_special = train_test_split(
            train_val_special, test_size=val_size / (1 - test_size), random_state=seed)
    else:
        train_special, val_special, test_special = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if unique_col and unique_col in df.columns:
        # Split based on unique values in the specified column
        unique_values = df[unique_col].unique()
        train_val_values, test_values = train_test_split(
            unique_values, test_size=test_size, random_state=seed)
        train_values, val_values = train_test_split(
            train_val_values, test_size=val_size / (1 - test_size), random_state=seed)

        train_df = df[df[unique_col].isin(train_values)]
        val_df = df[df[unique_col].isin(val_values)]
        test_df = df[df[unique_col].isin(test_values)]
    else:
        # Standard split without considering unique values in a column
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=seed)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size / (1 - test_size), random_state=seed)

    # Add the special value rows back into each set
    train_df = pd.concat([train_df, train_special]).reset_index(drop=True)
    val_df = pd.concat([val_df, val_special]).reset_index(drop=True)
    test_df = pd.concat([test_df, test_special]).reset_index(drop=True)

    return train_df, val_df, test_df


def clean_ade_data(df):
    """
    Cleans the ADE data by deleting rows where 'smiles' is either '[PLACEBO]' or '[NOSMILES]'.

    Parameters:
    df (pd.DataFrame): The ADE data DataFrame to be cleaned.

    Returns:
    pd.DataFrame: The cleaned ADE data DataFrame.
    """
    df_ = df.copy()

    # Define the condition to delete the rows
    condition_to_delete = (df_['smiles'] == '[PLACEBO]') | (
        df_['smiles'] == '[NOSMILES]')

    # Apply the condition to delete rows and reset the index
    cleaned_df = df_[~condition_to_delete].reset_index(drop=True)

    return cleaned_df


def augment_train_df(train_df, val_df, test_df, smiles_col='smiles'):
    """
    Transfer all non-valid instances (no smiles or placebo groups) to the train set.
    """
    # Step 1: Identify rows where `smiles` is "[PLACEBO]" or "[NOSMILES]" in val and test datasets
    val_placebo_nosmiles = val_df[val_df[smiles_col].isin(
        ["[PLACEBO]", "[NOSMILES]"])]
    test_placebo_nosmiles = test_df[test_df[smiles_col].isin(
        ["[PLACEBO]", "[NOSMILES]"])]

    # Step 2: Create a new train data DataFrame with these rows appended
    new_train_df = pd.concat(
        [train_df, val_placebo_nosmiles, test_placebo_nosmiles], ignore_index=True)

    # Step 3: Create new validation and test data DataFrames without these rows
    new_val_df = val_df[~val_df[smiles_col].isin(["[PLACEBO]", "[NOSMILES]"])]
    new_test_df = test_df[~test_df[smiles_col].isin(
        ["[PLACEBO]", "[NOSMILES]"])]

    # Step 4: Reset the index for the new datasets
    new_train_df.reset_index(drop=True, inplace=True)
    new_val_df.reset_index(drop=True, inplace=True)
    new_test_df.reset_index(drop=True, inplace=True)

    return new_train_df, new_val_df, new_test_df