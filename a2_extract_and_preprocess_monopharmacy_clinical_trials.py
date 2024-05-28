import copy
import json
import multiprocessing
from tqdm.auto import tqdm
from pathlib import Path
import re
from typing import Optional, Dict, Any, List, Tuple, Union


def get_NCTID(ct_data: Dict) -> Optional[str]:
    """
    Extracts the NCT ID from clinical trial data.

    Args:
        ct_data (Dict): The dictionary containing clinical trial information.

    Returns:
        Optional[str]: The NCT ID if found, otherwise None.
    """
    try:
        return ct_data["ProtocolSection"]["IdentificationModule"]["NCTId"]
    except:
        return None


def get_EligibilityCriteria(ct_data: Dict) -> Optional[str]:
    """
    Extracts the eligibility criteria from clinical trial data.

    Args:
        ct_data (Dict): The dictionary containing clinical trial information.

    Returns:
        Optional[str]: The eligibility criteria text if found, otherwise None.
    """
    try:
        return ct_data["ProtocolSection"]["EligibilityModule"]["EligibilityCriteria"]
    except:
        return None


def get_BriefTitle(ct_data: Dict) -> Optional[str]:
    """
    Extracts the brief title of a clinical trial from the trial data.

    Args:
        ct_data (Dict): The dictionary containing clinical trial information.

    Returns:
        Optional[str]: The brief title if found, otherwise None.
    """
    return (
        ct_data.get("ProtocolSection", {})
        .get("IdentificationModule", {})
        .get("BriefTitle", None)
    )


def get_OverallStatus(ct_data: Dict) -> Optional[str]:
    """
    Retrieves the overall status of the clinical trial.

    Args:
        ct_data (Dict): The dictionary containing clinical trial information.

    Returns:
        Optional[str]: The overall status if found, otherwise None.
    """
    return (
        ct_data.get("ProtocolSection", {})
        .get("StatusModule", {})
        .get("OverallStatus", None)
    )


def get_LeadSponsorName(ct_data: Dict) -> Optional[str]:
    """
    Gets the name of the lead sponsor of the clinical trial.

    Args:
        ct_data (Dict): The dictionary containing clinical trial information.

    Returns:
        Optional[str]: The lead sponsor name if available, otherwise None.
    """
    return (
        ct_data.get("ProtocolSection", {})
        .get("SponsorCollaboratorsModule", {})
        .get("LeadSponsor", {})
        .get("LeadSponsorName", None)
    )


def get_Collaborator(ct_data: Dict) -> Optional[str]:
    """
    Retrieves a concatenated string of all collaborator names involved in the clinical trial.

    Args:
        ct_data (Dict): The dictionary containing clinical trial information.

    Returns:
        Optional[str]: A string of collaborator names separated by " | ", or None if no collaborators are listed.
    """
    collaborator_list = (
        ct_data.get("ProtocolSection", {})
        .get("SponsorCollaboratorsModule", {})
        .get("CollaboratorList", {})
        .get("Collaborator", [])
    )
    if collaborator_list:
        collaborator_name_list = [list(i.values())[0] for i in collaborator_list]
        return " | ".join(collaborator_name_list)
    return None


def get_HealthyVolunteers(ct_data: Dict) -> Optional[str]:
    """
    Indicates whether healthy volunteers are accepted in the clinical trial.

    Args:
        ct_data (Dict): The dictionary containing clinical trial information.

    Returns:
        Optional[str]: The status of accepting healthy volunteers ("Yes" or "No"), otherwise None.
    """
    return (
        ct_data.get("ProtocolSection", {})
        .get("EligibilityModule", {})
        .get("HealthyVolunteers", None)
    )


def get_Gender(ct_data: Dict) -> Optional[str]:
    """
    Retrieves the gender eligibility for the clinical trial.

    Args:
        ct_data (Dict): The dictionary containing clinical trial information.

    Returns:
        Optional[str]: The gender requirement if specified ("All", "Male", "Female"), otherwise None.
    """
    return (
        ct_data.get("ProtocolSection", {})
        .get("EligibilityModule", {})
        .get("Gender", None)
    )


def get_stdAge(ct_data: Dict) -> Optional[str]:
    """
    Retrieves the standard age range(s) for eligibility in the clinical trial.

    Args:
        ct_data (Dict): The dictionary containing clinical trial information.

    Returns:
        Optional[str]: A concatenated string of age ranges separated by " | ", or None if no specific age ranges are specified.
    """
    StdAge = (
        ct_data.get("ProtocolSection", {})
        .get("EligibilityModule", {})
        .get("StdAgeList", {})
        .get("StdAge", [])
    )
    if StdAge:
        return " | ".join(StdAge)
    return None


def get_Phase(ct_data: Dict) -> Optional[str]:
    """
    Retrieves the phase(s) of the clinical trial.

    Args:
        ct_data (Dict): The dictionary containing clinical trial information.

    Returns:
        Optional[str]: A concatenated string of phases separated by " | ", or None if no phases are specified.
    """
    Phase = (
        ct_data.get("ProtocolSection", {})
        .get("DesignModule", {})
        .get("PhaseList", {})
        .get("Phase", [])
    )
    if Phase:
        return " | ".join(Phase)
    return None


def get_EnrollmentCount(ct_data: Dict) -> Optional[int]:
    """
    Retrieves the enrollment count from the clinical trial data, converting it to an integer if present.

    Args:
        ct_data (Dict): The dictionary containing clinical trial information.

    Returns:
        Optional[int]: The enrollment count as an integer if the data is present and valid, otherwise None.
    """
    EnrollmentCount = (
        ct_data.get("ProtocolSection", {})
        .get("DesignModule", {})
        .get("EnrollmentInfo", {})
        .get("EnrollmentCount", None)
    )
    if EnrollmentCount:
        try:
            return int(EnrollmentCount)
        except ValueError:
            return None
    return None


def get_ct_details(ct_data: Dict) -> Dict[str, Optional[Union[str, int]]]:
    """
    Aggregates various details from clinical trial data into a single dictionary.

    Args:
        ct_data (Dict): The dictionary containing clinical trial information.

    Returns:
        Dict[str, Optional[Union[str, int]]]: A dictionary containing key details of the clinical trial.
    """
    details = {
        "title": get_BriefTitle(ct_data),
        "status": get_OverallStatus(ct_data),
        "sponsor": get_LeadSponsorName(ct_data),
        "collaborators": get_Collaborator(ct_data),
        "healthy_volunteers": get_HealthyVolunteers(ct_data),
        "gender": get_Gender(ct_data),
        "age": get_stdAge(ct_data),
        "phase": get_Phase(ct_data),
        "enrollment_count": get_EnrollmentCount(ct_data),
    }
    return details


def extract_interventions(ct_data: Dict) -> Optional[List[Dict]]:
    """
    Extracts a list of interventions from clinical trial data.

    Args:
        ct_data (Dict): The dictionary containing clinical trial information.

    Returns:
        Optional[List[Dict]]: A list of interventions, each as a dictionary; returns None if no interventions are found.
    """
    try:
        # Access the intervention list from the nested dictionary structure
        interventions = ct_data["ProtocolSection"]["ArmsInterventionsModule"][
            "InterventionList"
        ]["Intervention"]
    except KeyError:
        # Return None if the structure does not contain an InterventionList or Intervention
        return None

    # List to store extracted interventions
    ct_interventions = []

    # Loop through each intervention in the data
    for intervention in interventions:
        # Extract the intervention type and name
        intervention_type = intervention.get("InterventionType")
        intervention_name = intervention.get("InterventionName", "")

        # Extract other names for the intervention, if available
        synonyms = intervention.get("InterventionOtherNameList", {}).get(
            "InterventionOtherName", []
        )

        # Ensure synonyms is a list, if not, make it a list
        if not isinstance(synonyms, list):
            synonyms = [synonyms] if synonyms else []

        # Ensure the intervention name is included in the synonyms
        if intervention_name and intervention_name not in synonyms:
            synonyms.append(intervention_name)

        # Create a dictionary with the intervention details
        intervention_details = {
            "type": intervention_type,
            "name": intervention_name,
            "synonyms": synonyms,
        }

        # Add the intervention details to the list
        ct_interventions.append(intervention_details)

    return ct_interventions if ct_interventions else None


def clean_synonyms(data: List[Dict]) -> List[Dict]:
    """
    Removes synonyms that map to more than one intervention name from the data.

    Args:
        data (List[Dict]): A list of dictionaries, each representing an intervention with a list of synonyms.

    Returns:
        List[Dict]: The updated list of interventions with cleaned synonyms.
    """
    data_copy = copy.deepcopy(data)
    synonym_map = {}

    # Map each synonym to the names it's associated with
    for item in data_copy:
        name = item["name"]
        for synonym in item["synonyms"]:
            if synonym not in synonym_map:
                synonym_map[synonym] = set()
            synonym_map[synonym].add(name)

    # Find all synonyms associated with more than one name
    synonyms_to_remove = {
        syn: names for syn, names in synonym_map.items() if len(names) > 1
    }

    # Clean data by removing problematic synonyms
    for item in data_copy:
        original_synonyms = item["synonyms"]
        cleaned_synonyms = [
            syn for syn in original_synonyms if syn not in synonyms_to_remove
        ]
        item["synonyms"] = cleaned_synonyms

    return data_copy


def split_string_at_first_colon(input_string: str) -> Tuple[str, str]:
    """
    Splits the input string at the first occurrence of a colon.

    Args:
        input_string (str): The string to split.

    Returns:
        Tuple[str, str]: A tuple containing the parts of the string before and after the first colon.
    """
    colon_index = input_string.index(":")
    before_colon = input_string[:colon_index]
    after_colon = input_string[colon_index + 1 :]
    return before_colon, after_colon


def extract_arm_groups_with_synonyms(
    ct_data: Dict, ct_interventions: List[Dict]
) -> Optional[List[Dict]]:
    """
    Extracts arm groups and maps intervention synonyms from clinical trial data.

    Args:
        ct_data (Dict): The dictionary containing clinical trial information.
        ct_interventions (List[Dict]): A list of cleaned intervention data.

    Returns:
        Optional[List[Dict]]: A list of arm groups with intervention synonyms if available, otherwise None.
    """
    try:
        ArmGroup = ct_data["ProtocolSection"]["ArmsInterventionsModule"][
            "ArmGroupList"
        ]["ArmGroup"]
        ct_arms_groups = []

        for arm_group in ArmGroup:
            arm_group_temp = copy.deepcopy(arm_group)
            intervention_names = arm_group.get("ArmGroupInterventionList", {}).get(
                "ArmGroupInterventionName", []
            )
            arm_group_intervention_synonyms = []

            for intervention_name in intervention_names:
                _, after_colon = split_string_at_first_colon(intervention_name)
                after_colon_clean = after_colon.lower().strip()
                for candidate_interventions in ct_interventions:
                    if after_colon_clean in [
                        i.lower().strip() for i in candidate_interventions["synonyms"]
                    ]:
                        arm_group_intervention_synonyms.append(
                            {intervention_name: candidate_interventions["synonyms"]}
                        )

            arm_group_temp["synonyms"] = arm_group_intervention_synonyms
            ct_arms_groups.append(arm_group_temp)

        total_arm_groups = len(ArmGroup)
        mapped_arm_groups = len(ct_arms_groups)

        if ct_arms_groups:
            return ct_arms_groups
        else:
            return None
    except KeyError:
        return None


def remove_duplicate_dicts(data: List[Dict]) -> List[Dict]:
    """
    Removes duplicate dictionaries from a list based on their JSON string representation.

    Args:
        data (List[Dict]): The list of dictionaries from which to remove duplicates.

    Returns:
        List[Dict]: A list containing only unique dictionaries.
    """
    # Use a set to track unique dictionary entries based on their JSON string representation
    unique_dicts = set()
    # List to hold the final unique dictionaries
    unique_data = []

    for entry in data:
        # Convert dictionary to a JSON string
        entry_str = json.dumps(entry, sort_keys=True)
        # Check if the JSON string is not already in the set
        if entry_str not in unique_dicts:
            unique_dicts.add(entry_str)
            unique_data.append(entry)

    return unique_data


def sanitize_number(s: Any) -> Optional[int]:
    """
    Attempts to sanitize and convert a string to an integer, removing non-numeric characters except for the decimal point and minus sign.

    Args:
        s (Any): The input string to sanitize.

    Returns:
        Optional[int]: The sanitized number as an integer if conversion is successful, otherwise None.
    """
    # Remove non-numeric characters except for the decimal point and minus sign
    cleaned_number = re.sub(r"[^\d.-]", "", str(s))
    try:
        # Attempt to convert the cleaned string to an integer
        return int(cleaned_number)
    except ValueError:
        # If conversion fails, return None or raise an appropriate error
        return None


def get_AdverseEvents(ct_data: Dict) -> Optional[Dict]:
    """
    Extracts adverse event data from clinical trial information.

    Args:
        ct_data (Dict): The dictionary containing clinical trial information.

    Returns:
        Optional[Dict]: A dictionary of adverse events grouped by their IDs, or None if no data is found.
    """
    ade_groups = {}

    try:
        event_groups = ct_data["ResultsSection"]["AdverseEventsModule"][
            "EventGroupList"
        ]["EventGroup"]
    except KeyError:
        # Handle missing EventGroupList or EventGroup
        return None

    for event_group in event_groups:

        # Serious ADEs
        serious_events = []
        try:
            group_key = event_group["EventGroupId"]
            serious_event_list = ct_data["ResultsSection"]["AdverseEventsModule"][
                "SeriousEventList"
            ]["SeriousEvent"]
            for serious_event in serious_event_list:
                for stat in serious_event["SeriousEventStatsList"]["SeriousEventStats"]:
                    if stat["SeriousEventStatsGroupId"] == group_key:
                        serious_events.append(
                            {
                                "ade_vocabulary": serious_event.get(
                                    "SeriousEventSourceVocabulary"
                                ),
                                "ade_term": serious_event.get(
                                    "SeriousEventTerm"
                                ),
                                "ade_organ_system": serious_event.get(
                                    "SeriousEventOrganSystem"
                                ),
                                "ade_num_affected": sanitize_number(
                                    stat.get("SeriousEventStatsNumAffected")
                                ),
                                "ade_num_at_risk": sanitize_number(
                                    stat.get("SeriousEventStatsNumAtRisk")
                                ),
                            }
                        )
        except KeyError:
            pass

        # Other ADEs
        other_events = []
        try:
            group_key = event_group["EventGroupId"]
            other_event_list = ct_data["ResultsSection"]["AdverseEventsModule"][
                "OtherEventList"
            ]["OtherEvent"]
            for other_event in other_event_list:
                for stat in other_event["OtherEventStatsList"]["OtherEventStats"]:
                    if stat["OtherEventStatsGroupId"] == group_key:
                        other_events.append(
                            {
                                "ade_vocabulary": other_event.get(
                                    "OtherEventSourceVocabulary"
                                ),
                                "ade_term": other_event.get("OtherEventTerm"),
                                "ade_organ_system": other_event.get(
                                    "OtherEventOrganSystem"
                                ),
                                "ade_num_affected": sanitize_number(
                                    stat.get("OtherEventStatsNumAffected")
                                ),
                                "ade_num_at_risk": sanitize_number(
                                    stat.get("OtherEventStatsNumAtRisk")
                                ),
                            }
                        )
        except KeyError:
            pass

        # Summary
        try:
            group_key = event_group["EventGroupId"]
            ade_groups[group_key] = {
                "title": event_group["EventGroupTitle"],
                "group_description": event_group["EventGroupDescription"],
            }
            ade_groups[group_key]["serious_events"] = serious_events
            ade_groups[group_key]["other_events"] = other_events
        except KeyError:
            continue

    if ade_groups:
        return ade_groups
    else:
        return None


def check_for_match(list1: List[str], list2: List[str]) -> bool:
    """
    Checks for any common element between two lists.

    Args:
        list1 (List[str]): The first list of strings.
        list2 (List[str]): The second list of strings.

    Returns:
        bool: True if there is at least one common element, otherwise False.
    """
    # Create sets from the lists for efficient lookup, excluding None
    set1 = set(str(item).strip().lower() for item in list1 if item is not None)
    set2 = set(str(item).strip().lower() for item in list2 if item is not None)

    # Check for any common element in both sets
    for item in set1:
        if item in set2:
            return True

    return False


def check_strict_match(list1: List[str], list2: List[str]) -> bool:
    """
    Checks for a strict match between two lists, where all elements must match exactly without duplicates.

    Args:
        list1 (List[str]): The first list of strings.
        list2 (List[str]): The second list of strings.

    Returns:
        bool: True if the lists strictly match, otherwise False.
    """
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


def find_arm_group_for_ade(ADE: Dict, ArmGroup: List[Dict]) -> Optional[Dict]:
    """
    Attempts to match an ADE to an arm group based on their descriptions.

    Args:
        ADE (Dict): The adverse event group data.
        ArmGroup (List[Dict]): A list of arm group data.

    Returns:
        Optional[Dict]: The matching arm group if found, otherwise None.
    """
    # Initialize the variables to track the matching arm group and count of matches
    matching_arm_group = None
    matches_found = 0

    # Prepare a list of tuples where each tuple contains the arm group and its matching values
    arm_groups_summary = []
    try:
        for arm_group in ArmGroup:
            matching_values = [
                arm_group.get("ArmGroupLabel"),
                arm_group.get("ArmGroupDescription"),
            ]
            # Append the tuple of the original arm group and its matching values
            arm_groups_summary.append((arm_group, matching_values))
    except KeyError:
        # Handle cases where the keys might be missing
        arm_groups_summary.append(
            ({"ArmGroupLabel": None, "ArmGroupDescription": None}, [None, None])
        )

    # Extract matching values from ADE
    ade_matching_values = [ADE["title"], ADE["group_description"]]

    # First pass with regular matching
    for arm_group, values in arm_groups_summary:
        if check_for_match(ade_matching_values, values):
            if not matching_arm_group:
                matching_arm_group = arm_group  # Store the original arm group
            matches_found += 1

    # Apply stricter matching criteria if more than one match is found
    if matches_found > 1:
        matching_arm_group = None
        matches_found = 0
        for arm_group, values in arm_groups_summary:
            if check_strict_match(ade_matching_values, values):
                matching_arm_group = arm_group  # Store the original arm group
                matches_found += 1

    # Ensure a single, unique match was found
    if matching_arm_group and matches_found == 1:
        return matching_arm_group
    else:
        return None


def is_all_zeroes(lst: List[int]) -> bool:
    """
    Checks if all elements in the list are zero.

    Args:
        lst (List[int]): The list to check.

    Returns:
        bool: True if all elements are zero, otherwise False.
    """
    return all(x == 0 for x in lst)


def is_all_ones(lst: List[int]) -> bool:
    """
    Checks if all elements in the list are one.

    Args:
        lst (List[int]): The list to check.

    Returns:
        bool: True if all elements are one, otherwise False.
    """
    return all(x == 1 for x in lst)


def group_resolution_preprocessing(ct_path: str) -> Tuple[str, Optional[Dict]]:
    """
    Processes a clinical trial JSON file to match adverse events to arm groups using group resolution., i.e.,
    matching arm groups to event groups.

    Args:
        ct_path (str): The path to the clinical trial JSON file.

    Returns:
        Tuple[str, Optional[Dict]]: A tuple containing the match status and the processed data, if available.
    """
    with open(ct_path, "r") as file:
        ct_data = json.load(file)["FullStudy"]["Study"]

    NCTID = get_NCTID(ct_data)
    if not NCTID:
        return ("no_NCTID", None)

    EligibilityCriteria = get_EligibilityCriteria(ct_data)
    if not EligibilityCriteria:
        return ("no_EligibilityCriteria", None)

    ct_interventions = extract_interventions(ct_data)
    if not ct_interventions:
        return ("no_ct_interventions", None)
    ct_interventions = clean_synonyms(ct_interventions)
    ct_interventions = remove_duplicate_dicts(ct_interventions)

    ArmGroup = extract_arm_groups_with_synonyms(ct_data, ct_interventions)
    if not ArmGroup:
        return ("no_ArmGroup_matched_to_interventions", None)

    adverse_events = get_AdverseEvents(ct_data)
    if not adverse_events:
        return ("no_adverse_events", None)

    trial_details = get_ct_details(ct_data)

    final_ct_output = {
        "nctid": NCTID,
        **trial_details,
        "eligibility_criteria": EligibilityCriteria,
        "study_groups": [],
    }

    arm_group_matching_log = [0] * len(adverse_events)
    counter = 0

    for group_code, ADE in adverse_events.items():
        matching_arm_group = find_arm_group_for_ade(ADE, ArmGroup)
        if matching_arm_group:
            try:
                study_group_key = f"{NCTID}_{group_code}"
                final_ct_output["study_groups"].append(
                    {
                        "group_code": study_group_key,
                        "intervention_details": {
                            "name": matching_arm_group["ArmGroupInterventionList"][
                                "ArmGroupInterventionName"
                            ],
                            "synonyms": matching_arm_group.get("synonyms", []),
                            "description": ADE["group_description"],
                        },
                        "adverse_events": {
                            "serious_events": ADE.get("serious_events", []),
                            "other_events": ADE.get("other_events", []),
                        },
                    }
                )
            except KeyError:
                arm_group_matching_log[counter] = 1
            counter += 1
        else:
            arm_group_matching_log[counter] = 1
            counter += 1

    if is_all_zeroes(arm_group_matching_log):
        return ("fully_matched", final_ct_output)
    elif not is_all_ones(arm_group_matching_log):
        return ("partially_matched", final_ct_output)
    else:
        return ("no_matching_arm_group_at_all", None)


def is_drug_group_resolution(x: str) -> bool:
    """
    Determines if the intervention details suggest a drug group that does not include a placebo.

    Args:
        x (str): The intervention detail string.

    Returns:
        bool: True if it's a drug group without a placebo, otherwise False.
    """
    return x.startswith("drug") and "placebo" not in x


def is_monopharmacy_group_resolution(row: List[str]) -> bool:
    """
    Determines if the intervention details suggest a monopharmacy group.

    Args:
        row (List[str]): A list containing intervention details.

    Returns:
        bool: True if it's a monopharmacy group, otherwise False.
    """
    lst = [i.strip().lower() for i in row]
    drug_count = sum([is_drug_group_resolution(x) for x in lst])
    if len(lst) == 1 and drug_count == 1:
        return True
    else:
        return False


def is_placebo_simple_preprocessing(x: Dict) -> bool:
    """
    Checks if the intervention includes a placebo.

    Args:
        x (Dict): The intervention details.

    Returns:
        bool: True if the intervention is a placebo, otherwise False.
    """
    return "placebo" in x["name"].lower().strip()


def is_drug_simple_preprocessing(x: Dict) -> bool:
    """
    Checks if the intervention is a drug and does not include a placebo.

    Args:
        x (Dict): The intervention details.

    Returns:
        bool: True if the intervention is a drug and not a placebo, otherwise False.
    """
    return (
        x["type"].lower().strip() == "drug"
        and "placebo" not in x["name"].lower().strip()
    )


def simple_preprocessing(ct_path: str) -> Tuple[str, Optional[Dict]]:
    """
    Simple preprocessing function for clinical trial data, focusing on matching drug interventions to adverse events, i.e.,
    we process clinical trials that have only one drug across all interventions. This allows to bypass the matching between
    arm groups and event groups.

    Args:
        ct_path (str): The path to the clinical trial JSON file.

    Returns:
        Tuple[str, Optional[Dict]]: A tuple containing the processing result type and the processed data, if any.
    """
    final_ct_output = None  # Initialize to None

    with open(ct_path, "r") as file:
        ct_data = json.load(file)["FullStudy"]["Study"]

    NCTID = get_NCTID(ct_data)
    if not NCTID:
        return ("no_NCTID", None)

    EligibilityCriteria = get_EligibilityCriteria(ct_data)
    if not EligibilityCriteria:
        return ("no_EligibilityCriteria", None)

    ct_interventions = extract_interventions(ct_data)
    if not ct_interventions:
        return ("no_ct_interventions", None)
    ct_interventions = clean_synonyms(ct_interventions)
    ct_interventions = remove_duplicate_dicts(ct_interventions)

    adverse_events = get_AdverseEvents(ct_data)
    if not adverse_events:
        return ("no_adverse_events", None)

    trial_details = get_ct_details(ct_data)

    placebo_count = 0
    drug_count = 0
    selected_intervention = None
    drug_clean_intervention = ""
    for intervention in ct_interventions:
        try:
            if is_placebo_simple_preprocessing(intervention):
                placebo_count += 1
            elif is_drug_simple_preprocessing(intervention):
                drug_count += 1
                drug_clean_intervention = intervention["name"].lower().strip()
                selected_intervention = intervention
        except KeyError:
            placebo_count, drug_count = None, None

    matches = []
    try:
        if drug_count == 1:
            if adverse_events and (len(adverse_events) == placebo_count + drug_count):
                for (
                    ade_group_key,
                    ade_group_value,
                ) in (
                    adverse_events.items()
                ):  # ensure 'ade_groups' is meant to be 'adverse_events'
                    title = ade_group_value["title"].lower().strip()
                    group_description = (
                        ade_group_value["group_description"].lower().strip()
                    )
                    if (
                        (drug_clean_intervention in title) and ("placebo" not in title)
                    ) or (
                        (drug_clean_intervention in group_description)
                        and ("placebo" not in group_description)
                    ):
                        matches.append(
                            (ade_group_key, ade_group_value, selected_intervention)
                        )
    except KeyError:
        matches = []

    if matches and len(matches) == 1:
        match = matches[0]
        final_ct_output = {
            "nctid": NCTID,
            **trial_details,
            "eligibility_criteria": EligibilityCriteria,
            "study_groups": [
                {
                    "group_code": f"{NCTID}_{match[0]}",
                    "intervention_details": {
                        "name": [f"{match[2]['type']}: {match[2]['name']}"],
                        "synonyms": match[2].get("synonyms", []),
                        "description": match[1]["group_description"],
                    },
                    "adverse_events": {
                        "serious_events": match[1].get("serious_events", []),
                        "other_events": match[1].get("other_events", []),
                    },
                }
            ],
        }

    if final_ct_output:
        return ("simple_fully_matched", final_ct_output)
    else:
        return ("no_simple_match", None)


def main():
    """Main function to preprocess monopharmacy trials."""
    # Load file paths
    folder_path = Path(
        "./data/clinicaltrials_gov/completed_or_terminated_interventional_results_cts"
    )
    all_file_paths = list(folder_path.glob("*.json"))

    ##### Simple monopharmacy pre-processing #####

    # Initialize dictionary for storing statistics
    statistics = {
        "no_NCTID": 0,
        "no_EligibilityCriteria": 0,
        "no_ct_interventions": 0,
        "no_adverse_events": 0,
        "no_simple_match": 0,
    }

    # Initialize dictionary for storing parsed clinical trials data
    parsed_cts_simple = {
        "simple_fully_matched": {
            "trials": [],
            "serious_events": 0,
            "other_events": 0,
            "no_serious_events": 0,
            "no_other_events": 0,
        }
    }

    # Process each file using multiprocessing
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(simple_preprocessing, all_file_paths),
                total=len(all_file_paths),
            )
        )

    # Aggregate results from multiprocessing
    for result in results:
        result_type = result[0]
        if result_type in ["simple_fully_matched"]:
            # Append the clinical trial data
            parsed_cts_simple[result_type]["trials"].append(result[1])

            # Count the serious and other events in the current trial
            for group in result[1]["study_groups"]:
                parsed_cts_simple[result_type]["serious_events"] += (
                    1 if len(group["adverse_events"]["serious_events"]) > 0 else 0
                )
                parsed_cts_simple[result_type]["other_events"] += (
                    1 if len(group["adverse_events"]["other_events"]) > 0 else 0
                )
                parsed_cts_simple[result_type]["no_serious_events"] += (
                    1 if len(group["adverse_events"]["serious_events"]) == 0 else 0
                )
                parsed_cts_simple[result_type]["no_other_events"] += (
                    1 if len(group["adverse_events"]["other_events"]) == 0 else 0
                )
        else:
            statistics[result_type] += 1

    simple_preprocessing_monopharmacy = {
        trial["nctid"]: trial
        for trial in parsed_cts_simple["simple_fully_matched"]["trials"]
    }

    simple_preprocessing_monopharmacy_unique_group_code = []
    for nctid, trial in simple_preprocessing_monopharmacy.items():
        for study_group in trial["study_groups"]:
            simple_preprocessing_monopharmacy_unique_group_code.append(
                study_group["group_code"]
            )

    assert len(simple_preprocessing_monopharmacy_unique_group_code) == len(
        list(set(simple_preprocessing_monopharmacy_unique_group_code))
    )
    print("For simple monopharmacy preprocessing:")
    print("Unique nctids:", len(simple_preprocessing_monopharmacy))
    print(
        "Unique study groups:", len(simple_preprocessing_monopharmacy_unique_group_code)
    )

    ##### Group resolution monopharmacy pre-processing #####

    # Initialize dictionary for storing statistics
    statistics = {
        "no_NCTID": 0,
        "no_EligibilityCriteria": 0,
        "no_ct_interventions": 0,
        "no_ArmGroup_matched_to_interventions": 0,
        "no_adverse_events": 0,
        "no_matching_arm_group_at_all": 0,
    }

    # Initialize dictionary for storing parsed clinical trials data
    parsed_cts_group_resolution = {
        "fully_matched": {
            "trials": [],
            "serious_events": 0,
            "other_events": 0,
            "no_serious_events": 0,
            "no_other_events": 0,
        },
        "partially_matched": {
            "trials": [],
            "serious_events": 0,
            "other_events": 0,
            "no_serious_events": 0,
            "no_other_events": 0,
        },
    }

    # Process each file using multiprocessing
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(group_resolution_preprocessing, all_file_paths),
                total=len(all_file_paths),
            )
        )

    # Aggregate results from multiprocessing
    for result in results:
        result_type = result[0]
        if result_type in ["fully_matched", "partially_matched"]:
            # Append the clinical trial data
            parsed_cts_group_resolution[result_type]["trials"].append(result[1])

            # Count the serious and other events in the current trial
            for group in result[1]["study_groups"]:
                parsed_cts_group_resolution[result_type]["serious_events"] += (
                    1 if len(group["adverse_events"]["serious_events"]) > 0 else 0
                )
                parsed_cts_group_resolution[result_type]["other_events"] += (
                    1 if len(group["adverse_events"]["other_events"]) > 0 else 0
                )
                parsed_cts_group_resolution[result_type]["no_serious_events"] += (
                    1 if len(group["adverse_events"]["serious_events"]) == 0 else 0
                )
                parsed_cts_group_resolution[result_type]["no_other_events"] += (
                    1 if len(group["adverse_events"]["other_events"]) == 0 else 0
                )
        else:
            statistics[result_type] += 1

    fully_matched_group_resolution = parsed_cts_group_resolution["fully_matched"]
    partially_matched_group_resolution = parsed_cts_group_resolution[
        "partially_matched"
    ]

    group_resolution_preprocessing_monopharmacy = {}

    for trial in fully_matched_group_resolution["trials"]:
        study_groups = trial["study_groups"]
        for study_group in study_groups:
            if is_monopharmacy_group_resolution(
                study_group["intervention_details"]["name"]
            ):
                study_group["intervention_details"]["synonyms"] = (
                    list(study_group["intervention_details"]["synonyms"][0].values())[0]
                    if study_group["intervention_details"]["synonyms"]
                    else []
                )
                if trial["nctid"] in group_resolution_preprocessing_monopharmacy:
                    group_resolution_preprocessing_monopharmacy[trial["nctid"]][
                        "study_groups"
                    ].append(study_group)
                else:
                    group_resolution_preprocessing_monopharmacy[trial["nctid"]] = {
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
                        "study_groups": [],
                    }
                    group_resolution_preprocessing_monopharmacy[trial["nctid"]][
                        "study_groups"
                    ].append(study_group)

    for trial in partially_matched_group_resolution["trials"]:
        study_groups = trial["study_groups"]
        for study_group in study_groups:
            if is_monopharmacy_group_resolution(
                study_group["intervention_details"]["name"]
            ):
                study_group["intervention_details"]["synonyms"] = (
                    list(study_group["intervention_details"]["synonyms"][0].values())[0]
                    if study_group["intervention_details"]["synonyms"]
                    else []
                )
                if trial["nctid"] in group_resolution_preprocessing_monopharmacy:
                    group_resolution_preprocessing_monopharmacy[trial["nctid"]][
                        "study_groups"
                    ].append(study_group)
                else:
                    group_resolution_preprocessing_monopharmacy[trial["nctid"]] = {
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
                        "study_groups": [],
                    }
                    group_resolution_preprocessing_monopharmacy[trial["nctid"]][
                        "study_groups"
                    ].append(study_group)

    group_resolution_monopharmacy_unique_group_code = []
    for nctid, trial in group_resolution_preprocessing_monopharmacy.items():
        for study_group in trial["study_groups"]:
            group_resolution_monopharmacy_unique_group_code.append(
                study_group["group_code"]
            )

    assert len(group_resolution_monopharmacy_unique_group_code) == len(
        list(set(group_resolution_monopharmacy_unique_group_code))
    )
    print("For group resolution monopharmacy preprocessing:")
    print("Unique nctids:", len(group_resolution_preprocessing_monopharmacy))
    print("Unique study groups:", len(group_resolution_monopharmacy_unique_group_code))

    ##### Merging the two strategy outputs #####

    merged_preprocessing_monopharmacy = copy.deepcopy(simple_preprocessing_monopharmacy)
    annomalies = []
    for nctid, trial in group_resolution_preprocessing_monopharmacy.items():
        if nctid in merged_preprocessing_monopharmacy:
            study_groups = trial["study_groups"]
            for study_group in study_groups:
                if study_group["group_code"] not in [
                    i["group_code"]
                    for i in merged_preprocessing_monopharmacy[nctid]["study_groups"]
                ]:
                    # This should not happen. If it does, it's due to poorly annotated CTs. Currently, this is happening one time for a clinical trial
                    # where the "armGroupLabels" have been reversed (see "NCT00679055").
                    # This trial (NCTID) must be removed from the final result.
                    # print(f"Annomaly found for {nctid} ({nctid} will be deleted from the final result).")
                    annomalies.append(nctid)
        else:
            merged_preprocessing_monopharmacy[nctid] = trial

    for annomaly in annomalies:
        del merged_preprocessing_monopharmacy[annomaly]

    merged_preprocessing_monopharmacy_unique_group_code = []
    for nctid, trial in merged_preprocessing_monopharmacy.items():
        for study_group in trial["study_groups"]:
            merged_preprocessing_monopharmacy_unique_group_code.append(
                study_group["group_code"]
            )

    assert len(merged_preprocessing_monopharmacy_unique_group_code) == len(
        list(set(merged_preprocessing_monopharmacy_unique_group_code))
    )
    print("For simple + group resolution monopharmacy preprocessing:")
    print("Unique nctids:", len(merged_preprocessing_monopharmacy))
    print(
        "Unique study groups:", len(merged_preprocessing_monopharmacy_unique_group_code)
    )

    file_path = Path("data/clinicaltrials_gov/preprocessed_monopharmacy_cts.json")
    output_folder = file_path.parent
    output_folder.mkdir(parents=True, exist_ok=True)

    # Open the file in write mode
    with open(file_path, "w") as file:
        # Serialize dict to JSON formatted string and write to the file
        json.dump(merged_preprocessing_monopharmacy, file, indent=4)


if __name__ == "__main__":
    main()