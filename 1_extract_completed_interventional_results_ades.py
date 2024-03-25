import glob
import os
import json
import shutil
from tqdm.auto import tqdm
from multiprocessing import Pool


def is_Study(data: dict) -> bool:
    """
    Check if the "Study" key exists in the data.

    Args:
        data: A dictionary containing the data to be checked.

    Returns:
        True if the "Study" key exists in the data, False otherwise.
    """
    return "Study" in data["FullStudy"]


def is_ResultsSection(data: dict) -> bool:
    """
    Check if the "ResultsSection" key exists in the data.

    Args:
        data: A dictionary containing the data to be checked.

    Returns:
        True if the "ResultsSection" key exists in the data, False otherwise.
    """
    return "ResultsSection" in data["FullStudy"]["Study"]


def is_AdverseEventsModule(data: dict) -> bool:
    """
    Check if the "AdverseEventsModule" key exists in the data.

    Args:
        data: A dictionary containing the data to be checked.

    Returns:
        True if the "AdverseEventsModule" key exists in the data, False otherwise.
    """
    return "AdverseEventsModule" in data["FullStudy"]["Study"]["ResultsSection"]


def is_Completed(data: dict) -> bool:
    """
    Check if the study is completed.

    Args:
        data: A dictionary containing the data to be checked.

    Returns:
        True if the "OverallStatus" field in the "StatusModule" section is "Completed", False otherwise.
    """
    return data["FullStudy"]["Study"]["ProtocolSection"]["StatusModule"]["OverallStatus"] == "Completed"


def is_Interventional(data: dict) -> bool:
    """
    Check if the study is interventional.

    Args:
        data: A dictionary containing the data to be checked.

    Returns:
        True if the "StudyType" field in the "DesignModule" section is "Interventional", False otherwise.
    """
    return data["FullStudy"]["Study"]["ProtocolSection"]["DesignModule"]["StudyType"] == "Interventional"

def process_file(file_path):
    """
    Process a single file: check criteria and copy file if it meets criteria.

    Args:
        file_path: Path to the file to be processed.

    Returns:
        The file path if criteria are met, None otherwise.
    """
    with open(file_path) as f:
        data = json.load(f) 

    if is_Study(data) and is_ResultsSection(data) and is_Completed(data) and is_Interventional(data) and is_AdverseEventsModule(data):
        return file_path
    else:
        return None

def copy_file(file_path):
    """
    Copy a file to the target directory.

    Args:
        file_path: Path to the file to be copied.
    """
    shutil.copyfile(file_path, target_dir_path + "/" + os.path.basename(file_path))

# Load path to all studies.
dir_path = "./data/all_cts"
json_paths = list(glob.iglob(os.path.join(dir_path, '*/*.json')))

# Define target directory.
target_dir_path = "./data/completed_interventional_results_ades"
if not os.path.exists(target_dir_path):
    os.makedirs(target_dir_path)

# Process files in parallel using multiprocessing.
with Pool() as pool:
    filtered_paths = list(tqdm(pool.imap_unordered(process_file, json_paths), total=len(json_paths), desc="Filtering"))

# Filter out None values.
completed_interventional_results_ades = [path for path in filtered_paths if path]

# Copy files in parallel.
with Pool() as pool:
    list(tqdm(pool.imap_unordered(copy_file, completed_interventional_results_ades), total=len(completed_interventional_results_ades), desc="Copying"))