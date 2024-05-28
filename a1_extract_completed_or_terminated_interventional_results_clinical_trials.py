import json
from pathlib import Path
from shutil import copy2
from multiprocessing import Pool
from tqdm.auto import tqdm
import logging


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def load_json_data(file_path: Path) -> dict:
    """Load JSON data from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {file_path}: {str(e)}")
        return {}
    except Exception as e:
        logging.error(f"Error reading {file_path}: {str(e)}")
        return {}


def criteria_check(data: dict) -> bool:
    """Check if the JSON data meets specific criteria: Completed, Interventional with ResultsSection."""
    try:
        if (
            ("Study" in data["FullStudy"])
            and (
                data["FullStudy"]["Study"]["ProtocolSection"]["StatusModule"][
                    "OverallStatus"
                ]
                in ["Completed", "Terminated"]
            )
            and (
                data["FullStudy"]["Study"]["ProtocolSection"]["DesignModule"][
                    "StudyType"
                ]
                == "Interventional"
            )
            and ("ResultsSection" in data["FullStudy"]["Study"])
        ):
            return True
        else:
            return False
    except KeyError as e:
        logging.debug(f"Key error: {str(e)} in data")
        return False


def process_file(file_path: Path) -> Path:
    """Process a single file to check criteria and return path if criteria met."""
    data = load_json_data(file_path)
    if data and criteria_check(data):
        return file_path
    return None


def copy_file(source: Path, target_dir: Path):
    """Copy a file to the target directory."""
    try:
        copy2(source, target_dir / source.name)
    except Exception as e:
        logging.error(f"Failed to copy {source} to {target_dir}: {str(e)}")


def copy_file_to_target(source_target_tuple):
    """Wrapper function for copying file, accepts a tuple (source, target_dir)."""
    source, target_dir = source_target_tuple
    copy_file(source, target_dir)


def main():
    setup_logging()
    dir_path = Path("data/clinicaltrials_gov/all_cts")
    target_dir_path = Path("data/clinicaltrials_gov/completed_or_terminated_interventional_results_cts")
    target_dir_path.mkdir(parents=True, exist_ok=True)

    json_paths = list(dir_path.rglob("*.json"))

    with Pool() as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_file, json_paths),
                total=len(json_paths),
                desc="Filtering",
            )
        )

    completed_files = [path for path in results if path]

    if not completed_files:
        logging.info("No files met the criteria.")

    with Pool() as pool:
        list(
            tqdm(
                pool.imap_unordered(
                    copy_file_to_target, ((p, target_dir_path) for p in completed_files)
                ),
                total=len(completed_files),
                desc="Copying",
            )
        )


if __name__ == "__main__":
    main()