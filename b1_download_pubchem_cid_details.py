import json
from pathlib import Path
import logging
import time
from typing import Optional, List, Dict
from tqdm import tqdm
import requests


def setup_logging() -> None:
    """
    Set up basic logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def load_cids_from_json(filepath: Path) -> set:
    """
    Load Compound IDs (CIDs) from a JSON file.

    Args:
        filepath (Path): Path to the JSON file.

    Returns:
        set: Set of CIDs loaded from the file.
    """
    with open(filepath, "r") as file:
        data = json.load(file)
    return [str(i) for i in data]


def fetch_synonyms(cid: str, retries: int = 5, delay: int = 2) -> Optional[List[str]]:
    """
    Fetch synonyms for a given Compound ID (CID) from PubChem.

    Args:
        cid (str): Compound ID.
        retries (int): Number of retries in case of failure. Default is 5.
        delay (int): Delay between retries in seconds. Default is 2.

    Returns:
        list: List of synonyms if successful, None otherwise.
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()["InformationList"]["Information"][0].get(
                "Synonym", []
            )
        except (requests.exceptions.RequestException, KeyError) as e:
            logging.error(f"Error on attempt {attempt + 1} for CID {cid}: {str(e)}")
            time.sleep(delay)
    logging.error(f"Failed to fetch synonyms for CID {cid} after {retries} attempts.")
    return None


def fetch_title(cid: str, retries: int = 5, delay: int = 2) -> Optional[str]:
    """
    Fetch title for a given Compound ID (CID) from PubChem.

    Args:
        cid (str): Compound ID.
        retries (int): Number of retries in case of failure. Default is 5.
        delay (int): Delay between retries in seconds. Default is 2.

    Returns:
        str: Title if successful, None otherwise.
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Title/JSON"
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()["PropertyTable"]["Properties"][0].get("Title", "")
        except (requests.exceptions.RequestException, KeyError) as e:
            logging.error(f"Error on attempt {attempt + 1} for CID {cid}: {str(e)}")
            time.sleep(delay)
    logging.error(f"Failed to fetch title for CID {cid} after {retries} attempts.")
    return None


def fetch_canonical_smiles(cid: str, retries: int = 5, delay: int = 2) -> Optional[str]:
    """
    Fetch Canonical SMILES for a given Compound ID (CID) from PubChem.

    Args:
        cid (str): Compound ID.
        retries (int): Number of retries in case of failure. Default is 5.
        delay (int): Delay between retries in seconds. Default is 2.

    Returns:
        str: Canonical SMILES if successful, None otherwise.
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()["PropertyTable"]["Properties"][0].get(
                "CanonicalSMILES", ""
            )
        except (requests.exceptions.RequestException, KeyError) as e:
            logging.error(f"Error on attempt {attempt + 1} for CID {cid}: {str(e)}")
            time.sleep(delay)
    logging.error(
        f"Failed to fetch Canonical SMILES for CID {cid} after {retries} attempts."
    )
    return None


def fetch_atc(cid: str, retries: int = 5, delay: int = 2) -> Optional[str]:
    """
    Fetch Anatomical Therapeutic Chemical (ATC) code for a given Compound ID (CID) from PubChem.

    Args:
        cid (str): Compound ID.
        retries (int): Number of retries in case of failure. Default is 5.
        delay (int): Delay between retries in seconds. Default is 2.

    Returns:
        str: ATC code if successful, None otherwise.
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    for attempt in range(retries):
        try:
            if attempt > 0:
                logging.info(f"Proceeding with attempt {attempt + 1} for CID {cid}")
            response = requests.get(url)
            response.raise_for_status()

            data = response.json()

            def json_extract(obj, key):
                arr = []

                def extract(obj, arr, key):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if isinstance(v, (dict, list)):
                                extract(v, arr, key)
                            elif k == key:
                                arr.append(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            extract(item, arr, key)
                    return arr

                return extract(obj, arr, key)

            urls = json_extract(data, "URL")
            atc_codes = [
                url.split("=")[1].split("&")[0]
                for url in urls
                if "www.whocc.no" in url and "code=" in url
            ]

            if atc_codes:
                longest_atc_code = max(atc_codes, key=len)
                return longest_atc_code
            else:
                return None

        except (requests.exceptions.RequestException, KeyError) as e:
            logging.error(f"Error on attempt {attempt + 1} for CID {cid}: {str(e)}")
            time.sleep(delay)
    logging.error(f"Failed to fetch ATC code for CID {cid} after {retries} attempts.")
    return None


def main() -> None:
    """
    Main function to fetch information for each Compound ID (CID).
    """
    setup_logging()
    cids_file_path = Path("data/pubchem/unique_cids.json")
    cids = load_cids_from_json(cids_file_path)

    cid_info_dict: Dict[str, Dict[str, Optional[str]]] = {}
    progress = tqdm(total=len(cids), desc="Fetching Information")

    for cid in cids:
        cid_info: Dict[str, Optional[str]] = {}
        smiles = fetch_canonical_smiles(cid)
        if not smiles:
            continue
        cid_info["title"] = fetch_title(cid)
        cid_info["synonyms"] = fetch_synonyms(cid)
        cid_info["smiles"] = smiles
        cid_info["atc_code"] = fetch_atc(cid)
        cid_info_dict[cid] = cid_info

        progress.update(1)

    output_path = Path("data/pubchem/cid_details.json")
    output_folder = output_path.parent
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cid_info_dict, f, indent=4)


if __name__ == "__main__":
    main()