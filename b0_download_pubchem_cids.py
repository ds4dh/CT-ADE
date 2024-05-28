import json
import logging
import requests
from pathlib import Path
from typing import List
from time import sleep


def setup_logging() -> None:
    """Sets up the logging configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def fetch_and_collect_cids(
    base_url: str, total_pages: int, max_retries: int = 3, delay: float = 2.0
) -> List[int]:
    """
    Fetches all pages from the base URL, retrying if necessary, and collects CIDs from each page.

    Args:
        base_url: The base URL for fetching the data, without the page parameter.
        total_pages: Total number of pages to fetch.
        max_retries: Maximum number of retries for a failed request.
        delay: Delay between retries in seconds.

    Returns:
        List of CIDs collected from all pages.
    """
    cids = []
    for page in range(1, total_pages + 1):
        retries = 0
        while retries <= max_retries:
            url = f"{base_url}&page={page}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                page_cids = parse_cids(data)
                cids.extend(page_cids)
                logging.info(
                    f"Page {page} of {total_pages} fetched and CIDs collected."
                )
                break
            else:
                retries += 1
                logging.error(
                    f"Failed to fetch page {page}: HTTP {response.status_code}. Retrying... Attempt {retries}"
                )
                sleep(delay)
        if retries > max_retries:
            logging.error(f"Failed to fetch page {page} after {max_retries} attempts.")
    return cids


def parse_cids(data: dict) -> List[int]:
    """
    Parses the JSON data to extract CIDs.

    Args:
        data: JSON data from which to extract CIDs.

    Returns:
        List of CIDs extracted from the data.
    """
    cids = []
    annotations = data.get("Annotations", {}).get("Annotation", [])
    for annotation in annotations:
        linked_cids = annotation.get("LinkedRecords", {}).get("CID", [])
        cids.extend(linked_cids)
    return cids


def save_cids(cids: List[int], output_path: Path) -> None:
    """
    Saves the list of CIDs to a JSON file.

    Args:
        cids: List of CIDs to save.
        output_path: Path where the CIDs should be saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(cids, file, indent=2)
    logging.info(f"CIDs have been saved to {output_path}")


def main() -> None:
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/annotations/heading/JSON/?source=ClinicalTrials.gov&heading_type=Compound&heading=ClinicalTrials.gov&response_type=save&response_basename=PubChemAnnotations_ClinicalTrials.gov_heading%3DClinicalTrials.gov"
    total_pages = 12
    cids = fetch_and_collect_cids(base_url, total_pages)
    unique_cids = [str(cid) for cid in list(set(cids))]
    save_directory = Path("data/pubchem")
    output_file = "unique_cids.json"
    save_cids(unique_cids, save_directory / output_file)


if __name__ == "__main__":
    setup_logging()
    main()