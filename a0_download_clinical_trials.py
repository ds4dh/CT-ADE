import wget
import zipfile
from pathlib import Path
import logging

def setup_logging() -> None:
    """Set up basic logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_and_extract_zip(url: str, directory: Path, filename: str) -> None:
    """
    Downloads a ZIP file from the specified URL and extracts it to the given directory.

    Args:
        url (str): The URL from where to download the ZIP file.
        directory (Path): The directory to store and extract the ZIP file.
        filename (str): The filename for the downloaded ZIP file.

    Raises:
        Exception: If downloading or extracting the ZIP file fails.
    """
    # Ensure the directory exists
    directory.mkdir(parents=True, exist_ok=True)
    
    # Define the path for the ZIP file
    zip_file_path = directory / filename

    # Download the ZIP file
    try:
        wget.download(url, out=str(zip_file_path))
        logging.info(f"ZIP file downloaded successfully: {zip_file_path}")
    except Exception as e:
        logging.error(f"Failed to download the ZIP file from {url}: {str(e)}")
        raise
    
    # Extract the ZIP file
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(directory)
        logging.info(f"ZIP file extracted successfully in {directory}")
    except zipfile.BadZipFile as e:
        logging.error(f"The file is not a ZIP file or it is corrupted: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Failed to extract the ZIP file: {str(e)}")
        raise

def main() -> None:
    """
    Main function to handle the downloading and extracting of a ZIP file.
    """
    setup_logging()
    url = "https://classic.clinicaltrials.gov/AllAPIJSON.zip"
    all_cts_dir = Path('data/clinicaltrials_gov/all_cts')
    filename = 'AllAPIJSON.zip'

    try:
        download_and_extract_zip(url, all_cts_dir, filename)
    except Exception as e:
        logging.error(f"An error occurred during the process: {str(e)}")

if __name__ == "__main__":
    main()