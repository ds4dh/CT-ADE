from typing import Any, Dict
from fuzzywuzzy import fuzz, utils


class FuzzyDict(Dict[str, Any]):
    """Dictionary that uses fuzzy matching to return values for keys that are not
    present in the dictionary.

    Parameters
    ----------
    fuzzy_threshold : int, optional
        Minimum ratio required for a fuzzy match. Defaults to 100.
    """

    def __init__(self, *args, fuzzy_threshold: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.fuzzy_threshold = fuzzy_threshold
        # Precompute the processed keys
        self.processed_keys = {key: utils.full_process(key) for key in self}
        # Create a reverse lookup for processed keys
        self.reverse_processed_keys = {
            v: k for k, v in self.processed_keys.items()}

    def __getitem__(self, key: str) -> Any:
        # If the key is in the dictionary, return the value as usual
        if key in self:
            return super().__getitem__(key)

        # Process the input key
        key_processed = utils.full_process(key)

        # Check if the processed input key matches any of the processed keys
        if key_processed in self.reverse_processed_keys:
            return super().__getitem__(self.reverse_processed_keys[key_processed])

        # Perform a fuzzy search to find the closest matching key
        closest_key = None
        closest_ratio = 0
        for k, k_processed in self.processed_keys.items():
            ratio = fuzz.token_sort_ratio(key_processed, k_processed)

            # If perfect match is found, return the value immediately
            if ratio == 100:
                return super().__getitem__(k)

            # Otherwise, continue searching for the closest match
            if ratio > closest_ratio:
                closest_key = k
                closest_ratio = ratio

        # If a sufficiently close match was found, return the value for that key
        if closest_ratio >= self.fuzzy_threshold:
            return super().__getitem__(closest_key)

        # Otherwise, raise a KeyError
        raise KeyError(key)

    # Override the __setitem__ method to update processed_keys when a new item is added
    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        processed_key = utils.full_process(key)
        self.processed_keys[key] = processed_key
        self.reverse_processed_keys[processed_key] = key

    # Override the __delitem__ method to keep processed_keys in sync when an item is deleted
    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        processed_key = self.processed_keys.pop(key)
        self.reverse_processed_keys.pop(processed_key)