import os
from typing import Dict, Tuple, List, Set


class Node:
    def __init__(self, code: str, term: str, level: str) -> None:
        """Initialize a Node with a code, term, and level.

        Args:
            code (str): The unique identifier for the medical term.
            term (str): The descriptive name of the medical term.
            level (str): The hierarchical level of the term in the MedDRA system.
        """
        self.code: str = code
        self.term: str = term
        self.level: str = level
        self.parents: Set["Node"] = set()


class MedDRA:
    def __init__(self) -> None:
        """Initialize an empty MedDra dictionary to hold Node objects."""
        self.nodes: Dict[Tuple[str, str], Node] = {}

    def add_node(self, code: str, term: str, level: str) -> None:
        """Add a node to the dictionary if it does not already exist.

        Args:
            code (str): The unique identifier for the medical term.
            term (str): The descriptive name of the medical term.
            level (str): The hierarchical level of the term in the MedDRA system.
        """
        node_key = (level, code)
        if node_key not in self.nodes:
            self.nodes[node_key] = Node(code, term, level)

    def add_edge(self, code1: str, level1: str, code2: str, level2: str) -> None:
        """Create a directed edge from node1 to node2.

        Args:
            code1 (str): The code of the parent node.
            level1 (str): The level of the parent node.
            code2 (str): The code of the child node.
            level2 (str): The level of the child node.
        """
        node1_key, node2_key = (level1, code1), (level2, code2)
        if node1_key in self.nodes and node2_key in self.nodes:
            self.nodes[node2_key].parents.add(self.nodes[node1_key])

    def find_paths(
        self, code: str, level: str, pad_levels: bool = True
    ) -> List[List[str]]:
        """Recursively find all paths from the given node to its ancestors.

        Args:
            code (str): The code of the node from which to start the path search.
            level (str): The level of the starting node.
            pad_levels (bool): Whether to include padding for bypassed levels in the path.

        Returns:
            List[List[str]]: A list of paths, each path is a list of node identifiers with levels.
        """
        node_key = (level, code)
        if node_key not in self.nodes:
            return []

        node = self.nodes[node_key]
        if not node.parents:
            return [[f"{node.code}@{node.level}"]]

        paths = []
        for parent in node.parents:
            parent_paths = self.find_paths(parent.code, parent.level, pad_levels)
            for path in parent_paths:
                path_with_padding = []
                if pad_levels:
                    parent_index = self.get_level_index(parent.level)
                    current_index = self.get_level_index(node.level)
                    padding_levels = current_index - parent_index - 1

                    for i in range(padding_levels):
                        padding_level = self.get_level_by_index(parent_index + i + 1)
                        path_with_padding.append(f"BYPASS@{padding_level}")

                paths.append(path + path_with_padding + [f"{node.code}@{node.level}"])

        return paths

    def find_node_by_term(
        self, term: str, levels: Set[str] = None, preprocess=None
    ) -> List[Node]:
        """Find all nodes that match the given term within specific levels with optional preprocessing.
    
        Args:
            term (str): The term to search for.
            levels (Set[str]): Optional set of levels to restrict the search.
            preprocess (Callable[[str], str]): Optional function to preprocess terms.
    
        Returns:
            List[Node]: A list of nodes with the given term after preprocessing.
        """
        if preprocess:
            term = preprocess(term)
        filtered_nodes = []
        for node in self.nodes.values():
            node_term = preprocess(node.term) if preprocess else node.term
            if node_term == term:
                if levels is None or node.level in levels:
                    filtered_nodes.append(node)
        return filtered_nodes

    @staticmethod
    def get_level_index(level: str) -> int:
        """Get the index of a level in the predefined order.

        Args:
            level (str): The level name.

        Returns:
            int: The index of the level in the MedDRA hierarchy.
        """
        level_order = ["SOC", "HLGT", "HLT", "PT", "LLT"]
        return level_order.index(level)

    @staticmethod
    def get_level_by_index(index: int) -> str:
        """Get the level name by index from the predefined order.

        Args:
            index (int): The index in the hierarchy levels.

        Returns:
            str: The level name corresponding to the index.
        """
        level_order = ["SOC", "HLGT", "HLT", "PT", "LLT"]
        return level_order[index]

    def load_data(self, path: str) -> None:
        """Load data from files to create nodes and their relationships.

        Args:
            path (str): The directory path where the MedDRA ASCII files are stored.
        """
        # Create nodes from .asc files
        for filename, level in [
            ("soc.asc", "SOC"),
            ("hlgt.asc", "HLGT"),
            ("hlt.asc", "HLT"),
            ("pt.asc", "PT"),
            ("llt.asc", "LLT"),
        ]:
            with open(os.path.join(path, filename), "r") as file:
                for line in file:
                    parts = line.strip().split("$")
                    code, term = parts[0], parts[1]
                    self.add_node(code, term, level)

        # Create relationships from .asc files
        for filename, level1, level2 in [
            ("soc_hlgt.asc", "SOC", "HLGT"),
            ("hlgt_hlt.asc", "HLGT", "HLT"),
            ("hlt_pt.asc", "HLT", "PT"),
        ]:
            with open(os.path.join(path, filename), "r") as file:
                for line in file:
                    parts = line.strip().split("$")
                    code1, code2 = parts[0], parts[1]
                    self.add_edge(code1, level1, code2, level2)

        # For PT and LLT, the parent level is unknown.
        # We need to check if a node with the parent code exists at each level above PT/LLT.
        for filename, level in [("pt.asc", "PT"), ("llt.asc", "LLT")]:
            with open(os.path.join(path, filename), "r") as file:
                for line in file:
                    parts = line.strip().split("$")
                    code, parent_code = parts[0], (
                        parts[2] if level == "LLT" else parts[3]
                    )
                    above_levels = (
                        ["PT", "HLT", "HLGT", "SOC"]
                        if level == "LLT"
                        else ["HLT", "HLGT", "SOC"]
                    )
                    for parent_level in above_levels:
                        parent_node_key = (parent_level, parent_code)
                        if parent_node_key in self.nodes:
                            self.add_edge(parent_code, parent_level, code, level)