# src/utils/path_discovery.py
# Helper to discover and handle paths in the classification scripts

import os
import re
from pathlib import Path
from typing import Tuple, Optional, Dict


class ClassificationPathDiscovery:
    """
    Utility class to discover and manage paths in classification scripts
    """

    def __init__(self, data_dir: str, classification_scripts_dir: str):
        self.data_dir = Path(data_dir)
        self.classification_scripts_dir = Path(classification_scripts_dir)

    def find_csv_file(self) -> Optional[Path]:
        """Find the audioset CSV file in the data directory"""
        possible_names = [
            "files_audioset.csv",
            "audioset_files.csv",
            "files.csv",
            "audioset.csv"
        ]

        for name in possible_names:
            csv_path = self.data_dir / name
            if csv_path.exists():
                return csv_path

        # Search recursively
        for csv_file in self.data_dir.rglob("*audioset*.csv"):
            return csv_file

        return None

    def discover_precompute_paths(self) -> Dict[str, str]:
        """Discover paths used in precompute.py by reading the script"""
        precompute_script = self.classification_scripts_dir / "precompute.py"

        if not precompute_script.exists():
            raise FileNotFoundError(f"precompute.py not found at {precompute_script}")

        paths = {}

        with open(precompute_script, 'r') as f:
            content = f.read()

        # Extract checkpoint path pattern
        checkpoint_match = re.search(r'checkpoint_path\s*=\s*["\']([^"\']+)["\']', content)
        if checkpoint_match:
            paths['original_checkpoint_path'] = checkpoint_match.group(1)

        # Extract CSV file path pattern
        csv_match = re.search(r'csv_file\s*=\s*["\']([^"\']+)["\']', content)
        if csv_match:
            paths['original_csv_path'] = csv_match.group(1)

        # Extract data directory pattern
        data_dir_match = re.search(r'data_dir\s*=\s*["\']([^"\']+)["\']', content)
        if data_dir_match:
            paths['original_data_dir'] = data_dir_match.group(1)

        # Extract embeddings directory pattern
        emb_dir_match = re.search(r'embeddings_dir\s*=\s*["\']([^"\']+)["\']', content)
        if emb_dir_match:
            paths['original_embeddings_dir'] = emb_dir_match.group(1)

        return paths

    def discover_group_paths(self) -> Dict[str, str]:
        """Discover paths used in group.py by reading the script"""
        group_script = self.classification_scripts_dir / "group.py"

        if not group_script.exists():
            raise FileNotFoundError(f"group.py not found at {group_script}")

        paths = {}

        with open(group_script, 'r') as f:
            content = f.read()

        # Extract embeddings root pattern
        emb_root_match = re.search(r'embeddings_root\s*=\s*["\']([^"\']+)["\']', content)
        if emb_root_match:
            paths['original_embeddings_root'] = emb_root_match.group(1)

        # Extract output root pattern
        output_root_match = re.search(r'output_root\s*=\s*["\']([^"\']+)["\']', content)
        if output_root_match:
            paths['original_output_root'] = output_root_match.group(1)

        return paths

    def create_path_mapping(self, eval_results_dir: Path) -> Dict[str, Path]:
        """Create a mapping from original paths to evaluation-specific paths"""
        embeddings_dir = eval_results_dir / "eval_embeddings"
        grouped_dir = eval_results_dir / "grouped_embeddings"

        # Find CSV file
        csv_file = self.find_csv_file()
        if not csv_file:
            raise FileNotFoundError("Could not find audioset CSV file")

        return {
            'csv_file': csv_file,
            'data_dir': self.data_dir,
            'embeddings_dir': embeddings_dir,
            'grouped_dir': grouped_dir,
            'eval_results_dir': eval_results_dir
        }

    def patch_precompute_script(self, checkpoint_path: str, path_mapping: Dict[str, Path]) -> str:
        """Create a patched version of precompute.py with correct paths"""
        precompute_script = self.classification_scripts_dir / "precompute.py"

        with open(precompute_script, 'r') as f:
            content = f.read()

        # Replace checkpoint path
        content = re.sub(
            r'checkpoint_path\s*=\s*["\'][^"\']+["\']',
            f'checkpoint_path = "{checkpoint_path}"',
            content
        )

        # Replace CSV file path
        content = re.sub(
            r'csv_file\s*=\s*["\'][^"\']+["\']',
            f'csv_file = "{path_mapping["csv_file"]}"',
            content
        )

        # Replace data directory
        content = re.sub(
            r'data_dir\s*=\s*["\'][^"\']+["\']',
            f'data_dir = "{path_mapping["data_dir"]}"',
            content
        )

        # Replace or add embeddings directory
        if 'embeddings_dir =' in content:
            content = re.sub(
                r'embeddings_dir\s*=\s*["\'][^"\']+["\']',
                f'embeddings_dir = "{path_mapping["embeddings_dir"]}"',
                content
            )
        else:
            # Add embeddings_dir if it doesn't exist
            # Find the line with os.makedirs and add before it
            content = re.sub(
                r'(os\.makedirs\(embeddings_dir, exist_ok=True\))',
                f'embeddings_dir = "{path_mapping["embeddings_dir"]}"\n    \\1',
                content
            )

        return content

    def patch_group_script(self, path_mapping: Dict[str, Path]) -> str:
        """Create a patched version of group.py with correct paths"""
        group_script = self.classification_scripts_dir / "group.py"

        with open(group_script, 'r') as f:
            content = f.read()

        # Replace embeddings_root
        content = re.sub(
            r'embeddings_root\s*=\s*["\'][^"\']+["\']',
            f'embeddings_root = "{path_mapping["embeddings_dir"]}"',
            content
        )

        # Replace output_root
        content = re.sub(
            r'output_root\s*=\s*["\'][^"\']+["\']',
            f'output_root = "{path_mapping["grouped_dir"]}"',
            content
        )

        return content

    def validate_data_structure(self) -> bool:
        """Validate that the data directory has the expected structure"""
        csv_file = self.find_csv_file()
        if not csv_file:
            print("❌ CSV file not found")
            return False

        if not self.data_dir.exists():
            print(f"❌ Data directory not found: {self.data_dir}")
            return False

        print(f"✅ Found CSV file: {csv_file}")
        print(f"✅ Found data directory: {self.data_dir}")

        return True

    def get_classification_data_path(self) -> Optional[Path]:
        """Find the classification input data path (grouped embeddings from previous runs)"""
        possible_paths = [
            self.data_dir / "classification_input",
            self.data_dir / "grouped_embeddings",
            self.data_dir / "graph_data_2d",
            self.data_dir.parent / "classification_input",
            self.data_dir.parent / "grouped_embeddings"
        ]

        for path in possible_paths:
            if path.exists() and any(path.iterdir()):  # Directory exists and is not empty
                print(f"✅ Found classification data at: {path}")
                return path

        print("⚠️ No existing classification data found - will be generated during evaluation")
        return None