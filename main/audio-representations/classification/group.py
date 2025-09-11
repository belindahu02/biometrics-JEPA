import os
import numpy as np
from collections import defaultdict
from typing import List, Optional, Union


def group_embeddings_by_frame(
        embeddings_root: str,
        output_root: str,
        include_prefixes: Optional[List[str]] = None,
        exclude_prefixes: Optional[List[str]] = None,
        verbose: bool = True
) -> None:
    """
    Group embedding files by frame ID and stack them together.

    Args:
        embeddings_root: Root directory containing embedding files
        output_root: Directory where grouped embeddings will be saved
        include_prefixes: List of filename prefixes to include (e.g., ['C2', 'Cpz'])
                         If None, all files are considered
        exclude_prefixes: List of filename prefixes to exclude (e.g., ['Fc1'])
                         If None, no files are excluded
        verbose: Whether to print progress information

    Example:
        # Group only files starting with 'C2' and 'Cpz'
        group_embeddings_by_frame(
            "/app/logs/eval_embeddings",
            "/app/logs/grouped_embeddings",
            include_prefixes=['C2', 'Cpz']
        )

        # Group all files except those starting with 'Fc1'
        group_embeddings_by_frame(
            "/app/logs/eval_embeddings",
            "/app/logs/grouped_embeddings",
            exclude_prefixes=['Fc1']
        )
    """

    def should_include_file(filename: str) -> bool:
        """Check if a file should be included based on prefix filters."""
        if not filename.endswith("_emb.npy"):
            return False

        # Check include_prefixes
        if include_prefixes is not None:
            if not any(filename.startswith(prefix) for prefix in include_prefixes):
                return False

        # Check exclude_prefixes
        if exclude_prefixes is not None:
            if any(filename.startswith(prefix) for prefix in exclude_prefixes):
                return False

        return True

    total_processed_dirs = 0
    total_stacked_frames = 0

    # Walk through all directories
    for root, dirs, files in os.walk(embeddings_root):
        # Filter files based on prefix criteria
        filtered_files = [f for f in files if should_include_file(f)]

        if not filtered_files:
            continue

        # Group files by frame ID
        frames_dict = defaultdict(list)

        for f in filtered_files:
            try:
                # Extract frame ID from filename (e.g., "C2_frame_000_emb.npy" -> "000")
                frame_id = f.split("_frame_")[1].split("_")[0]
                frames_dict[frame_id].append(f)
            except IndexError:
                if verbose:
                    print(f"⚠️  Warning: Could not parse frame ID from {f}")
                continue

        if not frames_dict:
            continue

        # Create output directory structure
        relative_path = os.path.relpath(root, embeddings_root)
        leaf_output_dir = os.path.join(output_root, relative_path)
        os.makedirs(leaf_output_dir, exist_ok=True)

        # Stack embeddings for each frame
        frames_processed = 0
        for frame_id, frame_files in frames_dict.items():
            try:
                frame_files.sort()  # Ensure consistent ordering
                arrays = [np.load(os.path.join(root, f)) for f in frame_files]
                stacked = np.vstack(arrays)

                save_path = os.path.join(leaf_output_dir, f"{frame_id}_stacked.npy")
                np.save(save_path, stacked)

                if verbose:
                    print(f"✅ Saved {save_path}: shape {stacked.shape} from {len(frame_files)} files")

                frames_processed += 1

            except Exception as e:
                if verbose:
                    print(f"❌ Error processing frame {frame_id} in {root}: {e}")

        if verbose:
            print(f"✅ Processed {root}: {frames_processed} frames stacked")

        total_processed_dirs += 1
        total_stacked_frames += frames_processed

    if verbose:
        print(f"\n🎉 Complete! Processed {total_processed_dirs} directories, "
              f"stacked {total_stacked_frames} total frames")


def group_embeddings_single_directory(
        input_dir: str,
        output_dir: str,
        include_prefixes: Optional[List[str]] = None,
        exclude_prefixes: Optional[List[str]] = None,
        verbose: bool = True
) -> None:
    """
    Group embeddings in a single directory (non-recursive version).

    Args:
        input_dir: Directory containing embedding files
        output_dir: Directory where grouped embeddings will be saved
        include_prefixes: List of filename prefixes to include
        exclude_prefixes: List of filename prefixes to exclude
        verbose: Whether to print progress information
    """
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Get all files in the directory
    files = os.listdir(input_dir)

    def should_include_file(filename: str) -> bool:
        """Check if a file should be included based on prefix filters."""
        if not filename.endswith("_emb.npy"):
            return False

        if include_prefixes is not None:
            if not any(filename.startswith(prefix) for prefix in include_prefixes):
                return False

        if exclude_prefixes is not None:
            if any(filename.startswith(prefix) for prefix in exclude_prefixes):
                return False

        return True

    # Filter files based on criteria
    filtered_files = [f for f in files if should_include_file(f)]

    if not filtered_files:
        if verbose:
            print("No files found matching the specified criteria")
        return

    # Group by frame ID
    frames_dict = defaultdict(list)

    for f in filtered_files:
        try:
            frame_id = f.split("_frame_")[1].split("_")[0]
            frames_dict[frame_id].append(f)
        except IndexError:
            if verbose:
                print(f"⚠️  Warning: Could not parse frame ID from {f}")
            continue

    # Stack embeddings for each frame
    for frame_id, frame_files in frames_dict.items():
        try:
            frame_files.sort()
            arrays = [np.load(os.path.join(input_dir, f)) for f in frame_files]
            stacked = np.vstack(arrays)

            save_path = os.path.join(output_dir, f"{frame_id}_stacked.npy")
            np.save(save_path, stacked)

            if verbose:
                print(f"✅ Saved {save_path}: shape {stacked.shape} from {len(frame_files)} files")

        except Exception as e:
            if verbose:
                print(f"❌ Error processing frame {frame_id}: {e}")

    if verbose:
        print(f"✅ Processed {len(frames_dict)} frames in {input_dir}")


# Example usage and configuration
if __name__ == "__main__":
    # Configuration
    # embeddings_root = "/app/logs/eval_embeddings"
    # output_root = "/app/logs/grouped_embeddings"

    embeddings_root = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/eval_embeddings"
    output_root = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/grouped_embeddings"


    # Example 1: Group only files with specific prefixes
    group_embeddings_by_frame(
        embeddings_root=embeddings_root,
        output_root=output_root,
        include_prefixes=['C2', 'Cpz'],  # Only process C2 and Cpz files
        verbose=True
    )

    # Example 2: Group all files except specific prefixes
    # group_embeddings_by_frame(
    #     embeddings_root=embeddings_root,
    #     output_root=output_root,
    #     exclude_prefixes=['Fc1'],  # Process all except Fc1 files
    #     verbose=True
    # )

    # Example 3: Group all files (original behavior)
    # group_embeddings_by_frame(
    #     embeddings_root=embeddings_root,
    #     output_root=output_root,
    #     verbose=True
    # )