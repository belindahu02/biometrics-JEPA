import os
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------- #
# Config
# ---------------------------------------------------------------------------- #
# embeddings_root = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/eval_embeddings"
# output_root = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/classification_input"
embeddings_root = "/app/logs/eval_embeddings"   # root of per-frame embeddings
output_root = "/app/logs/grouped_embeddings"              # where to save stacked embeddings
# os.makedirs(output_root, exist_ok=True)
#
# ---------------------------------------------------------------------------- #
# Walk through all leaf folders
# ---------------------------------------------------------------------------- #
for root, dirs, files in os.walk(embeddings_root):
    # only process leaf folders with .npy files
    npy_files = [f for f in files if f.endswith("_emb.npy")]
    if not npy_files:
        continue

    # prepare frame-wise grouping
    frames_dict = defaultdict(list)

    for f in npy_files:
        # Example filename: Af3_frame_000_emb.npy
        frame_id = f.split("_frame_")[1].split("_")[0]  # note the underscore after 'frame'
        frames_dict[frame_id].append(f)

    # stack embeddings for each frame
    leaf_output_dir = os.path.join(output_root, os.path.relpath(root, embeddings_root))
    os.makedirs(leaf_output_dir, exist_ok=True)

    for frame_id, frame_files in frames_dict.items():
        frame_files.sort()
        arrays = [np.load(os.path.join(root, f)) for f in frame_files]
        stacked = np.vstack(arrays)
        save_path = os.path.join(leaf_output_dir, f"{frame_id}_stacked.npy")
        np.save(save_path, stacked)
        print(f"✅ Saved {save_path}: stacked shape {stacked.shape} from {len(frame_files)} files")
        print(f"✅ Saved {save_path}: stacked shape {stacked.shape} from {len(frame_files)} files")

    print(f"✅ Processed {root}: {len(frames_dict)} frames stacked")

