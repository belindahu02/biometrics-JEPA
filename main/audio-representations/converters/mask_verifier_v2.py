import os
import numpy as np
import matplotlib.pyplot as plt

def plot_mask_verification(original_path, masked_path, out_dir="mask_debug"):
    # Load arrays
    orig = np.load(original_path)
    masked = np.load(masked_path)

    # Remove leading dimension if present
    if orig.ndim == 3 and orig.shape[0] == 1:
        orig = orig.squeeze(0)
    if masked.ndim == 3 and masked.shape[0] == 1:
        masked = masked.squeeze(0)

    # Sanity check
    if orig.shape != masked.shape:
        print(f"❌ Shape mismatch: {original_path} {orig.shape} vs {masked.shape}")
        return

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    vmin, vmax = orig.min(), orig.max()  # consistent color scale

    axes[0].imshow(orig, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    axes[0].set_title("Original")
    axes[0].set_xlabel("Time frames")
    axes[0].set_ylabel("Frequency bins")

    axes[1].imshow(masked, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    axes[1].set_title("Masked")
    axes[1].set_xlabel("Time frames")

    plt.suptitle(os.path.basename(original_path), fontsize=14)
    out_file = os.path.join(out_dir, f"verify_{os.path.basename(original_path)}.png")
    plt.savefig(out_file, dpi=150)
    plt.close()

    print(f"✅ Saved: {out_file}")


def batch_verify(original_dir, masked_dir, out_dir="frame_mask_debug"):
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(original_dir):
        if not fname.endswith(".npy"):
            continue

        orig_path = os.path.join(original_dir, fname)
        masked_path = os.path.join(masked_dir, fname)

        if not os.path.exists(masked_path):
            print(f"⚠️ No masked version found for {fname}")
            continue

        plot_mask_verification(orig_path, masked_path, out_dir)


if __name__ == "__main__":
    # Example usage (update these paths):
    original_dir = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/S001/S001R01"
    masked_dir   = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/masked_frames/S001/S001R01"

    batch_verify(original_dir, masked_dir)
