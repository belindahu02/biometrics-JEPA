"""
Script to verify that EEG masking worked correctly by comparing original and masked spectrograms.

This script helps you:
1. Load and compare original vs masked .npy files
2. Calculate masking statistics
3. Visualize differences between original and masked spectrograms
4. Verify masking patterns and coverage

Usage:
    python verify_masking.py /path/to/original/data /path/to/masked/data

    # Or for specific files:
    python verify_masking.py --original_file path/to/original.npy --masked_file path/to/masked.npy
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy import stats
import pandas as pd


class MaskingVerifier:
    def __init__(self, original_path, masked_path):
        self.original_path = Path(original_path)
        self.masked_path = Path(masked_path)

    def load_spectrogram(self, filepath):
        """Load a spectrogram .npy file and handle different shapes."""
        data = np.load(filepath)

        # Handle different tensor shapes
        if data.ndim == 3:
            # Shape: (1, n_mels, time_frames) -> squeeze to (n_mels, time_frames)
            data = data.squeeze(0)
        elif data.ndim == 4:
            # Shape: (1, 1, n_mels, time_frames) -> squeeze to (n_mels, time_frames)
            data = data.squeeze()

        return data

    def find_masked_regions_in_spectrogram(self, original, masked, threshold_db=-50):
        """
        Find masked regions by comparing spectrograms.
        Returns binary mask where True indicates masked regions.
        """
        # Calculate difference in log-mel domain
        diff = original - masked

        # Find regions where the masked version is significantly lower
        # (indicating the original signal was zeroed before spectrogram conversion)
        masked_regions = diff > abs(threshold_db)

        # Also check for near-zero values in masked spectrogram
        near_zero_regions = masked < (np.min(original) + 1.0)  # Within 1 dB of minimum

        # Combine both conditions
        combined_mask = masked_regions | near_zero_regions

        return combined_mask

    def calculate_masking_statistics(self, original, masked):
        """Calculate statistics about the masking."""
        # Find masked regions
        mask = self.find_masked_regions_in_spectrogram(original, masked)

        total_timeframes = original.shape[1]
        total_freq_bins = original.shape[0]
        total_elements = total_timeframes * total_freq_bins

        # Count masked elements
        masked_elements = np.sum(mask)
        masking_percentage = (masked_elements / total_elements) * 100

        # Find masked time frames (any frequency bin masked in that time frame)
        masked_timeframes = np.any(mask, axis=0)
        masked_time_percentage = (np.sum(masked_timeframes) / total_timeframes) * 100

        # Estimate number of mask blocks (connected components in time)
        masked_time_indices = np.where(masked_timeframes)[0]
        if len(masked_time_indices) > 0:
            # Count blocks by finding discontinuities
            time_diffs = np.diff(masked_time_indices)
            num_blocks = np.sum(time_diffs > 1) + 1  # +1 for the first block
        else:
            num_blocks = 0

        # Calculate energy reduction
        original_energy = np.mean(np.exp(original))  # Convert back from log scale
        masked_energy = np.mean(np.exp(masked))
        energy_reduction_percent = ((original_energy - masked_energy) / original_energy) * 100

        return {
            'total_elements': total_elements,
            'masked_elements': masked_elements,
            'masking_percentage': masking_percentage,
            'masked_timeframes': np.sum(masked_timeframes),
            'total_timeframes': total_timeframes,
            'masked_time_percentage': masked_time_percentage,
            'estimated_blocks': num_blocks,
            'energy_reduction_percent': energy_reduction_percent,
            'mask_array': mask
        }

    def visualize_comparison(self, original, masked, filename_stem="comparison", save_plot=True):
        """Create visualization comparing original and masked spectrograms."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Masking Verification: {filename_stem}', fontsize=16)

        # Calculate statistics
        stats = self.calculate_masking_statistics(original, masked)
        mask = stats['mask_array']

        # 1. Original spectrogram
        im1 = axes[0, 0].imshow(original, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 0].set_title('Original Spectrogram')
        axes[0, 0].set_ylabel('Mel Frequency Bins')
        plt.colorbar(im1, ax=axes[0, 0])

        # 2. Masked spectrogram
        im2 = axes[0, 1].imshow(masked, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 1].set_title('Masked Spectrogram')
        plt.colorbar(im2, ax=axes[0, 1])

        # 3. Difference (original - masked)
        diff = original - masked
        im3 = axes[1, 0].imshow(diff, aspect='auto', origin='lower', cmap='RdBu')
        axes[1, 0].set_title('Difference (Original - Masked)')
        axes[1, 0].set_xlabel('Time Frames')
        axes[1, 0].set_ylabel('Mel Frequency Bins')
        plt.colorbar(im3, ax=axes[1, 0])

        # 4. Detected mask regions
        im4 = axes[1, 1].imshow(mask.astype(float), aspect='auto', origin='lower', cmap='Reds')
        axes[1, 1].set_title(f'Detected Masked Regions\n({stats["masking_percentage"]:.1f}% masked)')
        axes[1, 1].set_xlabel('Time Frames')
        plt.colorbar(im4, ax=axes[1, 1])

        plt.tight_layout()

        if save_plot:
            plot_path = Path(f'masking_verification_{filename_stem}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")

        plt.show()

        return stats

    def compare_files(self, original_file, masked_file):
        """Compare a single pair of original and masked files."""
        print(f"\nComparing:")
        print(f"  Original: {original_file}")
        print(f"  Masked:   {masked_file}")

        # Load spectrograms
        try:
            original = self.load_spectrogram(original_file)
            masked = self.load_spectrogram(masked_file)
        except Exception as e:
            print(f"Error loading files: {e}")
            return None

        print(f"  Shape: {original.shape}")

        # Calculate and display statistics
        stats = self.calculate_masking_statistics(original, masked)

        print("\nMasking Statistics:")
        print(f"  Total elements: {stats['total_elements']:,}")
        print(f"  Masked elements: {stats['masked_elements']:,}")
        print(f"  Masking percentage: {stats['masking_percentage']:.2f}%")
        print(
            f"  Masked time frames: {stats['masked_timeframes']}/{stats['total_timeframes']} ({stats['masked_time_percentage']:.1f}%)")
        print(f"  Estimated mask blocks: {stats['estimated_blocks']}")
        print(f"  Energy reduction: {stats['energy_reduction_percent']:.1f}%")

        # Create visualization
        filename_stem = original_file.stem
        viz_stats = self.visualize_comparison(original, masked, filename_stem)

        return stats

    def batch_compare_directories(self, max_files=10):
        """Compare multiple files from two directories."""
        original_files = list(self.original_path.glob("**/*.npy"))[:max_files]

        if not original_files:
            print(f"No .npy files found in {self.original_path}")
            return

        print(f"Found {len(original_files)} files to compare (showing first {max_files})")

        all_stats = []

        for orig_file in original_files:
            # Find corresponding masked file
            rel_path = orig_file.relative_to(self.original_path)
            masked_file = self.masked_path / rel_path

            if not masked_file.exists():
                print(f"Warning: Masked file not found: {masked_file}")
                continue

            stats = self.compare_files(orig_file, masked_file)
            if stats:
                stats['filename'] = orig_file.name
                all_stats.append(stats)

        # Summary statistics
        if all_stats:
            df = pd.DataFrame(all_stats)
            print(f"\n{'=' * 50}")
            print("SUMMARY STATISTICS ACROSS ALL FILES:")
            print(f"{'=' * 50}")
            print(
                f"Average masking percentage: {df['masking_percentage'].mean():.2f}% ± {df['masking_percentage'].std():.2f}%")
            print(
                f"Average masked time frames: {df['masked_time_percentage'].mean():.1f}% ± {df['masked_time_percentage'].std():.1f}%")
            print(f"Average estimated blocks: {df['estimated_blocks'].mean():.1f} ± {df['estimated_blocks'].std():.1f}")
            print(
                f"Average energy reduction: {df['energy_reduction_percent'].mean():.1f}% ± {df['energy_reduction_percent'].std():.1f}%")

            # Create summary plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Masking Statistics Summary', fontsize=16)

            axes[0, 0].hist(df['masking_percentage'], bins=10, alpha=0.7)
            axes[0, 0].set_title('Masking Percentage Distribution')
            axes[0, 0].set_xlabel('Masking %')

            axes[0, 1].hist(df['masked_time_percentage'], bins=10, alpha=0.7)
            axes[0, 1].set_title('Masked Time Percentage Distribution')
            axes[0, 1].set_xlabel('Masked Time %')

            axes[1, 0].hist(df['estimated_blocks'], bins=10, alpha=0.7)
            axes[1, 0].set_title('Estimated Blocks Distribution')
            axes[1, 0].set_xlabel('Number of Blocks')

            axes[1, 1].hist(df['energy_reduction_percent'], bins=10, alpha=0.7)
            axes[1, 1].set_title('Energy Reduction Distribution')
            axes[1, 1].set_xlabel('Energy Reduction %')

            plt.tight_layout()
            plt.savefig('data/masking_summary_stats.png', dpi=150, bbox_inches='tight')
            print(f"\nSummary plot saved to: masking_summary_stats.png")
            plt.show()


def main():
    parser = argparse.ArgumentParser(description='Verify EEG masking by comparing original and masked spectrograms')

    parser.add_argument('--original_dir', type=str, help='Directory containing original .npy files')
    parser.add_argument('--masked_dir', type=str, help='Directory containing masked .npy files')
    parser.add_argument('--original_file', type=str, help='Single original .npy file')
    parser.add_argument('--masked_file', type=str, help='Single masked .npy file')
    parser.add_argument('--max_files', type=int, default=10, help='Maximum files to process in batch mode')

    args = parser.parse_args()

    # Single file comparison
    if args.original_file and args.masked_file:
        verifier = MaskingVerifier(Path(args.original_file).parent, Path(args.masked_file).parent)
        verifier.compare_files(Path(args.original_file), Path(args.masked_file))

    # Directory comparison
    elif args.original_dir and args.masked_dir:
        verifier = MaskingVerifier(args.original_dir, args.masked_dir)
        verifier.batch_compare_directories(args.max_files)

    else:
        print("Please provide either:")
        print("  --original_dir and --masked_dir for batch comparison")
        print("  --original_file and --masked_file for single file comparison")
        print("\nExample usage:")
        print("  python verify_masking.py --original_dir /path/to/original --masked_dir /path/to/masked")
        print("  python verify_masking.py --original_file orig.npy --masked_file masked.npy")


if __name__ == "__main__":
    main()