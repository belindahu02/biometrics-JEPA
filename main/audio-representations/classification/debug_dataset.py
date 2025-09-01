# =============================================
# Dataset Debugging Script
# Run this to verify your dataset is loading correctly
# =============================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
from data_loader_2d_lazy import create_memory_efficient_dataloaders, get_file_paths_and_labels


def debug_dataset_loading(data_path, user_ids, samples_per_user=100,
                          normalization='log_scale', batch_size=16):
    """
    Comprehensive debugging of dataset loading
    """
    print("=" * 60)
    print("DATASET DEBUGGING STARTED")
    print("=" * 60)

    # 1. Check file paths and basic structure
    print("\n1. CHECKING FILE STRUCTURE...")
    file_paths, labels, sessions = get_file_paths_and_labels(data_path, user_ids)

    print(f"Total files found: {len(file_paths)}")
    print(f"Total labels: {len(labels)}")
    print(f"Unique labels: {np.unique(labels)}")
    print(f"Sessions per user: {sessions}")

    if len(file_paths) == 0:
        print("❌ CRITICAL ERROR: No files found!")
        return False

    # Check label distribution
    label_counts = Counter(labels)
    print(f"Label distribution: {dict(label_counts)}")

    # Check if labels are balanced
    min_count = min(label_counts.values())
    max_count = max(label_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"Class imbalance ratio: {imbalance_ratio:.2f} (should be < 3.0 for good training)")

    if imbalance_ratio > 5.0:
        print("⚠️  WARNING: Severe class imbalance detected!")

    # 2. Check a few sample files manually
    print("\n2. CHECKING SAMPLE FILES...")
    sample_indices = np.random.choice(len(file_paths), min(5, len(file_paths)), replace=False)

    for i, idx in enumerate(sample_indices):
        file_path = file_paths[idx]
        label = labels[idx]
        print(f"\nSample {i + 1}:")
        print(f"  File: {os.path.basename(file_path)}")
        print(f"  Label: {label} (User ID: {user_ids[label]})")

        try:
            spec = np.load(file_path)
            print(f"  Shape: {spec.shape}")
            print(f"  Dtype: {spec.dtype}")
            print(f"  Range: [{np.min(spec):.3f}, {np.max(spec):.3f}]")
            print(f"  Has NaN: {np.isnan(spec).any()}")
            print(f"  Has Inf: {np.isinf(spec).any()}")

            # Check for constant spectrograms (sign of corrupted data)
            if np.std(spec) < 1e-8:
                print(f"  ⚠️  WARNING: Spectrogram appears to be constant!")

        except Exception as e:
            print(f"  ❌ ERROR loading file: {e}")
            return False

    # 3. Test data loader creation
    print("\n3. TESTING DATA LOADER CREATION...")
    try:
        train_loader, val_loader, test_loader, _ = create_memory_efficient_dataloaders(
            data_path=data_path,
            user_ids=user_ids,
            samples_per_user=samples_per_user,
            normalization=normalization,
            batch_size=batch_size,
            augment_train=False,  # Disable augmentation for debugging
            cache_size=50  # Smaller cache for debugging
        )
        print("✅ Data loaders created successfully")
    except Exception as e:
        print(f"❌ ERROR creating data loaders: {e}")
        return False

    # 4. Check data loader outputs
    print("\n4. CHECKING DATA LOADER OUTPUTS...")

    # Check train loader
    print("TRAIN LOADER:")
    train_batch = next(iter(train_loader))
    X_batch, y_batch = train_batch
    print(f"  Batch shape: {X_batch.shape}")
    print(f"  Label shape: {y_batch.shape}")
    print(f"  Data type: {X_batch.dtype}")
    print(f"  Label type: {y_batch.dtype}")
    print(f"  Data range: [{X_batch.min().item():.3f}, {X_batch.max().item():.3f}]")
    print(f"  Labels in batch: {y_batch.unique().tolist()}")
    print(f"  Has NaN in batch: {torch.isnan(X_batch).any().item()}")
    print(f"  Has Inf in batch: {torch.isinf(X_batch).any().item()}")

    # Check validation loader
    print("\nVALIDATION LOADER:")
    val_batch = next(iter(val_loader))
    X_val, y_val = val_batch
    print(f"  Batch shape: {X_val.shape}")
    print(f"  Labels in batch: {y_val.unique().tolist()}")

    # Check test loader
    print("\nTEST LOADER:")
    test_batch = next(iter(test_loader))
    X_test, y_test = test_batch
    print(f"  Batch shape: {X_test.shape}")
    print(f"  Labels in batch: {y_test.unique().tolist()}")

    # 5. Check data splits
    print("\n5. CHECKING DATA SPLITS...")

    def check_split_labels(loader, split_name):
        all_labels = []
        for _, labels in loader:
            all_labels.extend(labels.tolist())

        label_counts = Counter(all_labels)
        print(f"{split_name}:")
        print(f"  Total samples: {len(all_labels)}")
        print(f"  Label distribution: {dict(label_counts)}")

        # Check if all classes are present
        expected_classes = set(range(len(user_ids)))
        actual_classes = set(all_labels)
        missing_classes = expected_classes - actual_classes
        if missing_classes:
            print(f"  ⚠️  WARNING: Missing classes: {missing_classes}")

        return all_labels

    train_labels_all = check_split_labels(train_loader, "TRAIN")
    val_labels_all = check_split_labels(val_loader, "VALIDATION")
    test_labels_all = check_split_labels(test_loader, "TEST")

    # 6. Check for data leakage between splits
    print("\n6. CHECKING FOR DATA LEAKAGE...")

    def get_file_paths_from_loader(dataset):
        return set(dataset.file_paths)

    train_files = get_file_paths_from_loader(train_loader.dataset)
    val_files = get_file_paths_from_loader(val_loader.dataset)
    test_files = get_file_paths_from_loader(test_loader.dataset)

    train_val_overlap = train_files & val_files
    train_test_overlap = train_files & test_files
    val_test_overlap = val_files & test_files

    if train_val_overlap:
        print(f"❌ CRITICAL: Train-Val overlap: {len(train_val_overlap)} files")
    if train_test_overlap:
        print(f"❌ CRITICAL: Train-Test overlap: {len(train_test_overlap)} files")
    if val_test_overlap:
        print(f"❌ CRITICAL: Val-Test overlap: {len(val_test_overlap)} files")

    if not (train_val_overlap or train_test_overlap or val_test_overlap):
        print("✅ No data leakage detected between splits")

    # 7. Visual inspection of a few spectrograms
    print("\n7. CREATING VISUAL INSPECTION PLOTS...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Show samples from different classes
    sample_batches = [next(iter(train_loader)) for _ in range(2)]

    plot_idx = 0
    for batch_idx, (X_batch, y_batch) in enumerate(sample_batches):
        for sample_idx in range(min(3, X_batch.shape[0])):
            if plot_idx >= 6:
                break

            spec = X_batch[sample_idx, 0].numpy()  # Remove channel dimension
            label = y_batch[sample_idx].item()
            user_id = user_ids[label]

            axes[plot_idx].imshow(spec.T, aspect='auto', origin='lower', cmap='viridis')
            axes[plot_idx].set_title(f'User {user_id} (Label {label})')
            axes[plot_idx].set_xlabel('Time')
            axes[plot_idx].set_ylabel('Frequency')

            plot_idx += 1

    plt.tight_layout()
    plt.savefig('dataset_debug_spectrograms.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 Saved spectrogram samples to 'dataset_debug_spectrograms.png'")

    # 8. Check normalization effectiveness
    print("\n8. CHECKING NORMALIZATION...")

    # Test different samples from each class
    sample_stats = {}
    for user_idx in range(len(user_ids)):
        user_samples = []
        sample_count = 0

        for X_batch, y_batch in train_loader:
            mask = y_batch == user_idx
            if mask.any():
                user_batch = X_batch[mask]
                user_samples.append(user_batch)
                sample_count += mask.sum().item()

                if sample_count >= 10:  # Get at least 10 samples per user
                    break

        if user_samples:
            all_user_data = torch.cat(user_samples, dim=0)
            stats = {
                'mean': all_user_data.mean().item(),
                'std': all_user_data.std().item(),
                'min': all_user_data.min().item(),
                'max': all_user_data.max().item(),
                'samples': all_user_data.shape[0]
            }
            sample_stats[f'User_{user_ids[user_idx]}'] = stats
            print(f"  User {user_ids[user_idx]}: mean={stats['mean']:.3f}, "
                  f"std={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}], "
                  f"samples={stats['samples']}")

    # Check if normalization is working consistently across users
    means = [stats['mean'] for stats in sample_stats.values()]
    stds = [stats['std'] for stats in sample_stats.values()]

    mean_variation = np.std(means)
    std_variation = np.std(stds)

    print(f"  Cross-user mean variation: {mean_variation:.4f}")
    print(f"  Cross-user std variation: {std_variation:.4f}")

    if mean_variation > 1.0:
        print("⚠️  WARNING: Large variation in means across users - normalization may not be working properly")

    # 9. Final diagnostic summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    issues_found = []

    if len(file_paths) == 0:
        issues_found.append("No data files found")

    if imbalance_ratio > 5.0:
        issues_found.append(f"Severe class imbalance (ratio: {imbalance_ratio:.2f})")

    if train_val_overlap or train_test_overlap or val_test_overlap:
        issues_found.append("Data leakage between splits detected")

    if mean_variation > 1.0:
        issues_found.append("Inconsistent normalization across users")

    if len(np.unique(train_labels_all)) != len(user_ids):
        issues_found.append("Not all classes present in training set")

    if len(np.unique(val_labels_all)) != len(user_ids):
        issues_found.append("Not all classes present in validation set")

    if issues_found:
        print("❌ ISSUES FOUND:")
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print("✅ No major issues detected in dataset loading")

    print(f"\nDataset summary:")
    print(f"  - {len(user_ids)} users")
    print(f"  - {len(file_paths)} total files")
    print(f"  - {len(train_labels_all)} training samples")
    print(f"  - {len(val_labels_all)} validation samples")
    print(f"  - {len(test_labels_all)} test samples")

    return len(issues_found) == 0


# Additional debugging functions
def check_label_consistency(data_path, user_ids):
    """
    Check if the same user appears in multiple files with different labels
    """
    print("\n" + "=" * 40)
    print("CHECKING LABEL CONSISTENCY")
    print("=" * 40)

    file_to_user = {}  # Map file paths to expected user IDs

    for user_idx, user_id in enumerate(user_ids):
        user_folder = f"S{user_id:03d}"
        user_path = os.path.join(data_path, user_folder)

        if os.path.exists(user_path):
            for item in os.listdir(user_path):
                session_path = os.path.join(user_path, item)
                if os.path.isdir(session_path):
                    for npy_file in os.listdir(session_path):
                        if npy_file.endswith("_stacked.npy"):
                            full_path = os.path.join(session_path, npy_file)
                            file_to_user[full_path] = user_id

    print(f"Found {len(file_to_user)} files with user mappings")

    # Now check if our labeling matches
    file_paths, labels, _ = get_file_paths_and_labels(data_path, user_ids)

    mismatches = 0
    for file_path, assigned_label in zip(file_paths, labels):
        expected_user_id = file_to_user.get(file_path)
        actual_user_id = user_ids[assigned_label]

        if expected_user_id != actual_user_id:
            print(f"❌ MISMATCH: {file_path}")
            print(f"   Expected User ID: {expected_user_id}")
            print(f"   Assigned User ID: {actual_user_id}")
            mismatches += 1

    if mismatches == 0:
        print("✅ All labels are consistent with file structure")
    else:
        print(f"❌ Found {mismatches} label mismatches!")

    return mismatches == 0


def quick_model_test(train_loader, val_loader, num_classes, device='cpu'):
    """
    Quick test to see if a simple model can learn on your data
    """
    print("\n" + "=" * 40)
    print("QUICK MODEL LEARNING TEST")
    print("=" * 40)

    # Get input shape from a batch
    sample_batch = next(iter(train_loader))
    input_shape = sample_batch[0].shape[1:]  # Remove batch dimension

    # Create a very simple model
    class SimpleTestModel(torch.nn.Module):
        def __init__(self, input_shape, num_classes):
            super().__init__()
            self.flatten = torch.nn.Flatten()
            input_size = np.prod(input_shape)
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(input_size, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes)
            )

        def forward(self, x):
            x = self.flatten(x)
            return self.classifier(x)

    model = SimpleTestModel(input_shape, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Testing with simple model on device: {device}")
    print(f"Input shape: {input_shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train for a few epochs
    model.train()
    initial_loss = None

    for epoch in range(5):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

            # Only train on first few batches for quick test
            if batch_idx >= 10:
                break

        accuracy = 100. * correct / total
        avg_loss = epoch_loss / (batch_idx + 1)

        if epoch == 0:
            initial_loss = avg_loss

        print(f"  Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")

    # Check if model is learning
    final_loss = avg_loss
    loss_reduction = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0

    print(f"\nLearning check:")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Loss reduction: {loss_reduction:.2%}")
    print(f"  Final accuracy: {accuracy:.2f}%")

    random_accuracy = 100.0 / num_classes
    print(f"  Random baseline: {random_accuracy:.2f}%")

    if accuracy > random_accuracy * 1.5 and loss_reduction > 0.1:
        print("✅ Model appears to be learning!")
        return True
    else:
        print("❌ Model does not appear to be learning properly")
        return False


# Main debugging function
def run_full_debug(data_path, user_ids, samples_per_user=100):
    """
    Run complete debugging suite
    """
    print("Starting comprehensive dataset debugging...\n")

    # Run all checks
    dataset_ok = debug_dataset_loading(data_path, user_ids, samples_per_user)
    labels_ok = check_label_consistency(data_path, user_ids)

    if dataset_ok and labels_ok:
        print("\n🔬 Running quick learning test...")

        # Create data loaders for testing
        train_loader, val_loader, test_loader, _ = create_memory_efficient_dataloaders(
            data_path=data_path,
            user_ids=user_ids,
            samples_per_user=samples_per_user,
            normalization='log_scale',
            batch_size=16,
            augment_train=False,
            cache_size=20
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        learning_ok = quick_model_test(train_loader, val_loader, len(user_ids), device)

        print("\n" + "=" * 60)
        print("FINAL DIAGNOSIS")
        print("=" * 60)

        if dataset_ok and labels_ok and learning_ok:
            print("✅ Dataset appears to be loaded correctly!")
            print("   The issue is likely in your model architecture, hyperparameters, or training loop.")
        else:
            print("❌ Issues detected in dataset loading:")
            if not dataset_ok:
                print("   - Dataset structure or loading problems")
            if not labels_ok:
                print("   - Label consistency problems")
            if not learning_ok:
                print("   - Data may not be learnable or has fundamental issues")
    else:
        print("\n❌ Critical issues found in dataset. Fix these before training.")


# Example usage:
if __name__ == "__main__":
    # Replace with your actual data path and user IDs
    DATA_PATH = "/app/data/grouped_embeddings"
    USER_IDS = list(range(1, 110))  # Increased to more users for better evaluation
    SAMPLES_PER_USER = 133

    run_full_debug(DATA_PATH, USER_IDS, SAMPLES_PER_USER)