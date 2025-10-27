"""
Main script to run the EEG masking experiment
Integrates all components: masking, preprocessing, training, and evaluation
WITH CHECKPOINT SUPPORT FOR RESUMING FROM INTERRUPTIONS
UPDATED: Now uses same hyperparameters and model configuration as scaling experiment
"""

import sys
import numpy as np
import torch
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


# Import your modules
from utils import EEGMaskingProcessor, create_masked_dataset_for_experiment, run_embedding_extraction, run_embedding_grouping
from data_loader import create_masking_experiment_dataloaders, MaskingExperimentTrainer
from backbones import LightweightSpectrogramResNet, SpectrogramResNet  # UPDATED: Changed from LightweightSpectrogramResNet
from checkpoint import MaskingExperimentCheckpoint


class EEGMaskingExperiment:
    """Complete EEG masking experiment pipeline"""

    def __init__(self, config):
        self.config = config
        self.experiment_dir = Path(config['experiment_dir'])
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path("/app/data/experiments/masking_full_time_mask/data")

        # Create subdirectories
#        self.data_dir = self.experiment_dir / "data"
        self.models_dir = self.experiment_dir / "models"
        self.results_dir = self.experiment_dir / "results"
        self.plots_dir = self.experiment_dir / "plots"

        for d in [self.data_dir, self.models_dir, self.results_dir, self.plots_dir]:
            d.mkdir(exist_ok=True)

        # Initialize checkpoint manager
        checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_mgr = MaskingExperimentCheckpoint(checkpoint_dir)

        # Masking parameter grid
        self.masking_percentages = np.arange(0, 55, 5)  # 0%, 5%, 10%, ..., 50%
        self.num_blocks_range = [1, 2, 3, 4, 5]

        # Store trained model
        self.base_model_path = None

        print(f"EEG Masking Experiment initialized")
        print(f"Experiment directory: {self.experiment_dir}")
        print(f"Grid size: {len(self.masking_percentages)} × {len(self.num_blocks_range)} = {len(self.masking_percentages) * len(self.num_blocks_range)}")
        print(f"Model configuration: {config['model_type']} with cosine classifier")
        print(f"Training hyperparameters:")
        print(f"  - Batch size: {config['batch_size']}")
        print(f"  - Learning rate: {config['learning_rate']}")
        print(f"  - Epochs: {config['train_epochs']}")
        print(f"  - Cosine scale: {config.get('cosine_scale', 40.0)}")
        print(f"  - Label smoothing: {config.get('label_smoothing', 0.1)}")

        # Check for existing checkpoint
        if self.checkpoint_mgr.has_checkpoint():
            print("⚠️  EXISTING CHECKPOINT FOUND")
            print("The experiment can be resumed from the last saved state")
            print("Set resume=True in run_complete_experiment() to continue")

    def step1_train_base_model(self):
        """Train the base model on unmasked training data (sessions 1-10)"""
        print("\n" + "="*60)
        print("STEP 1: Training Base Model (Unmasked Data)")
        print("="*60)

        # Generate unmasked training data for sessions 1-12 (1-10 train, 11-12 val)
        print("Generating unmasked training/validation data...")
        train_val_sessions = list(range(1, 13))  # Sessions 1-12

        unmasked_dir = self.data_dir / "unmasked_train_val"
        if unmasked_dir.exists() and any(unmasked_dir.iterdir()):
           print(f"Dataset already exists at {unmasked_dir}, skipping creation.")
        else:
            create_masked_dataset_for_experiment(
                raw_eeg_dir=self.config['raw_eeg_dir'],
                output_dir=unmasked_dir,
                user_ids=self.config['user_ids'],
                session_nums=train_val_sessions,
                masking_percentage=0,  # No masking
                num_blocks=1
            )

        # Run preprocessing pipeline
        print("Running preprocessing pipeline...")
        embeddings_dir = self.data_dir / "embeddings_train_val"
        grouped_dir = self.data_dir / "grouped_train_val"

        # Check if preprocessing has already been done
        embeddings_exist = embeddings_dir.exists() and any(embeddings_dir.iterdir())
        grouped_exist = grouped_dir.exists() and any(grouped_dir.iterdir())

        # Extract embeddings using pre-trained JEPA encoder
        if not embeddings_exist:
            print("Extracting embeddings...")
            run_embedding_extraction(
                csv_path=unmasked_dir / "files_audioset.csv",
                data_dir=unmasked_dir,
                embeddings_dir=embeddings_dir,
                model_checkpoint_path=self.config['model_checkpoint_path'],
                config_path=self.config['model_config_path'],
                batch_size=self.config['batch_size'],
                num_workers=2,
                device=self.device if hasattr(self, 'device') else None
            )
        else:
            print(f"Embeddings already exist in {embeddings_dir}, skipping extraction...")

        # Group embeddings
        if not grouped_exist:
            print("Grouping embeddings...")
            run_embedding_grouping(embeddings_dir, grouped_dir)
        else:
            print(f"Grouped embeddings already exist in {grouped_dir}, skipping grouping...")

        # Create data loaders
        print("Creating data loaders for training...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Modified to only use training sessions for this step
        train_loader, val_loader, _ = create_masking_experiment_dataloaders(
            data_dir=grouped_dir,
            user_ids=self.config['user_ids'],
            batch_size=self.config['batch_size']
        )

        # Get input shape from sample batch
        sample_batch = next(iter(train_loader))
        input_shape = sample_batch[0].shape[1:]  # Remove batch dim
        num_classes = len(self.config['user_ids'])
        print(f"Input shape: {input_shape}")
        print(f"Number of classes: {num_classes}")

        # Initialize model based on config
        model_type = self.config.get('model_type', 'lightweight')
        print(f"Initializing {model_type} model...")

        if model_type == 'full':
            # Full model with more capacity
            model = SpectrogramResNet(
                input_channels=input_shape[0],
                num_classes=num_classes,
                channels=[64, 128, 256, 512],  # Full model channels
                dropout_rate=self.config.get('dropout_rate', 0.0),
                classifier_dropout=self.config.get('classifier_dropout', 0.0)
            )
            print(f"  Architecture: Full ResNet with channels [64, 128, 256, 512]")
        elif model_type == 'lightweight':
            # Lightweight model (baseline)
            model = LightweightSpectrogramResNet(
                input_channels=input_shape[0],
                num_classes=num_classes,
                channels=[32, 64, 128]  # Lightweight model channels
            )
            print(f"  Architecture: Lightweight ResNet with channels [32, 64, 128]")
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'full' or 'lightweight'")


        # UPDATED: Initialize trainer with cosine classifier and advanced training techniques
        trainer = MaskingExperimentTrainer(
            model,
            device=device,
            use_cosine_classifier=self.config.get('use_cosine_classifier', True),
            cosine_scale=self.config.get('cosine_scale', 40.0),
            label_smoothing=self.config.get('label_smoothing', 0.1),
            warmup_epochs=self.config.get('warmup_epochs', 5)
        )

        print(f"Training model for {self.config['train_epochs']} epochs...")
        print(f"Using advanced training techniques:")
        print(f"  - Cosine classifier: {self.config.get('use_cosine_classifier', True)}")
        print(f"  - Cosine scale: {self.config.get('cosine_scale', 40.0)}")
        print(f"  - Label smoothing: {self.config.get('label_smoothing', 0.1)}")
        print(f"  - Warmup epochs: {self.config.get('warmup_epochs', 5)}")
        print(f"  - Weight decay: {self.config.get('weight_decay', 1e-4)}")

        # UPDATED: Train with weight decay parameter
        best_val_acc = trainer.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.config['train_epochs'],
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )

        # Save trained model
        self.base_model_path = self.models_dir / "base_model.pth"
        trainer.save_model(self.base_model_path)

        print(f"Base model training completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Model saved to: {self.base_model_path}")

        return best_val_acc

    def step2_generate_test_data_variants(self):
        """Generate test data with different masking configurations"""
        print("\n" + "="*60)
        print("STEP 2: Generating Test Data Variants")
        print("="*60)

        test_sessions = self.config['test_sessions']  # Sessions 13-14

        print(f"Checking/generating {len(self.masking_percentages) * len(self.num_blocks_range)} test data variants...")

        test_data_dirs = {}

        for mask_pct in self.masking_percentages:
            for num_blocks in self.num_blocks_range:
                # Create unique identifier
                variant_id = f"test_mask_{mask_pct}pct_{num_blocks}blocks"
                variant_dir = self.data_dir / variant_id
                embeddings_dir = self.data_dir / f"embeddings_{variant_id}"
                grouped_dir = self.data_dir / f"grouped_{variant_id}"

                # Check if this variant already exists (all 3 stages)
                variant_exists = variant_dir.exists() and any(variant_dir.iterdir())
                embeddings_exist = embeddings_dir.exists() and any(embeddings_dir.iterdir())
                grouped_exist = grouped_dir.exists() and any(grouped_dir.iterdir())

                if variant_exists and embeddings_exist and grouped_exist:
                    print(f"✓ Variant {mask_pct}%/{num_blocks}blocks already exists, skipping")
                    test_data_dirs[(mask_pct, num_blocks)] = grouped_dir
                    continue

                print(f"Creating test data: {mask_pct}% masking, {num_blocks} blocks")

                # Generate masked test data (only if needed)
                if not variant_exists:
                    create_masked_dataset_for_experiment(
                        raw_eeg_dir=self.config['raw_eeg_dir'],
                        output_dir=variant_dir,
                        user_ids=self.config['user_ids'],
                        session_nums=test_sessions,
                        masking_percentage=mask_pct,
                        num_blocks=num_blocks
                    )
                else:
                    print(f"  Raw data exists, skipping generation")

                # Extract embeddings (only if needed)
                if not embeddings_exist:
                    print(f"  Extracting embeddings...")
                    run_embedding_extraction(
                        csv_path=variant_dir / "files_audioset.csv",
                        data_dir=variant_dir,
                        embeddings_dir=embeddings_dir,
                        model_checkpoint_path=self.config['model_checkpoint_path'],
                        config_path=self.config['model_config_path'],
                        batch_size=self.config['batch_size'],
                        num_workers=2,
                        device=self.device if hasattr(self, 'device') else None
                    )
                else:
                    print(f"  Embeddings exist, skipping extraction")

                # Group embeddings (only if needed)
                if not grouped_exist:
                    print(f"  Grouping embeddings...")
                    run_embedding_grouping(embeddings_dir, grouped_dir)
                else:
                    print(f"  Grouped data exists, skipping grouping")

                test_data_dirs[(mask_pct, num_blocks)] = grouped_dir

        print(f"All {len(test_data_dirs)} test data variants ready")
        return test_data_dirs

    def step3_evaluate_masking_effects(self, test_data_dirs, resume_state=None):
        """
        Evaluate the trained model on all masking variants with checkpoint support

        Args:
            test_data_dirs: Dictionary of test data directories
            resume_state: Tuple of (accuracy_matrix, kappa_matrix, detailed_results) if resuming
        """
        print("\n" + "="*60)
        print("STEP 3: Evaluating Masking Effects")
        print("="*60)

        if self.base_model_path is None:
            raise ValueError("Base model not trained. Run step1_train_base_model first.")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # UPDATED: Load trained model with same configuration as training
        num_classes = len(self.config['user_ids'])
        model_type = self.config.get('model_type', 'lightweight')

        print(f"Loading {model_type} model for evaluation...")

        if model_type == 'full':
            # Full model - must match training configuration
            model = SpectrogramResNet(
                input_channels=1,
                num_classes=num_classes,
                channels=[64, 128, 256, 512],
                dropout_rate=self.config.get('dropout_rate', 0.0),
                classifier_dropout=self.config.get('classifier_dropout', 0.0)
            )
        elif model_type == 'lightweight':
            # Lightweight model - must match training configuration
            model = LightweightSpectrogramResNet(
                input_channels=1,
                num_classes=num_classes,
                channels=[32, 64, 128]
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        )

        # UPDATED: Initialize trainer with same configuration as training
        trainer = MaskingExperimentTrainer(
            model,
            device=device,
            use_cosine_classifier=self.config.get('use_cosine_classifier', True),
            cosine_scale=self.config.get('cosine_scale', 40.0),
            label_smoothing=self.config.get('label_smoothing', 0.1),
            warmup_epochs=self.config.get('warmup_epochs', 5)
        )
        trainer.load_model(self.base_model_path)

        print(f"Loaded trained model from: {self.base_model_path}")

        # Initialize or load existing results
        if resume_state:
            accuracy_matrix, kappa_matrix, detailed_results = resume_state
            completed_combinations = {(r['masking_percentage'], r['num_blocks']) for r in detailed_results}
            print(f"Resuming evaluation: {len(completed_combinations)} combinations already completed")
        else:
            accuracy_matrix = np.zeros((len(self.masking_percentages), len(self.num_blocks_range)))
            kappa_matrix = np.zeros((len(self.masking_percentages), len(self.num_blocks_range)))
            detailed_results = []
            completed_combinations = set()

        total_evaluations = len(self.masking_percentages) * len(self.num_blocks_range)
        current_eval = len(completed_combinations)

        print(f"Total evaluations: {total_evaluations}")
        print(f"Remaining: {total_evaluations - current_eval}")

        for i, mask_pct in enumerate(self.masking_percentages):
            for j, num_blocks in enumerate(self.num_blocks_range):

                # Skip if already completed
                if (mask_pct, num_blocks) in completed_combinations:
                    print(f"Skipping {mask_pct}% / {num_blocks} blocks (already completed)")
                    continue

                current_eval += 1

                print(f"\nEvaluation {current_eval}/{total_evaluations}: {mask_pct}% masking, {num_blocks} blocks")

                try:
                    # Get test data directory
                    grouped_dir = test_data_dirs[(mask_pct, num_blocks)]

                    # Create test data loader (only test sessions)
                    test_dataset = TestOnlyDataset(grouped_dir, self.config['user_ids'])
                    test_loader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=self.config['batch_size'],
                        shuffle=False
                    )

                    if len(test_loader) == 0:
                        print(f"  No test data found for this variant")
                        accuracy_matrix[i, j] = np.nan
                        kappa_matrix[i, j] = np.nan
                        continue

                    # Run multiple evaluations and average
                    accuracies = []
                    kappa_scores = []

                    for run in range(self.config['eval_runs_per_variant']):
                        acc, kappa = trainer.evaluate_with_kappa(test_loader)
                        accuracies.append(acc)
                        kappa_scores.append(kappa)

                    avg_acc = np.mean(accuracies)
                    avg_kappa = np.mean(kappa_scores)
                    std_acc = np.std(accuracies)
                    std_kappa = np.std(kappa_scores)

                    accuracy_matrix[i, j] = avg_acc
                    kappa_matrix[i, j] = avg_kappa

                    result_record = {
                        'masking_percentage': mask_pct,
                        'num_blocks': num_blocks,
                        'avg_accuracy': avg_acc,
                        'std_accuracy': std_acc,
                        'avg_kappa': avg_kappa,
                        'std_kappa': std_kappa,
                        'accuracies': accuracies,
                        'kappa_scores': kappa_scores,
                        'num_runs': len(accuracies)
                    }
                    detailed_results.append(result_record)
                    completed_combinations.add((mask_pct, num_blocks))

                    print(f"  Results: Acc={avg_acc:.4f}±{std_acc:.4f}, Kappa={avg_kappa:.4f}±{std_kappa:.4f}")

                    # Save checkpoint after each variant (every 5 variants to avoid overhead)
                    if current_eval % 5 == 0:
                        self.checkpoint_mgr.save_checkpoint(
                            stage='evaluating',
                            stage_data={
                                'base_model_path': str(self.base_model_path),
                                'completed_variants': list(completed_combinations),
                                'current_progress': f"{current_eval}/{total_evaluations}"
                            },
                            accuracy_matrix=accuracy_matrix,
                            kappa_matrix=kappa_matrix,
                            detailed_results=detailed_results,
                            metadata={
                                'step': 3,
                                'progress': f"{current_eval}/{total_evaluations}",
                                'description': 'Evaluation in progress'
                            }
                        )
                        print(f"  Checkpoint saved ({current_eval}/{total_evaluations} completed)")

                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
                    accuracy_matrix[i, j] = np.nan
                    kappa_matrix[i, j] = np.nan

                    # Save checkpoint even on error
                    self.checkpoint_mgr.save_checkpoint(
                        stage='evaluating',
                        stage_data={
                            'base_model_path': str(self.base_model_path),
                            'completed_variants': list(completed_combinations),
                            'current_progress': f"{current_eval}/{total_evaluations}",
                            'last_error': str(e)
                        },
                        accuracy_matrix=accuracy_matrix,
                        kappa_matrix=kappa_matrix,
                        detailed_results=detailed_results,
                        metadata={'step': 3, 'progress': f"{current_eval}/{total_evaluations}"}
                    )

        return accuracy_matrix, kappa_matrix, detailed_results

    def step4_create_visualizations(self, accuracy_matrix, kappa_matrix, detailed_results):
        """Create heatmaps and other visualizations"""
        print("\n" + "="*60)
        print("STEP 4: Creating Visualizations")
        print("="*60)

        # Accuracy heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            accuracy_matrix,
            xticklabels=self.num_blocks_range,
            yticklabels=[f"{p}%" for p in self.masking_percentages],
            annot=True,
            fmt='.3f',
            cmap='viridis',
            cbar_kws={'label': 'Test Accuracy'},
            vmin=0,
            vmax=1
        )
        plt.title('EEG Classification Accuracy vs Masking Configuration\n(Y: Masking %, X: Number of Blocks)',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Number of Masking Blocks', fontsize=12)
        plt.ylabel('Total Masking Percentage', fontsize=12)
        plt.tight_layout()

        acc_heatmap_path = self.plots_dir / "accuracy_heatmap.png"
        plt.savefig(acc_heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Kappa heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            kappa_matrix,
            xticklabels=self.num_blocks_range,
            yticklabels=[f"{p}%" for p in self.masking_percentages],
            annot=True,
            fmt='.3f',
            cmap='viridis',
            cbar_kws={'label': 'Cohen\'s Kappa'}
        )
        plt.title('EEG Classification Kappa vs Masking Configuration\n(Y: Masking %, X: Number of Blocks)',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Number of Masking Blocks', fontsize=12)
        plt.ylabel('Total Masking Percentage', fontsize=12)
        plt.tight_layout()

        kappa_heatmap_path = self.plots_dir / "kappa_heatmap.png"
        plt.savefig(kappa_heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Performance degradation heatmap
        baseline_acc = accuracy_matrix[0, 0]  # 0% masking
        if not np.isnan(baseline_acc):
            degradation_matrix = (baseline_acc - accuracy_matrix) / baseline_acc * 100

            plt.figure(figsize=(12, 10))
            sns.heatmap(
                degradation_matrix,
                xticklabels=self.num_blocks_range,
                yticklabels=[f"{p}%" for p in self.masking_percentages],
                annot=True,
                fmt='.1f',
                cmap='Reds',
                cbar_kws={'label': 'Performance Degradation (%)'},
                vmin=0,
                vmax=100
            )
            plt.title('Performance Degradation vs Baseline (0% Masking)\n(Y: Masking %, X: Number of Blocks)',
                      fontsize=14, fontweight='bold')
            plt.xlabel('Number of Masking Blocks', fontsize=12)
            plt.ylabel('Total Masking Percentage', fontsize=12)
            plt.tight_layout()

            deg_heatmap_path = self.plots_dir / "degradation_heatmap.png"
            plt.savefig(deg_heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()

        # Trend plots
        self._create_trend_plots(accuracy_matrix, kappa_matrix)

        print(f"Visualizations saved to: {self.plots_dir}")

    def _create_trend_plots(self, accuracy_matrix, kappa_matrix):
        """Create trend analysis plots"""

        # Accuracy vs masking percentage for different numbers of blocks
        plt.figure(figsize=(12, 8))
        for j, num_blocks in enumerate(self.num_blocks_range):
            accuracies = accuracy_matrix[:, j]
            plt.plot(self.masking_percentages, accuracies, 'o-',
                    label=f'{num_blocks} block(s)', linewidth=2, markersize=6)

        plt.xlabel('Masking Percentage (%)')
        plt.ylabel('Test Accuracy')
        plt.title('EEG Classification Accuracy vs Masking Percentage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        acc_trend_path = self.plots_dir / "accuracy_trend_by_percentage.png"
        plt.savefig(acc_trend_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Accuracy vs number of blocks for different masking percentages
        plt.figure(figsize=(12, 8))
        for i, mask_pct in enumerate(self.masking_percentages[1:], 1):  # Skip 0%
            accuracies = accuracy_matrix[i, :]
            plt.plot(self.num_blocks_range, accuracies, 'o-',
                    label=f'{mask_pct}% masking', linewidth=2, markersize=6)

        plt.xlabel('Number of Masking Blocks')
        plt.ylabel('Test Accuracy')
        plt.title('EEG Classification Accuracy vs Number of Masking Blocks')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        blocks_trend_path = self.plots_dir / "accuracy_trend_by_blocks.png"
        plt.savefig(blocks_trend_path, dpi=300, bbox_inches='tight')
        plt.close()

    def step5_save_results(self, accuracy_matrix, kappa_matrix, detailed_results, base_val_acc):
        """Save all experiment results"""
        print("\n" + "="*60)
        print("STEP 5: Saving Results")
        print("="*60)

        # Create comprehensive results dictionary
        results_data = {
            'experiment_config': self.config,
            'experiment_timestamp': datetime.now().isoformat(),
            'masking_percentages': self.masking_percentages.tolist(),
            'num_blocks_range': self.num_blocks_range,
            'base_model_validation_accuracy': base_val_acc,
            'accuracy_matrix': accuracy_matrix.tolist(),
            'kappa_matrix': kappa_matrix.tolist(),
            'detailed_results': detailed_results,
            'summary_statistics': self._calculate_summary_stats(accuracy_matrix, kappa_matrix)
        }

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(x) for x in obj]
            return obj

        # -----------------------------
        # Save as JSON
        # -----------------------------
        json_path = self.results_dir / "experiment_results.json"
        with open(json_path, 'w') as f:
            json.dump(convert_numpy(results_data), f, indent=2)

        # -----------------------------
        # Save as pickle
        # -----------------------------
        import pickle
        pickle_path = self.results_dir / "experiment_results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results_data, f)

        # -----------------------------
        # Save matrices as NumPy arrays
        # -----------------------------
        npz_path = self.results_dir / "results_matrices.npz"
        np.savez(
            npz_path,
            accuracy_matrix=accuracy_matrix,
            kappa_matrix=kappa_matrix,
            detailed_results=np.array(detailed_results, dtype=object),
            base_val_acc=float(base_val_acc),
            masking_percentages=np.array(self.masking_percentages),
            num_blocks_range=np.array(self.num_blocks_range)
        )

        # -----------------------------
        # Print saved paths
        # -----------------------------
        print(f"Results saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Pickle: {pickle_path}")
        print(f"  NumPy: {npz_path}")

        return results_data

    def _calculate_summary_stats(self, accuracy_matrix, kappa_matrix):
        """Calculate summary statistics"""
        valid_acc_mask = ~np.isnan(accuracy_matrix)
        valid_kappa_mask = ~np.isnan(kappa_matrix)

        baseline_acc = accuracy_matrix[0, 0] if not np.isnan(accuracy_matrix[0, 0]) else np.nan
        baseline_kappa = kappa_matrix[0, 0] if not np.isnan(kappa_matrix[0, 0]) else np.nan

        stats = {
            'baseline_accuracy': float(baseline_acc),
            'baseline_kappa': float(baseline_kappa),
            'accuracy_stats': {
                'mean': float(np.nanmean(accuracy_matrix)),
                'std': float(np.nanstd(accuracy_matrix)),
                'min': float(np.nanmin(accuracy_matrix)),
                'max': float(np.nanmax(accuracy_matrix)),
                'valid_measurements': int(np.sum(valid_acc_mask))
            },
            'kappa_stats': {
                'mean': float(np.nanmean(kappa_matrix)),
                'std': float(np.nanstd(kappa_matrix)),
                'min': float(np.nanmin(kappa_matrix)),
                'max': float(np.nanmax(kappa_matrix)),
                'valid_measurements': int(np.sum(valid_kappa_mask))
            }
        }

        # Performance degradation analysis
        if not np.isnan(baseline_acc):
            degradation_matrix = (baseline_acc - accuracy_matrix) / baseline_acc * 100
            stats['degradation_stats'] = {
                'mean_degradation_pct': float(np.nanmean(degradation_matrix[1:])),
                'max_degradation_pct': float(np.nanmax(degradation_matrix[1:])),
                'min_degradation_pct': float(np.nanmin(degradation_matrix[1:]))
            }

        return stats

    def run_complete_experiment(self, resume=True):
        """
        Run the complete masking experiment pipeline with checkpoint support

        Args:
            resume: If True, resume from checkpoint if available. If False, start fresh.
        """
        print("Starting EEG Masking Experiment")
        print(f"Configuration: {json.dumps(self.config, indent=2)}")

        experiment_start_time = time.time()

        # Check for existing checkpoint
        checkpoint_state = None
        if resume and self.checkpoint_mgr.has_checkpoint():
            print("\n" + "="*60)
            print("RESUMING FROM CHECKPOINT")
            print("="*60)
            checkpoint_state = self.checkpoint_mgr.load_checkpoint()

            if checkpoint_state:
                completed_stage = checkpoint_state['checkpoint_data']['stage']
                print(f"Last completed stage: {completed_stage}")
                print(f"Timestamp: {checkpoint_state['checkpoint_data']['timestamp']}")

                # Load existing results
                if checkpoint_state.get('accuracy_matrix') is not None:
                    print(f"Loaded existing accuracy matrix: {checkpoint_state['accuracy_matrix'].shape}")
                if checkpoint_state.get('detailed_results'):
                    print(f"Loaded {len(checkpoint_state['detailed_results'])} existing results")

        try:
            # Step 1: Train base model (or load if already trained)
            if checkpoint_state and checkpoint_state['checkpoint_data']['stage'] != 'init':
                print("\n✓ Step 1: Base model already trained, loading...")
                stage_data = checkpoint_state['checkpoint_data']['stage_data']
                self.base_model_path = Path(stage_data.get('base_model_path'))
                base_val_acc = stage_data.get('base_val_acc', 0.0)
                print(f"  Model path: {self.base_model_path}")
                print(f"  Validation accuracy: {base_val_acc:.4f}")
            else:
                base_val_acc = self.step1_train_base_model()

                # Save checkpoint after training
                self.checkpoint_mgr.save_checkpoint(
                    stage='train_complete',
                    stage_data={
                        'base_model_path': str(self.base_model_path),
                        'base_val_acc': float(base_val_acc)
                    },
                    metadata={'step': 1, 'description': 'Base model training completed'}
                )

            # Step 2: Generate test data variants (or skip if already done)
            if checkpoint_state and checkpoint_state['checkpoint_data']['stage'] in ['variants_complete', 'evaluate_complete', 'complete']:
                print("\n✓ Step 2: Test data variants already generated, loading...")
                stage_data = checkpoint_state['checkpoint_data']['stage_data']
                # Convert string keys back to tuple keys
                test_data_dirs = {}
                for k, v in stage_data.get('test_data_dirs', {}).items():
                    # Parse string representation of tuple back to tuple
                    # Format is like "(0, 1)" -> (0, 1)
                    mask_pct, num_blocks = eval(k)
                    test_data_dirs[(mask_pct, num_blocks)] = Path(v)
                print(f"  Loaded {len(test_data_dirs)} test data variants")
            else:
                test_data_dirs = self.step2_generate_test_data_variants()

                # Save checkpoint after generating variants
                test_data_dirs_serializable = {str(k): str(v) for k, v in test_data_dirs.items()}
                self.checkpoint_mgr.save_checkpoint(
                    stage='variants_complete',
                    stage_data={
                        'base_model_path': str(self.base_model_path),
                        'base_val_acc': float(base_val_acc),
                        'test_data_dirs': test_data_dirs_serializable
                    },
                    metadata={'step': 2, 'description': 'Test data variants generated'}
                )

            # Step 3: Evaluate masking effects (with checkpointing per variant)
            if checkpoint_state and checkpoint_state['checkpoint_data']['stage'] in ['evaluate_complete', 'complete']:
                print("\n✓ Step 3: Evaluation already complete, loading results...")
                accuracy_matrix = checkpoint_state.get('accuracy_matrix')
                kappa_matrix = checkpoint_state.get('kappa_matrix')
                detailed_results = checkpoint_state.get('detailed_results', [])
                print(f"  Loaded evaluation results: {len(detailed_results)} variants")
            else:
                # Resume evaluation from checkpoint if available
                if checkpoint_state and 'accuracy_matrix' in checkpoint_state:
                    accuracy_matrix = checkpoint_state['accuracy_matrix']
                    kappa_matrix = checkpoint_state['kappa_matrix']
                    detailed_results = checkpoint_state.get('detailed_results') or []
                    print(f"  Resuming evaluation with {len(detailed_results)} completed variants")
                else:
                    accuracy_matrix = None
                    kappa_matrix = None
                    detailed_results = []

                accuracy_matrix, kappa_matrix, detailed_results = self.step3_evaluate_masking_effects(
                    test_data_dirs,
                    resume_state=(accuracy_matrix, kappa_matrix, detailed_results) if accuracy_matrix is not None else None
                )

                # Save checkpoint after evaluation
                self.checkpoint_mgr.save_checkpoint(
                    stage='evaluate_complete',
                    stage_data={
                        'base_model_path': str(self.base_model_path),
                        'base_val_acc': float(base_val_acc),
                        'evaluation_complete': True
                    },
                    accuracy_matrix=accuracy_matrix,
                    kappa_matrix=kappa_matrix,
                    detailed_results=detailed_results,
                    metadata={'step': 3, 'description': 'Evaluation completed'}
                )

            # Step 4: Create visualizations
            self.step4_create_visualizations(accuracy_matrix, kappa_matrix, detailed_results)

            # Step 5: Save results
            results_data = self.step5_save_results(accuracy_matrix, kappa_matrix, detailed_results, base_val_acc)

            # Mark experiment as complete
            self.checkpoint_mgr.save_checkpoint(
                stage='complete',
                stage_data={
                    'base_model_path': str(self.base_model_path),
                    'base_val_acc': float(base_val_acc),
                    'experiment_complete': True,
                    'total_variants_evaluated': len(detailed_results)
                },
                accuracy_matrix=accuracy_matrix,
                kappa_matrix=kappa_matrix,
                detailed_results=detailed_results,
                metadata={'step': 5, 'description': 'Experiment completed successfully'}
            )

            # Final summary
            experiment_time = time.time() - experiment_start_time

            print("\n" + "="*60)
            print("EXPERIMENT COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Total experiment time: {experiment_time/3600:.2f} hours")
            print(f"Baseline accuracy (0% masking): {accuracy_matrix[0,0]:.4f}")
            print(f"Results directory: {self.experiment_dir}")

            # Print key findings
            valid_mask = ~np.isnan(accuracy_matrix[1:])
            if valid_mask.any():
                masked_accuracies = accuracy_matrix[1:][valid_mask]
                best_masked_acc = np.max(masked_accuracies)
                worst_masked_acc = np.min(masked_accuracies)

                print(f"Best masked performance: {best_masked_acc:.4f}")
                print(f"Worst masked performance: {worst_masked_acc:.4f}")
                print(f"Performance range: {best_masked_acc - worst_masked_acc:.4f}")

                if not np.isnan(accuracy_matrix[0,0]):
                    degradation_pct = (accuracy_matrix[0,0] - best_masked_acc) / accuracy_matrix[0,0] * 100
                    print(f"Maximum degradation: {degradation_pct:.1f}%")

            return results_data

        except KeyboardInterrupt:
            print("\n" + "="*60)
            print("EXPERIMENT INTERRUPTED BY USER")
            print("="*60)
            print("Progress has been saved. You can resume by running the experiment again with resume=True")
            return None

        except Exception as e:
            print(f"\nEXPERIMENT FAILED: {e}")
            import traceback
            traceback.print_exc()
            print("\nProgress has been saved up to the last checkpoint.")
            print("You can try to resume by running the experiment again with resume=True")
            return None


class TestOnlyDataset(torch.utils.data.Dataset):
    """Dataset that only loads test session data (sessions 13-14)"""

    def __init__(self, data_dir, user_ids):
        self.data_dir = Path(data_dir)
        self.user_ids = user_ids

        self.file_paths = []
        self.labels = []

        # Only load files from test sessions (13-14)
        test_sessions = [13, 14]

        for user_idx, user_id in enumerate(user_ids):
            user_folder = f"S{user_id:03d}"
            user_path = self.data_dir / user_folder

            if not user_path.exists():
                continue

            for session_num in test_sessions:
                session_folder = f"{user_folder}R{session_num:02d}"
                session_path = user_path / session_folder

                if not session_path.exists():
                    continue

                # Load all stacked files from this session
                for npy_file in session_path.glob("*_stacked.npy"):
                    self.file_paths.append(npy_file)
                    self.labels.append(user_idx)

        print(f"TestOnlyDataset: {len(self.file_paths)} files from test sessions")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        if data.ndim == 2:
            data = data[np.newaxis, :, :]  # Add channel dimension

        return torch.tensor(data, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def main():
    """Main execution function"""

    # Experiment configuration
    config = {
        # Paths (adjust for your environment)
        'experiment_dir': '/app/data/experiments/masking_subset10_time_mask_restored_params',
        'raw_eeg_dir': '/app/data/1.0.0',

        # Pre-trained model paths - CRITICAL: Set these to your actual paths
        'model_checkpoint_path': '/app/data/jepa_logs_time_mask_ckpt/last.ckpt',
        'model_config_path': '/app/configs',  # Directory containing train.yaml (REQUIRED)

        # Local paths (comment out if using server)
        # 'experiment_dir': '/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/experiments/eeg_masking_experiment',
        # 'raw_eeg_dir': '/Users/belindahu/Desktop/thesis/biometrics-JEPA/mmi/dataset/physionet.org/files/eegmmidb/1.0.0',
        # 'model_checkpoint_path': '/Users/belindahu/Desktop/thesis/biometrics-JEPA/path/to/your/model/checkpoint.ckpt',
        # 'model_config_path': '/Users/belindahu/Desktop/thesis/biometrics-JEPA/configs',

        # Experiment parameters
        'user_ids': list(range(1, 11)),  # S001-S010
        'train_sessions': list(range(1, 11)),  # Sessions 1-10 for training
        'val_sessions': [11, 12],              # Sessions 11-12 for validation
        'test_sessions': [13, 14],             # Sessions 13-14 for testing

        # Training parameters - UPDATED to match scaling experiment
        'train_epochs': 100,
        'batch_size': 8,
        'learning_rate': 0.001,
        'eval_runs_per_variant': 1,
        'weight_decay': 0, #1e-4,

        # Advanced training parameters - NEW from scaling experiment
        'use_cosine_classifier': True,
        'cosine_scale': 0, #40.0,
        'label_smoothing': 0, #0.1,
        'warmup_epochs': 0, #5,
        'dropout_rate': 0, # 0.2,
        'classifier_dropout': 0.0,

        # Model parameters - UPDATED to match scaling experiment
        'model_type': 'full', #full,
        'input_channels': 1,
        'num_mel_bins': 80,
        'expected_time_frames': 2000,

        # EEG processing parameters
        'sample_rate': 16000,
        'frame_duration': 20.0,
        'frame_stride': 10.0,

        # Experiment metadata
        'description': 'EEG masking experiment with cosine classifier and advanced training techniques'
    }

    print("EEG Masking Experiment")
    print("=" * 60)
    print(f"Users: S001-S010 ({len(config['user_ids'])} users)")
    print(f"Training: Sessions {config['train_sessions']}")
    print(f"Validation: Sessions {config['val_sessions']}")
    print(f"Testing: Sessions {config['test_sessions']}")
    print(f"Experiment directory: {config['experiment_dir']}")
    print("=" * 60)

    # Initialize and run experiment
    experiment = EEGMaskingExperiment(config)

    results = experiment.run_complete_experiment()

    if results is not None:
        print("\nExperiment completed successfully!")
        print(f"All results saved to: {experiment.experiment_dir}")
        return True
    else:
        print("\nExperiment failed!")
        return False


if __name__ == "__main__":
    import torch.nn as nn
    import torch.utils.data

    success = main()
    sys.exit(0 if success else 1)