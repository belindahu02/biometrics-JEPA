import os
import sys
import subprocess
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

# Add parent directories to path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "classification"))

from embedding_processor import EmbeddingProcessor

class DownstreamEvaluationCallback(Callback):
    """
    Custom callback that runs downstream classification evaluation every N epochs
    and tracks the best performing JEPA model based on downstream accuracy.
    """

    def __init__(
            self,
            eval_every_n_epochs: int = 5,
            classification_scripts_dir: str = "./classification",
            eval_data_dir: str = "./data",
            temp_checkpoint_dir: str = "./temp_checkpoints",
            results_dir: str = "./downstream_results",
            # EmbeddingProcessor specific parameters
            csv_file: str = "./data/files_evaluation_subset10.csv",
            data_dir: str = "./data",
            batch_size: int = 16,
            num_workers: int = 4,
            crop_frames: int = None,
            # Simple classifier parameters
            num_classification_runs: int = 5,
            classification_epochs: int = 50,
            classification_lr: float = 0.001,
            classification_model_type: str = "lightweight",
            classification_normalization: str = "none",
            # Callback parameters
            metric_name: str = "downstream_accuracy",
            mode: str = "max",
            save_top_k: int = 3,
            verbose: bool = True,
            cleanup_temp_checkpoints: bool = False
    ):
        super().__init__()
        self.eval_every_n_epochs = eval_every_n_epochs
        self.classification_scripts_dir = Path(classification_scripts_dir)
        self.eval_data_dir = Path(eval_data_dir)
        self.temp_checkpoint_dir = Path(temp_checkpoint_dir)
        self.results_dir = Path(results_dir)
        self.metric_name = metric_name
        self.mode = mode
        self.save_top_k = save_top_k
        self.verbose = verbose
        self.cleanup_temp_checkpoints = cleanup_temp_checkpoints

        # EmbeddingProcessor parameters
        self.csv_file = csv_file
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_frames = crop_frames

        # Simple classifier parameters
        self.num_classification_runs = num_classification_runs
        self.classification_epochs = classification_epochs
        self.classification_lr = classification_lr
        self.classification_model_type = classification_model_type
        self.classification_normalization = classification_normalization

        # Create directories
        self.temp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Track best models
        self.best_models = []  # List of (score, epoch, checkpoint_path)
        self.current_epoch = 0

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch"""
        self.current_epoch = trainer.current_epoch

        # Check if we should evaluate
        if (self.current_epoch + 1) % self.eval_every_n_epochs != 0:
            return

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Running downstream evaluation at epoch {self.current_epoch + 1}")
            print(f"{'=' * 60}")

        try:
            # Save temporary checkpoint
            temp_ckpt_path = self._save_temp_checkpoint(trainer, pl_module)

            # Run downstream evaluation
            accuracy = self._run_downstream_evaluation(temp_ckpt_path, pl_module)

            if accuracy is not None:
                # Log the metric
                trainer.logger.log_metrics({self.metric_name: accuracy}, step=trainer.global_step)

                # Update best models tracking
                self._update_best_models(accuracy, self.current_epoch + 1, temp_ckpt_path)

                if self.verbose:
                    print(f"Downstream accuracy: {accuracy:.4f}")
                    if self.best_models:
                        best_acc = max(self.best_models)[0]
                        best_epoch = max(self.best_models)[1]
                        print(f"Current best: {best_acc:.4f} (epoch {best_epoch})")

            else:
                print("⚠️ Downstream evaluation failed")

        except Exception as e:
            print(f"❌ Error in downstream evaluation: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()

    def _save_temp_checkpoint(self, trainer, pl_module):
        """Save a temporary checkpoint for evaluation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_ckpt_path = self.temp_checkpoint_dir / f"temp_epoch_{self.current_epoch + 1}_{timestamp}.ckpt"

        # Save checkpoint
        trainer.save_checkpoint(temp_ckpt_path)

        return temp_ckpt_path

    def _run_downstream_evaluation(self, checkpoint_path, pl_module):
        """Run the downstream classification pipeline using EmbeddingProcessor and simple_classifier"""
        try:
            # Create unique results directory for this evaluation
            eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            eval_results_dir = self.results_dir / f"epoch_{self.current_epoch + 1}_{eval_timestamp}"
            eval_results_dir.mkdir(parents=True, exist_ok=True)

            # Set up paths for this evaluation
            embeddings_dir = eval_results_dir / "eval_embeddings"
            grouped_embeddings_dir = eval_results_dir / "grouped_embeddings"

            # Initialize EmbeddingProcessor with evaluation-specific paths
            processor = EmbeddingProcessor(
                csv_file=self.csv_file,
                data_dir=self.data_dir,
                embeddings_dir=str(embeddings_dir),
                grouped_embeddings_dir=str(grouped_embeddings_dir),
                checkpoint_path=str(checkpoint_path),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                crop_frames=self.crop_frames
            )

            if self.verbose:
                print("Step 1/2: Running embedding extraction and grouping...")

            # Run the complete embedding pipeline (precompute + group)
            # Pass the current model directly to avoid reloading from checkpoint
            processor.process_all(model=pl_module)

            if self.verbose:
                print("Step 2/2: Running classification training with simple_classifier...")

            # Run classification training using simple_classifier
            accuracy = self._run_simple_classification_training(eval_results_dir, grouped_embeddings_dir)

            # Save evaluation metadata
            self._save_evaluation_metadata(eval_results_dir, checkpoint_path, accuracy)

            return accuracy

        except Exception as e:
            print(f"❌ Error in downstream evaluation: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None

    def _run_simple_classification_training(self, eval_results_dir, grouped_embeddings_dir):
        """Run the simple classification training step"""
        try:
            # Prepare command for simple_classifier.py
            train_cmd = [
                "python",
                str(self.classification_scripts_dir / "simple_classifier.py"),
                f"--data_path={grouped_embeddings_dir}",
                f"--output_dir={eval_results_dir}",
                f"--model_path={eval_results_dir / 'temp_models'}",  # Store models in eval dir
                f"--num_runs={self.num_classification_runs}",
                f"--epochs={self.classification_epochs}",
                f"--batch_size={self.batch_size}",
                f"--lr={self.classification_lr}",
                f"--normalization_method={self.classification_normalization}",
                f"--model_type={self.classification_model_type}",
                f"--device={'cuda' if torch.cuda.is_available() else 'cpu'}",
            ]

            # Add verbose flag if needed
            if self.verbose:
                train_cmd.append("--verbose")

            if self.verbose:
                print(f"Running command: {' '.join(train_cmd)}")

            result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode != 0:
                print(f"❌ Classification training failed with return code {result.returncode}")
                print(f"Stderr: {result.stderr}")
                print(f"Stdout: {result.stdout}")
                return None

            # Extract accuracy from results
            accuracy = self._extract_accuracy_from_simple_results(eval_results_dir)
            return accuracy

        except subprocess.TimeoutExpired:
            print("❌ Classification training timed out")
            return None
        except Exception as e:
            print(f"❌ Error in classification training: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None

    def _extract_accuracy_from_simple_results(self, eval_results_dir):
        """Extract accuracy from the simple classifier results"""
        try:
            results_file = Path(eval_results_dir) / "classification_results.json"

            if not results_file.exists():
                print(f"Results file not found: {results_file}")
                return None

            with open(results_file, 'r') as f:
                results = json.load(f)

            # Extract mean accuracy from the simple classifier results format
            if 'results' in results and 'mean_accuracy' in results['results']:
                mean_accuracy = results['results']['mean_accuracy']

                if self.verbose:
                    experiment_info = results.get('experiment_info', {})
                    successful_runs = experiment_info.get('successful_runs', 0)
                    total_runs = experiment_info.get('total_runs_attempted', 0)
                    std_accuracy = results['results'].get('std_accuracy', 0)
                    mean_kappa = results['results'].get('mean_kappa', 0)
                    std_kappa = results['results'].get('std_kappa', 0)

                    print(f"\nClassification Results:")
                    print(f"  Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
                    print(f"  Kappa: {mean_kappa:.4f} ± {std_kappa:.4f}")
                    print(f"  Successful runs: {successful_runs}/{total_runs}")

                return mean_accuracy

            print("Could not find mean_accuracy in results")
            return None

        except Exception as e:
            print(f"Error extracting accuracy from results: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None

    def _save_evaluation_metadata(self, eval_results_dir, checkpoint_path, accuracy):
        """Save metadata about this evaluation"""
        metadata = {
            'epoch': self.current_epoch + 1,
            'checkpoint_path': str(checkpoint_path),
            'downstream_accuracy': accuracy,
            'timestamp': datetime.now().isoformat(),
            'eval_results_dir': str(eval_results_dir),
            'embedding_processor_config': {
                'csv_file': self.csv_file,
                'data_dir': self.data_dir,
                'batch_size': self.batch_size,
                'num_workers': self.num_workers,
                'crop_frames': self.crop_frames
            },
            'classification_config': {
                'num_runs': self.num_classification_runs,
                'epochs': self.classification_epochs,
                'learning_rate': self.classification_lr,
                'model_type': self.classification_model_type,
                'normalization': self.classification_normalization
            }
        }

        metadata_file = eval_results_dir / "evaluation_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _update_best_models(self, accuracy, epoch, checkpoint_path):
        """Update tracking of best performing models"""
        self.best_models.append((accuracy, epoch, str(checkpoint_path)))

        # Sort by accuracy (descending) and keep top k
        self.best_models.sort(reverse=(self.mode == "max"))

        if len(self.best_models) > self.save_top_k:
            # Remove worst model and delete its checkpoint if cleanup is enabled
            removed = self.best_models.pop()
            if self.cleanup_temp_checkpoints:
                try:
                    if os.path.exists(removed[2]):
                        os.remove(removed[2])
                        if self.verbose:
                            print(f"🗑️ Removed checkpoint: {Path(removed[2]).name}")
                except:
                    pass

        # Save best models list
        best_models_file = self.results_dir / "best_models.json"
        with open(best_models_file, 'w') as f:
            json.dump([
                {
                    'accuracy': score,
                    'epoch': epoch,
                    'checkpoint_path': path,
                    'rank': rank + 1
                }
                for rank, (score, epoch, path) in enumerate(self.best_models)
            ], f, indent=2)

    def on_train_end(self, trainer, pl_module):
        """Called when training ends"""
        if self.best_models:
            best_score, best_epoch, best_path = max(self.best_models)
            print(f"\n{'=' * 60}")
            print(f"🏆 Best downstream accuracy: {best_score:.4f} (epoch {best_epoch})")
            print(f"📍 Best checkpoint: {best_path}")
            print(f"{'=' * 60}")

            # Copy best checkpoint to final location
            final_best_path = self.results_dir / "best_downstream_model.ckpt"
            if os.path.exists(best_path):
                import shutil
                shutil.copy2(best_path, final_best_path)
                print(f"✅ Best model saved to: {final_best_path}")

        # Cleanup temporary files if requested
        if self.cleanup_temp_checkpoints:
            self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """Clean up temporary files created during evaluation"""
        try:
            # Clean up temporary checkpoints (keep only the best ones)
            temp_checkpoints = list(self.temp_checkpoint_dir.glob("temp_epoch_*.ckpt"))
            best_paths = {path for _, _, path in self.best_models}

            for temp_ckpt in temp_checkpoints:
                if str(temp_ckpt) not in best_paths:
                    try:
                        os.remove(temp_ckpt)
                        if self.verbose:
                            print(f"🧹 Cleaned up temporary checkpoint: {temp_ckpt.name}")
                    except:
                        pass

            # Clean up evaluation directories (keep only best ones)
            eval_dirs = list(self.results_dir.glob("epoch_*"))
            best_epochs = {epoch for _, epoch, _ in self.best_models}

            for eval_dir in eval_dirs:
                try:
                    # Extract epoch from directory name
                    epoch_str = eval_dir.name.split('_')[1]
                    epoch_num = int(epoch_str)

                    if epoch_num not in best_epochs:
                        import shutil
                        shutil.rmtree(eval_dir)
                        if self.verbose:
                            print(f"🧹 Cleaned up evaluation directory: {eval_dir.name}")
                except:
                    pass

            if self.verbose:
                print(f"🧹 Cleanup completed")

        except Exception as e:
            print(f"⚠️ Warning: Cleanup failed: {e}")

    def get_best_model_info(self):
        """Get information about the best model found so far"""
        if not self.best_models:
            return None

        best_score, best_epoch, best_path = max(self.best_models)
        return {
            'accuracy': best_score,
            'epoch': best_epoch,
            'checkpoint_path': best_path,
            'total_evaluations': len(self.best_models)
        }