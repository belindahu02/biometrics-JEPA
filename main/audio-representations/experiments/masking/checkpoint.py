"""
Checkpoint management for the masking experiment
Allows resuming from interruptions at any stage
"""

import json
import pickle
from pathlib import Path
from datetime import datetime
import shutil


class MaskingExperimentCheckpoint:
    """Manages checkpoints for the masking experiment"""

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.checkpoint_dir / "experiment_checkpoint.json"
        self.state_file = self.checkpoint_dir / "experiment_state.pkl"
        self.backup_dir = self.checkpoint_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        print(f"Checkpoint manager initialized at: {self.checkpoint_dir}")

    def save_checkpoint(self, stage, stage_data, accuracy_matrix=None, kappa_matrix=None,
                        detailed_results=None, metadata=None):
        """
        Save checkpoint for current experiment stage

        Args:
            stage: Current stage name ('train', 'generate_variants', 'evaluate', 'complete')
            stage_data: Stage-specific data (dict)
            accuracy_matrix: Current accuracy matrix (if available)
            kappa_matrix: Current kappa matrix (if available)
            detailed_results: List of detailed results (if available)
            metadata: Additional metadata
        """
        timestamp = datetime.now().isoformat()

        # Create backup of previous checkpoint
        if self.checkpoint_file.exists():
            backup_path = self.backup_dir / f"checkpoint_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy2(self.checkpoint_file, backup_path)

            # Keep only last 5 backups
            backups = sorted(self.backup_dir.glob("checkpoint_backup_*.json"))
            if len(backups) > 5:
                for old_backup in backups[:-5]:
                    old_backup.unlink()

        # Main checkpoint data (JSON-serializable)
        checkpoint_data = {
            'timestamp': timestamp,
            'stage': stage,
            'stage_data': stage_data,
            'metadata': metadata or {}
        }

        # Save JSON checkpoint
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        # Save full state (including numpy arrays) as pickle
        state_data = {
            'checkpoint_data': checkpoint_data,
            'accuracy_matrix': accuracy_matrix,
            'kappa_matrix': kappa_matrix,
            'detailed_results': detailed_results
        }

        with open(self.state_file, 'wb') as f:
            pickle.dump(state_data, f)

        print(f"Checkpoint saved: stage={stage}, timestamp={timestamp}")

    def load_checkpoint(self):
        """Load the most recent checkpoint"""
        if not self.checkpoint_file.exists():
            print("No checkpoint found")
            return None

        try:
            # Load JSON checkpoint
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)

            # Load full state
            if self.state_file.exists():
                with open(self.state_file, 'rb') as f:
                    state_data = pickle.load(f)

                print(f"Checkpoint loaded: stage={checkpoint_data['stage']}, timestamp={checkpoint_data['timestamp']}")
                return state_data
            else:
                print("State file not found, returning checkpoint data only")
                return {'checkpoint_data': checkpoint_data}

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None

    def get_completed_variants(self):
        """Get list of completed masking variants"""
        state = self.load_checkpoint()
        if state is None:
            return set()

        stage_data = state['checkpoint_data'].get('stage_data', {})
        return set(stage_data.get('completed_variants', []))

    def has_checkpoint(self):
        """Check if a checkpoint exists"""
        return self.checkpoint_file.exists()

    def clear_checkpoint(self):
        """Clear all checkpoint data"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.state_file.exists():
            self.state_file.unlink()
        print("Checkpoint cleared")


def add_checkpointing_to_experiment(experiment_class):
    """Decorator to add checkpointing to the experiment class"""

    original_init = experiment_class.__init__

    def new_init(self, config):
        original_init(self, config)
        # Add checkpoint manager
        checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_mgr = MaskingExperimentCheckpoint(checkpoint_dir)

    experiment_class.__init__ = new_init

    return experiment_class