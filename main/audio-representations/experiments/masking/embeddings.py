"""
Embedding extraction for masking experiment using pre-trained JEPA encoder
Based on your original precompute.py but adapted for the masking experiment pipeline
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import hydra
from pathlib import Path
import torch.nn.functional as F


class MaskingEvalDataset(Dataset):
    """
    Dataset for embedding extraction in masking experiment
    Based on your original EvalDataset but simplified
    """
    def __init__(self, csv_file, data_dir, crop_frames=208, repeat_short=True):
        self.df = pd.read_csv(csv_file)
        self.data_dir = Path(data_dir)
        self.crop_frames = crop_frames
        self.repeat_short = repeat_short

        print(f"MaskingEvalDataset created with {len(self.df)} files")
        print(f"Data directory: {self.data_dir}")
        print(f"Crop frames: {self.crop_frames}")

    def __len__(self):
        return len(self.df)

    def complete_audio(self, lms):
        """Complete audio processing - from your original complete_audio logic"""
        l = lms.shape[-1]

        # Repeat if shorter than crop_frames
        if self.repeat_short and l < self.crop_frames:
            while l < self.crop_frames:
                lms = torch.cat([lms, lms], dim=-1)
                l = lms.shape[-1]

        # Crop if longer than crop_frames
        if l > self.crop_frames:
            lms = lms[..., :self.crop_frames]  # Take first crop_frames
        # Pad if shorter
        elif l < self.crop_frames:
            pad_param = [0, self.crop_frames - l] + [0, 0] * (lms.ndim - 1)
            lms = F.pad(lms, pad_param, mode='constant', value=0)

        return lms

    def __getitem__(self, idx):
        file_path = self.data_dir / self.df.iloc[idx, 0]

        try:
            data = np.load(file_path)
            if data.ndim == 2:
                data = data[np.newaxis, :, :]  # Add channel dim

            data = torch.tensor(data, dtype=torch.float32)
            data = self.complete_audio(data)

            return data, self.df.iloc[idx, 0]

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zero tensor as fallback
            fallback_data = torch.zeros((1, 80, self.crop_frames), dtype=torch.float32)
            return fallback_data, self.df.iloc[idx, 0]


class EmbeddingExtractor:
    """
    Extract embeddings using pre-trained JEPA encoder
    Adapted from your original precompute.py
    """

    def __init__(self, checkpoint_path, config_path=None, device=None):
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path) if config_path else None

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"EmbeddingExtractor initialized:")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  Device: {self.device}")

        self.model = None
        self.crop_frames = 208  # Default, will be updated from config

    def load_model_from_checkpoint(self):
        """Load the pre-trained model from checkpoint"""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        print("Loading model checkpoint...")

        try:
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            # Extract model configuration from checkpoint if available
            if 'hyper_parameters' in checkpoint:
                # Lightning checkpoint format
                hparams = checkpoint['hyper_parameters']
                if 'model' in hparams and 'encoder' in hparams['model']:
                    encoder_config = hparams['model']['encoder']
                    if 'img_size' in encoder_config:
                        self.crop_frames = encoder_config['img_size'][1]  # Time dimension

            elif 'model_config' in checkpoint:
                # Custom checkpoint format
                model_config = checkpoint['model_config']
                if 'encoder' in model_config and 'img_size' in model_config['encoder']:
                    self.crop_frames = model_config['encoder']['img_size'][1]

            print(f"Using crop_frames: {self.crop_frames}")

            # If we have a config file, use Hydra to instantiate the model
            if self.config_path and self.config_path.exists():
                self.model = self._load_model_with_hydra(checkpoint)
            else:
                # Fallback: try to instantiate model from checkpoint metadata
                self.model = self._load_model_from_checkpoint_metadata(checkpoint)

            self.model.eval()
            self.model.to(self.device)

            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _load_model_with_hydra(self, checkpoint):
        """Load model using Hydra configuration"""
        try:
            # This requires your original config structure
            with hydra.initialize(config_path=str(self.config_path.parent)):
                cfg = hydra.compose(config_name=self.config_path.stem)
                model = hydra.utils.instantiate(cfg.model)

                # Load state dict
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)

                return model

        except Exception as e:
            print(f"Failed to load with Hydra: {e}")
            raise

    def _load_model_from_checkpoint_metadata(self, checkpoint):
        """Fallback method to load model from checkpoint metadata"""
        # This is a simplified fallback - you may need to adjust based on your model structure
        print("Warning: Loading model without Hydra config. This may not work for all model types.")

        # Try to extract model architecture info from checkpoint
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Create a basic model structure (you'll need to adjust this for your specific model)
        # This is just a placeholder - the actual implementation depends on your model architecture

        class PlaceholderEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # This is a placeholder - replace with your actual model architecture
                self.placeholder = torch.nn.Identity()

            def forward(self, x):
                return x  # Placeholder implementation

        model = PlaceholderEncoder()

        try:
            model.load_state_dict(state_dict)
            print("Warning: Using placeholder model. Please provide proper config file.")
        except:
            print("Error: Cannot load model without proper configuration.")
            raise ValueError("Model loading failed. Please provide a proper config file.")

        return model

    def extract_embeddings(self, csv_file, data_dir, output_dir, batch_size=16, num_workers=4):
        """
        Extract embeddings for all files listed in CSV
        """
        if self.model is None:
            self.load_model_from_checkpoint()

        print(f"Extracting embeddings...")
        print(f"  Input CSV: {csv_file}")
        print(f"  Data directory: {data_dir}")
        print(f"  Output directory: {output_dir}")
        print(f"  Batch size: {batch_size}")

        # Create dataset and dataloader
        dataset = MaskingEvalDataset(
            csv_file=csv_file,
            data_dir=data_dir,
            crop_frames=self.crop_frames,
            repeat_short=True
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )

        print(f"Dataset loaded with {len(dataset)} samples")
        print(f"Processing {len(dataloader)} batches...")

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract embeddings
        total_processed = 0

        with torch.no_grad():
            for batch_idx, (batch_data, filenames) in enumerate(dataloader):
                batch_data = batch_data.to(self.device)

                # Get embeddings from encoder
                try:
                    batch_embeddings = self.model.encoder(batch_data)
                    if isinstance(batch_embeddings, tuple):
                        batch_embeddings = batch_embeddings[0]  # Take first output if tuple

                    batch_embeddings = batch_embeddings.cpu().numpy()

                except AttributeError:
                    # Fallback if model doesn't have .encoder attribute
                    print("Warning: Model doesn't have .encoder attribute, using full model")
                    batch_embeddings = self.model(batch_data)
                    if isinstance(batch_embeddings, tuple):
                        batch_embeddings = batch_embeddings[0]
                    batch_embeddings = batch_embeddings.cpu().numpy()

                # Save individual embeddings
                for emb, fname in zip(batch_embeddings, filenames):
                    # Create output path maintaining directory structure
                    rel_path = Path(fname)
                    save_path = output_dir / f"{rel_path.parent}" / f"{rel_path.stem}_emb.npy"
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    np.save(save_path, emb)
                    total_processed += 1

                # Progress logging
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(dataloader)} batches ({total_processed} files)")

                # Memory cleanup
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

        print(f"Embedding extraction completed!")
        print(f"  Total files processed: {total_processed}")
        print(f"  Output directory: {output_dir}")

        return str(output_dir)


def extract_embeddings_for_masking_experiment(
    csv_file,
    data_dir,
    output_dir,
    checkpoint_path,
    config_path=None,
    batch_size=16,
    device=None
):
    """
    Convenience function for embedding extraction in masking experiment
    """
    extractor = EmbeddingExtractor(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device
    )

    return extractor.extract_embeddings(
        csv_file=csv_file,
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=2  # Reduced for stability
    )


def test_embedding_extraction():
    """Test the embedding extraction functionality"""
    print("Testing embedding extraction...")

    # Test configuration - adjust paths as needed
    test_config = {
        'checkpoint_path': '/app/data/jepa_logs_subset10/xps/97d170e1/checkpoints/last.ckpt',
        'data_dir': '/app/data/test_spectrograms',
        'csv_file': '/app/data/test_files.csv',
        'output_dir': '/app/data/test_embeddings',
        'batch_size': 4
    }

    # Create test CSV if it doesn't exist
    test_data_dir = Path(test_config['data_dir'])
    if test_data_dir.exists():
        npy_files = list(test_data_dir.glob("**/*.npy"))[:10]  # Just test first 10 files

        if npy_files:
            df = pd.DataFrame({
                'file_name': [str(f.relative_to(test_data_dir)) for f in npy_files]
            })
            df.to_csv(test_config['csv_file'], index=False)

            try:
                result = extract_embeddings_for_masking_experiment(**test_config)
                print(f"Test successful! Embeddings saved to: {result}")
                return True
            except Exception as e:
                print(f"Test failed: {e}")
                return False
        else:
            print("No test data files found")
            return False
    else:
        print("Test data directory not found")
        return False


if __name__ == "__main__":
    success = test_embedding_extraction()
    if success:
        print("Embedding extraction test passed!")
    else:
        print("Embedding extraction test failed!")