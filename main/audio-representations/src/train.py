import os
from typing import Any, Dict, List, Optional, Tuple

from dora import hydra_main
import hydra

import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    register_resolvers,
    task_wrapper,
)

# Import your JEPA classifier
from src.models.jepa_classifier import JEPAClassifier  # Adjust import path

log = RankedLogger(__name__, rank_zero_only=True)
register_resolvers()


def load_pretrained_jepa(checkpoint_path: str) -> torch.nn.Module:
    """
    Load pre-trained JEPA model from checkpoint.

    Args:
        checkpoint_path: Path to JEPA model checkpoint

    Returns:
        Pre-trained JEPA model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"JEPA checkpoint not found: {checkpoint_path}")

    log.info(f"Loading JEPA model from: {checkpoint_path}")

    # Option 1: If you saved a Lightning checkpoint
    try:
        # Load the full Lightning module
        jepa_lightning_model = hydra.utils.instantiate(cfg.model.encoder._target_).load_from_checkpoint(checkpoint_path)
        # Extract just the encoder/model part
        jepa_model = jepa_lightning_model.model  # or .encoder, depending on your structure
        log.info("Successfully loaded JEPA model from Lightning checkpoint")
        return jepa_model
    except Exception as e:
        log.warning(f"Could not load as Lightning checkpoint: {e}")

    # Option 2: If you saved just the model weights
    try:
        # Initialize the model architecture first
        jepa_model = hydra.utils.instantiate(cfg.model.encoder)
        # Load the weights
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        jepa_model.load_state_dict(state_dict, strict=False)
        log.info("Successfully loaded JEPA model weights")
        return jepa_model
    except Exception as e:
        log.error(f"Could not load JEPA model: {e}")
        raise e


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the JEPA classifier model."""

    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Load pre-trained JEPA model
    if cfg.model.get("pretrained_jepa_path"):
        jepa_model = load_pretrained_jepa(cfg.model.pretrained_jepa_path)
    else:
        raise ValueError("pretrained_jepa_path must be specified in model config")

    # Create classifier model with proper instantiation
    log.info("Creating JEPA classifier model")
    model = JEPAClassifier(
        jepa_model=jepa_model,
        **{k: v for k, v in cfg.model.items() if k != 'pretrained_jepa_path' and k != '_target_' and k != 'encoder'}
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Handle checkpoint resuming for classifier training
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path is None and os.path.exists(cfg.paths.ckpt_dir):
        print(cfg.paths.ckpt_dir)
        candidates = [os.path.join(cfg.paths.ckpt_dir, ckpt_file)
                      for ckpt_file in os.listdir(cfg.paths.ckpt_dir)
                      if ckpt_file.endswith(".ckpt")]
        if candidates:
            ckpt_path = max(candidates, key=os.path.getmtime)
            log.info(f"Resuming classifier training from checkpoint {ckpt_path}...")
        else:
            ckpt_path = None

    log.info(f"Number of CPUs allocated: {os.cpu_count()}")
    log.info(f"Number of GPUs allocated: {torch.cuda.device_count()}")
    log.info(f"Number of threads: {torch.get_num_threads()}, {torch.get_num_interop_threads()}")
    log.info(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")

    if cfg.get("train"):
        log.info("Starting classifier training!")
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # Merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra_main(version_base="1.3", config_path="../configs", config_name="train_classifier.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for classifier training."""

    # Handle A100 GPUs
    if torch.cuda.is_available() and "A100" in torch.cuda.get_device_name():
        torch.set_float32_matmul_precision("high")

    # Apply extra utilities
    extras(cfg)

    # Train the classifier
    metric_dict, _ = train(cfg)

    # Safely retrieve metric value for hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value


if __name__ == "__main__":
    main()