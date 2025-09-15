# src/utils/conditional_callbacks.py
# Helper to conditionally add downstream callbacks based on config

import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Any


def get_conditional_callbacks(cfg: DictConfig) -> List[Any]:
    """
    Get list of callbacks, conditionally including downstream evaluation callbacks
    based on the downstream_evaluation.enabled setting.
    """
    callbacks = []

    # Always include default callbacks from your existing setup
    if "callbacks" in cfg:
        # Get callbacks from your existing callbacks config
        for callback_name, callback_config in cfg.callbacks.items():
            if callback_config is not None and hasattr(callback_config, '_target_'):
                try:
                    callback = hydra.utils.instantiate(callback_config)
                    callbacks.append(callback)
                except Exception as e:
                    print(f"Warning: Failed to instantiate callback {callback_name}: {e}")

    # Conditionally add downstream evaluation callbacks
    if cfg.get("downstream_evaluation", {}).get("enabled", False):
        print("🔄 Downstream evaluation enabled - adding downstream callbacks")

        # Add downstream evaluation callback
        try:
            downstream_callback = hydra.utils.instantiate(cfg.downstream_evaluation_callback)
            callbacks.append(downstream_callback)
            print("✅ Added downstream evaluation callback")
        except Exception as e:
            print(f"❌ Failed to add downstream evaluation callback: {e}")

        # Add downstream model checkpoint callback
        try:
            downstream_checkpoint = hydra.utils.instantiate(cfg.model_checkpoint_downstream)
            callbacks.append(downstream_checkpoint)
            print("✅ Added downstream model checkpoint callback")
        except Exception as e:
            print(f"❌ Failed to add downstream checkpoint callback: {e}")

        # Add downstream early stopping callback
        if cfg.downstream_evaluation.get("early_stopping_patience", 0) > 0:
            try:
                downstream_early_stopping = hydra.utils.instantiate(cfg.early_stopping_downstream)
                callbacks.append(downstream_early_stopping)
                print("✅ Added downstream early stopping callback")
            except Exception as e:
                print(f"❌ Failed to add downstream early stopping callback: {e}")

    else:
        print("ℹ️ Downstream evaluation disabled - using standard callbacks only")

    print(f"📋 Total callbacks loaded: {len(callbacks)}")
    return callbacks


def update_trainer_with_conditional_callbacks(trainer_cfg: DictConfig, all_cfg: DictConfig) -> DictConfig:
    """
    Update trainer configuration with conditional callbacks.
    """
    # Get the callbacks
    callbacks = get_conditional_callbacks(all_cfg)

    # Create a new trainer config with the callbacks
    updated_trainer_cfg = OmegaConf.create(trainer_cfg)

    # If trainer doesn't have callbacks list, add it
    if "callbacks" not in updated_trainer_cfg:
        updated_trainer_cfg["callbacks"] = []

    # Replace the callbacks list with our conditional callbacks
    # (This assumes your trainer accepts a list of instantiated callbacks)
    updated_trainer_cfg["callback_instances"] = callbacks

    return updated_trainer_cfg