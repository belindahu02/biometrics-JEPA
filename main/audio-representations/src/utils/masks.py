from typing import Tuple

import torch


def unstructured_mask(m: int,
                        n: int,
                        masking_ratio: float,
                        num_masks: int = 1,
                        device: torch.device = torch.device("cpu")) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
    p = round((1 - masking_ratio) * m * n)

    flat_masks = torch.zeros((num_masks, m * n), dtype=torch.bool, device=device)

    # Generate random noise for all masks
    noise = torch.rand((num_masks, m * n), device=device)

    # Sort the noise tensor along the last dimension
    positions = noise.argsort(dim=-1)

    # Get the first p positions for each mask
    positions = positions[:, :p]

    # Set the corresponding elements to True in the flat_masks
    flat_masks.scatter_(1, positions, True)

    masks = flat_masks.view(num_masks, m, n)

    return masks, ~masks

def time_mask(m: int,
              n: int,
              masking_ratio: float,
              num_masks: int = 1,
              device: torch.device = torch.device("cpu")) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
    """
    Time-series masking - mask contiguous vertical strips (entire frequency spectrum for certain time steps).

    Args:
        m: Number of frequency patches
        n: Number of time patches
        masking_ratio: Ratio of patches to mask (applied to time dimension)
        num_masks: Batch size
        device: Device to create tensors on

    Returns:
        context_mask: Boolean mask for context patches (True = keep)
        target_mask: Boolean mask for target patches (True = mask/predict)
    """
    # Calculate number of time steps to keep as context
    num_context_time = round((1 - masking_ratio) * n)

    # Initialize masks - all False initially
    context_mask = torch.zeros((num_masks, m, n), dtype=torch.bool, device=device)

    # For each mask in the batch, randomly select which time steps to keep
    for i in range(num_masks):
        # Generate random noise for time steps
        noise = torch.rand(n, device=device)
        # Get indices of time steps to keep as context
        time_indices = noise.argsort()[:num_context_time]
        # Set all frequency bins for selected time steps to True (context)
        context_mask[i, :, time_indices] = True

    # Target mask is the inverse of context mask
    target_mask = ~context_mask

    return context_mask, target_mask


def multi_block_mask(m: int,
                     n: int,
                     masking_ratio: float,
                     num_masks: int = 1,
                     num_blocks: int = 4,
                     device: torch.device = torch.device("cpu")) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
    """
    Multi-block masking - sample multiple rectangular blocks as context and target.

    Args:
        m: Number of frequency patches
        n: Number of time patches
        masking_ratio: Ratio of patches to mask
        num_masks: Batch size
        num_blocks: Number of rectangular blocks to sample
        device: Device to create tensors on

    Returns:
        context_mask: Boolean mask for context patches (True = keep)
        target_mask: Boolean mask for target patches (True = mask/predict)
    """
    # Total number of patches to use as context
    total_context_patches = round((1 - masking_ratio) * m * n)
    # Patches per block (approximately)
    patches_per_block = total_context_patches // num_blocks

    context_mask = torch.zeros((num_masks, m, n), dtype=torch.bool, device=device)

    for i in range(num_masks):
        remaining_patches = total_context_patches

        for _ in range(num_blocks):
            if remaining_patches <= 0:
                break

            # Sample block size
            block_patches = min(patches_per_block, remaining_patches)

            # Randomly determine block dimensions
            # Try to make roughly square blocks
            block_area = block_patches
            block_h = torch.randint(1, min(m, block_area) + 1, (1,), device=device).item()
            block_w = min(n, max(1, block_area // block_h))

            # Randomly position the block
            start_h = torch.randint(0, max(1, m - block_h + 1), (1,), device=device).item()
            start_w = torch.randint(0, max(1, n - block_w + 1), (1,), device=device).item()

            # Set the block to True
            context_mask[i, start_h:start_h + block_h, start_w:start_w + block_w] = True

            # Update remaining patches
            actual_added = (context_mask[i].sum() - (
                        remaining_patches - block_patches + total_context_patches - remaining_patches))
            remaining_patches -= actual_added.item()

    target_mask = ~context_mask

    return context_mask, target_mask