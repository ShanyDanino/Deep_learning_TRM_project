"""Evaluation utilities and main evaluation loop."""

import os
from typing import Optional, List, Any
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import wandb
import numpy as np

from ..training.config import PretrainConfig, TrainState
from ..utils import load_model_class

# Import from original locations (these haven't moved yet)
from ..data.puzzle_dataset import PuzzleDatasetMetadata


def plot_nonogram_combined(board, clues_grid, title="Nonogram"):
    """
    Visualizes a Nonogram and returns the Figure object for logging.

    Parameters:
    - board: (N, M) numpy array (binary 0/1).
    - clues_grid: (N, M) numpy array (dtype=object).
      clues_grid[i, j] must contain (row_clues_for_row_i, col_clues_for_col_j).
    """
    unique_vals = np.unique(board)
    # Check if every value in the board is either 0 or 1
    if not np.all(np.isin(unique_vals, [0, 1])):
        print(f"Visualizer Error: Board contains non-binary values! Found: {unique_vals}. Expected only {{0, 1}}.")
        raise ValueError(
            f"Visualizer Error: Board contains non-binary values! Found: {unique_vals}. Expected only {{0, 1}}.")

    rows, cols = board.shape

    # Extract Clues
    row_clues = [clues_grid[r, 0][0] for r in range(rows)]
    col_clues = [clues_grid[0, c][1] for c in range(cols)]

    # Plotting Logic
    fig, ax = plt.subplots(figsize=(cols / 1.5 + 2, rows / 1.5 + 2))
    ax.imshow(1 - board, cmap='gray', vmin=0, vmax=1, interpolation='nearest')

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1)

    for r in range(5, rows, 5):
        ax.axhline(r - 0.5, color='black', linewidth=2.5)
    for c in range(5, cols, 5):
        ax.axvline(c - 0.5, color='black', linewidth=2.5)

    ax.tick_params(axis='both', which='both', left=False, bottom=False,
                   labelleft=False, labelbottom=False)

    # Render Clues (Filtering Zeros)
    for r in range(rows):
        cleaned_clues = [str(x) for x in row_clues[r] if x > 0]
        txt = " ".join(cleaned_clues)
        ax.text(-0.6, r, txt, ha='right', va='center', fontsize=16, fontweight='bold')

    for c in range(cols):
        cleaned_clues = [str(x) for x in col_clues[c] if x > 0]
        txt = "\n".join(cleaned_clues)
        ax.text(c, -0.6, txt, ha='center', va='bottom', fontsize=16, fontweight='bold')

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.title(title, y=1.25, fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    return fig
    
def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    """Create evaluator instances from config.
    
    Args:
        config: Training configuration
        eval_metadata: Evaluation dataset metadata
        
    Returns:
        List of evaluator instances
    """
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "tiny_recursive_models.evaluation.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    """Run evaluation on test set.
    
    This function:
    1. Runs model inference on all test batches
    2. Computes basic metrics (accuracy, loss)
    3. Calls task-specific evaluators for advanced metrics
    4. Saves predictions if requested
    
    Args:
        config: Training configuration
        train_state: Current training state
        eval_loader: DataLoader for test set
        eval_metadata: Metadata about test dataset
        evaluators: List of task-specific evaluators
        rank: Current process rank
        world_size: Total processes
        cpu_group: CPU process group for communication
        
    Returns:
        Dictionary of metrics (on rank 0 only)
    """
    reduced_metrics = None
    processed_batches = 0
    total_samples = 0
    
    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        return_keys.add("logits")
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    # Visualize 10 examples from the first batch
                    if rank == 0 and processed_batches == 1:
                        try:
                            num_plot = min(batch["inputs"].shape[0], 10)
                            wandb_images = []
                            
                            print(f"[Visualizer] Generating {num_plot} plots for WandB...")

                            for idx in range(num_plot):
                                size = eval_metadata.size

                                # Get prediction
                                pred_flat = torch.argmax(preds["logits"], dim=-1)[idx].cpu().numpy()
                                board = pred_flat.reshape(size, size)
                                
                                # Get label and check correctness
                                label_flat = batch["labels"][idx].cpu().numpy()
                                label_board = label_flat.reshape(size, size)
                                is_correct = np.all(board == label_board)
                                status_str = "Correct" if is_correct else "Incorrect"

                                # Get Clues for
                                raw_inputs = batch["inputs"][idx].cpu().numpy()
                                reshaped_inputs = raw_inputs.reshape(size, size, 2, eval_metadata.clues_max_num)
                                clues = reshaped_inputs - 1  # Restore (0 -> -1 padding)
                                
                                # Format Clues
                                clues_grid_obj = np.empty((size, size), dtype=object)
                                for r in range(size):
                                    for c in range(size):
                                        row_c = clues[r, c, 0, :].astype(int)
                                        col_c = clues[r, c, 1, :].astype(int)
                                        clues_grid_obj[r, c] = (row_c, col_c)
                                
                                # Plot
                                fig = plot_nonogram_combined(
                                    board, 
                                    clues_grid_obj, 
                                    title=f"Nonogram #{idx} is {status_str} in step {train_state.step}"
                                )
                                
                                wandb_images.append(wandb.Image(fig))
                                plt.close(fig)

                            # Log images
                            if wandb.run is not None:
                                wandb.log({"eval/predictions_gallery": wandb_images}, step=train_state.step)

                        except Exception as e:
                            print(f"[Visualizer Error] {e}")
                            import traceback
                            traceback.print_exc()                    
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics
