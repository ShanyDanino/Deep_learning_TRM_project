import os
import csv
import json
import numpy as np
import math
import glob
import zipfile
import shutil

from typing import Optional
from argdantic import ArgParser
from pydantic import BaseModel 
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from .common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    size                  : int = 5 # default size for a nonogram (5x5)
    dataset_path          : str = "../../NonoDataset"
    orig_dataset_path     : str = os.path.join(dataset_path, f"{size}x{size}")
    processed_dataset_path: str = "data/nonogram_dataset"
    clues_max_num         : int = math.ceil(float(size)/2) # 5->3, 10->5, 15->8
    num_aug               : int = 0
    subsample_size_train  : Optional[int] = None
    subsample_size_test   : Optional[int] = None
    min_difficulty        : Optional[int] = None
    
def rm_dir(directory_path):
  if os.path.exists(directory_path):
      try:
          shutil.rmtree(directory_path)
          print(f"Directory '{directory_path}' and all its contents removed.")
      except OSError as e:
          print(f"Error: {e.strerror}")
  else:
      print(f"Directory '{directory_path}' doesn't exist, no removal needed")
      
def download_files(set_name, config: DataProcessConfig):
  # If size dir already exists, remove it
  rm_dir(config.processed_dataset_path)

  # Unzip all zip files
  zip_files = glob.glob(os.path.join(config.orig_dataset_path, f"**{set_name}**/*.zip"), recursive=True)
  for z_file in zip_files:
    try:
      with zipfile.ZipFile(z_file, 'r') as zip_ref:
        # Extract to the specific folder where the zip is located
        extract_to_path = os.path.join(config.orig_dataset_path, f"{config.size}x{config.size}")
        zip_ref.extractall(extract_to_path)
        print(f"Extracted: {os.path.basename(z_file)}")
    except zipfile.BadZipFile:
      print(f"Skipping corrupted zip: {z_file}")

  all_npz = glob.glob(os.path.join(config.orig_dataset_path, f"**{set_name}**/*.npz"), recursive=True)
  os.makedirs(config.processed_dataset_path, exist_ok=False)
  os.makedirs(os.path.join(config.processed_dataset_path, "before_subsets"), exist_ok=False)

  # Loop through and load them
  for file_path in all_npz:
      filename = os.path.basename(file_path).lower()

      # Only load relevant files
      if 'database' in filename or 'backtracking' in filename:
        print(f"Skipping: {filename}")
        continue

      try:
        data  = np.load(file_path, allow_pickle=True)
        array = next(iter(data.values()))
        
        total_in_file       = array.shape[0]
        subsample_size = config.subsample_size_train if set_name == "train" else config.subsample_size_test
        num_samples_to_save = min(subsample_size, array.shape[0])
        indices          = np.random.choice(total_in_file, size=num_samples_to_save, replace=False)
        subsampled_array = array[indices]
        np.save(os.path.join(config.processed_dataset_path, "before_subsets", filename), subsampled_array)

      except Exception as e:
        print(f"Error loading {filename}: {e}")

  rm_dir(config.dataset_path)

  print("=" * 90)
  print(f"Finished downloading dataset with size {config.size}x{config.size}")
  
def load_files(set_name, config: DataProcessConfig):
  if config.size==5:
    clues = np.load(os.path.join(config.processed_dataset_path, "before_subsets", "train_combined.npz.npy"))
    labels = np.load(os.path.join(config.processed_dataset_path, "before_subsets", "target_combined.npz.npy"))
  elif config.size==10:
    clues = np.load(os.path.join(config.processed_dataset_path, "before_subsets", f"x_{set_name}_dataset.npz.npy"))
    labels = np.load(os.path.join(config.processed_dataset_path, "before_subsets", f"y_{set_name}_dataset.npz.npy"))
  elif config.size==15:
    clues = np.load(os.path.join(config.processed_dataset_path, "before_subsets", f"x_{set_name}_15x15_ok.npz.npy"))
    labels = np.load(os.path.join(config.processed_dataset_path, "before_subsets", f"y_{set_name}_15x15_ok.npz.npy"))
  
  nonograms_number = clues.shape[0]
  parsed_clues  = clues.reshape(nonograms_number, 2, config.size, config.clues_max_num)
  parsed_labels = labels.reshape(nonograms_number, config.size, config.size)
  
  return broadcast_nonogram_clues(parsed_clues, config), parsed_labels
  
def broadcast_nonogram_clues(parsed_clues, config: DataProcessConfig):
    """
    Transforms raw clues (N, 2, config.size, config.clues_max_num) 
    into pixel-wise grid (N, config.size, config.size, 2, config.clues_max_num).
    """
    # raw_clues index 0: row clues, index 1: col clues
    row_clues = parsed_clues[:, 0, :, :] # (N, config.size, config.clues_max_num)
    col_clues = parsed_clues[:, 1, :, :] # (N, config.size, config.clues_max_num)
    
    # Broadcast rows across the horizontal axis (j)
    # Result: (N, config.size, config.size, config.clues_max_num)
    row_grid = np.tile(np.expand_dims(row_clues, axis=2), (1, 1, config.size, 1))
    
    # Broadcast columns across the vertical axis (i)
    # Result: (N, config.size, config.size, config.clues_max_num)
    col_grid = np.tile(np.expand_dims(col_clues, axis=1), (1, config.size, 1, 1))
    
    # Stack into the final 5D tensor
    return np.stack([row_grid, col_grid], axis=3)

def augment_nonogram(row_clues: np.ndarray, col_clues: np.ndarray, solution: np.ndarray):
    # Create a random rotated version of the solution
    k = np.random.randint(4) # number of times the board will be rotated
    rotated_solution  = np.rot90(solution, k)
    rotated_row_clues = np.rot90(row_clues, k)
    rotated_col_clues = np.rot90(col_clues, k)

    # Transpose (Swap Rows <-> Cols)
    if np.random.rand() < 0.5:
        rotated_row_clues, rotated_col_clues = rotated_row_clues, rotated_col_clues
        rotated_solution = rotated_solution.T

    return rotated_row_clues, rotated_col_clues, rotated_solution

def convert_subset(set_name, config: DataProcessConfig):
    # 1. Load and process clues into your requested 5D shape
    inputs, labels = load_files(set_name, config)

    # 3. Generate dataset with indexing
    num_augments = config.num_aug if set_name == "train" else 0
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for i in tqdm(range(len(inputs)), desc=f"Processing {set_name}"):
        orig_inp = inputs[i]
        orig_out = labels[i]
        
        for aug_idx in range(1 + num_augments):
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                # Use the augmentation logic we discussed earlier
                inp, out = augment_nonogram(orig_inp, orig_out)

            # Flatten to 1D sequence for TRM architecture
            results["inputs"].append(inp.flatten())
            results["labels"].append(out.flatten())
            
            example_id += 1
            puzzle_id += 1
            
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
            
        # Push group boundary (tracks original + all its augments)
        results["group_indices"].append(puzzle_id)

    # 4. Final conversion and +1 vocab shift
    def _to_numpy(seq):
        return np.array(seq, dtype=np.int32) + 1

    results = {
        "inputs": _to_numpy(results["inputs"]),
        "labels": _to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # 5. Metadata and Save
    # Note: seq_len is now (config.size * config.size * 2 * config.clues_max_num)
    metadata = PuzzleDatasetMetadata(
        seq_len=config.size*config.size,
        vocab_size=config.size + 3, # All possible clues numbers + PAD + Full + Empty
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1 + num_augments,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"],
        size=config.size,
        clues_max_num=config.clues_max_num
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.processed_dataset_path, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
        
    # Save IDs mapping (for visualization only)
    with open(os.path.join(save_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)



@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    download_files("train", config)
    download_files("test", config)
    convert_subset("train", config)
    convert_subset("test", config)
    
    rm_dir(os.path.join(config.processed_dataset_path, "before_subsets"))


if __name__ == "__main__":
    cli()
