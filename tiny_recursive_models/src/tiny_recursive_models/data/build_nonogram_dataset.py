from typing import Optional
import os
import csv
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel 
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from trm.data.common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/nonogram_dataset"
    size : int = None
    dir_path : str = ""
    config.subsample_size: Optional[int] = None
    min_difficulty: Optional[int] = None
    num_aug: int = 0
    
def download_files(dir_path):
  rm_dir("sampled_dataset")
  rm_dir("tmp")

  # Unzip all zip files
  zip_files = glob.glob(os.path.join(dir_path, "**/*.zip"), recursive=True)
  for z_file in zip_files:
    try:
      with zipfile.ZipFile(z_file, 'r') as zip_ref:
        # Extract to the specific folder where the zip is located
        extract_path = os.path.join("./", os.path.dirname(z_file))
        os.makedirs(os.path.dirname(extract_path), exist_ok=True)
        zip_ref.extractall(extract_path)
        print(f"Extracted: {os.path.basename(z_file)}")
    except zipfile.BadZipFile:
      print(f"Skipping corrupted zip: {z_file}")

  all_npz = glob.glob(os.path.join("./", "**/*.npz"), recursive=True)
  collected_data = {}
  os.makedirs("sampled_dataset", exist_ok=True)

  # Loop through and load them
  for file_path in all_npz:
      filename = os.path.basename(file_path).lower()

      # Only load relevant files
      if 'database' in filename or 'backtracking' in filename:
        print(f"Skipping: {filename}")
        continue

      try:
        data = np.load(file_path, allow_pickle=True)
        array = next(iter(data.values()))
        print(f"Loaded {filename} from  {file_path}")
      except Exception as e:
        print(f"Error loading {filename}: {e}")

      new_dir_path = os.path.join("sampled_dataset", (os.path.basename(os.path.dirname(file_path)))) # sampled_dataset/sizexsize/
      os.makedirs(new_dir_path, exist_ok=True)
      np.save(os.path.join(new_dir_path, filename), array)

  rm_dir("NonoDataset")

  print("=" * 90)
  print("Finish")
  
def load_files(config.size, config.dir_path, set_name):
  if config.size == 5:
    size_dir = "5x5"
  elif config.size == 10:
    size_dir = "10x10"
  elif config.size == 15:
    size_dir = "15x15"
  
  max_clue_len: int = math.ceil(float(config.size)/2) # 5->3, 10->5, 15->8
  if config.size==5:
    clues = np.load(os.path.join(config.dir_path, size_dir, "train_combined.npz.npy"))
    labels = np.load(os.path.join(config.dir_path, size_dir, "target_combined.npz.npy"))

  elif config.size==10:
    clues = np.load(os.path.join(config.dir_path, size_dir, f"x_{set_name}_dataset.npz.npy"))
    labels = np.load(os.path.join(config.dir_path, size_dir, f"y_{set_name}_dataset.npz.npy"))
  elif config.size==15:
    clues = np.load(os.path.join(config.dir_path, size_dir, f"x_{set_name}_15x15_ok.npz.npy"))
    labels = np.load(os.path.join(config.dir_path, size_dir, f"y_{set_name}_15x15_ok.npz.npy"))
  nonograms_number = clues.shape[0]
  parsed_clues  = clues.reshape(nonograms_number,2, config.size, max_clue_len)
  parsed_labels = labels.reshape(nonograms_number,config.size, config.size)
  
  return broadcast_nonogram_clues(parsed_clues, config.size, max_clue_len), parsed_labels
  
def broadcast_nonogram_clues(parsed_clues, config.size, max_clue_len):
    """
    Transforms raw clues (N, 2, config.size, max_clue_len) 
    into pixel-wise grid (N, config.size, config.size, 2, max_clue_len).
    """
    # raw_clues index 0: row clues, index 1: col clues
    row_clues = parsed_clues[:, 0, :, :] # (N, config.size, max_clue_len)
    col_clues = parsed_clues[:, 1, :, :] # (N, config.size, max_clue_len)
    
    # Broadcast rows across the horizontal axis (j)
    # Result: (N, config.size, config.size, max_clue_len)
    row_grid = np.tile(np.expand_dims(row_clues, axis=2), (1, 1, config.size, 1))
    
    # Broadcast columns across the vertical axis (i)
    # Result: (N, config.size, config.size, max_clue_len)
    col_grid = np.tile(np.expand_dims(col_clues, axis=1), (1, config.size, 1, 1))
    
    # Stack into the final 5D tensor
    return np.stack([row_grid, col_grid], axis=3)

def augment_nonogram(row_clues: np.ndarray, col_clues: np.ndarray, solution: np.ndarray):
    # Create a random rrotated version of the solution
    k = np.random.randint(4) # number of times the board will be rotated
    rotated_solution  = np.rot90(solution, k)
    rotated_row_clues = np.rot90(row_clues, k)
    rotated_col_clues = np.rot90(col_clues, k)

    # Transpose (Swap Rows <-> Cols)
    if np.random.rand() < 0.5:
        rotated_row_clues, rotated_col_clues = rotated_row_clues, rotated_col_clues
        rotated_solution = rotated_solution.T

    return rotated_row_clues, rotated_col_clues, rotated_solution

def convert_subset(set_name,config: DataProcessConfig):
    # 1. Load and process clues into your requested 5D shape
    inputs, labels = load_files(config.size, config.dir_path, set_name)
    

    # 2. Match Sudoku Subsampling logic
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, config.size=config.subsample_size, replace=False)
            inputs = inputs[indices]
            labels = labels[indices]

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
    # Note: seq_len is now (config.size * config.size * 2 * max_clue_len)
    metadata = PuzzleDatasetMetadata(
        seq_len=config.size*config.size,
        #All possible clues numbers + PAD + Full+Empty
        vocab_size=config.size+3,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1 + num_augments,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, config.size, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
        
    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)



@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):

    GITHUB_DATASET_REPO_URL = "https://github.com/josebambu/NonoDataset.git"

    dataset_repo_name = GITHUB_DATASET_REPO_URL.split("/")[-1].replace(".git", "")

    if not os.path.exists(dataset_repo_name):
        print(f"Cloning {dataset_repo_name}...")
        !git clone $GITHUB_DATASET_REPO_URL
    else:
        print(f"Repository '{dataset_repo_name}' already exists. Skipping clone.")
    
    download_files(os.path.join(dataset_repo_name,f"{size}x{size}"))
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
