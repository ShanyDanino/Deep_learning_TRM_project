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
from pydantic import BaseModel, model_validator
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from .common import PuzzleDatasetMetadata

IGNORE_LABEL_ID = -100

cli = ArgParser()


class DataProcessConfig(BaseModel):
    size: int = 5  # default size for a nonogram (5x5)
    dataset_path: str = "../../NonoDataset"
    orig_dataset_path: str = os.path.join(dataset_path, f"{size}x{size}")
    processed_dataset_path: str = "data/nonogram_dataset"
    clues_max_num: int = math.ceil(float(size) / 2)  # 5->3, 10->5, 15->8
    num_aug: int = 0
    subsample_size_train: Optional[int] = None
    subsample_size_test: Optional[int] = None
    min_difficulty: Optional[int] = None

    @model_validator(mode='after')
    def update_dependent_fields(self):
        # This runs AFTER the size is set by the command line
        self.orig_dataset_path = os.path.join(self.dataset_path, f"{self.size}x{self.size}")
        self.clues_max_num = math.ceil(float(self.size) / 2)
        return self


def rm_dir(directory_path):
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            print(f"Directory '{directory_path}' and all its contents removed.")
        except OSError as e:
            print(f"Error: {e.strerror}")
    else:
        print(f"Directory '{directory_path}' doesn't exist, no removal needed")


def load_files(set_name, config: DataProcessConfig):
    dir_path = os.path.join(config.processed_dataset_path, f"{config.size}x{config.size}", f"{set_name}")
    clues = np.load(os.path.join(dir_path, f"x_dataset.npy"))
    labels = np.load(os.path.join(dir_path, f"y_dataset.npy"))

    subsample_size = config.subsample_size_train if set_name == "train" else config.subsample_size_test
    subsampled_clues = clues[:subsample_size]
    subsampled_labels = labels[:subsample_size]

    parsed_clues = subsampled_clues.reshape(subsample_size, 2, config.size, config.clues_max_num)
    parsed_labels = subsampled_labels.reshape(subsample_size, config.size, config.size)

    return broadcast_nonogram_clues(parsed_clues, config), parsed_labels


def broadcast_nonogram_clues(parsed_clues, config: DataProcessConfig):
    """
    Transforms raw clues (N, 2, config.size, config.clues_max_num)
    into pixel-wise grid (N, config.size, config.size, 2, config.clues_max_num).
    """
    # raw_clues index 0: row clues, index 1: col clues
    row_clues = parsed_clues[:, 0, :, :]  # (N, config.size, config.clues_max_num)
    col_clues = parsed_clues[:, 1, :, :]  # (N, config.size, config.clues_max_num)

    # Broadcast rows across the horizontal axis (j)
    # Result: (N, config.size, config.size, config.clues_max_num)
    row_grid = np.tile(np.expand_dims(row_clues, axis=2), (1, 1, config.size, 1))

    # Broadcast columns across the vertical axis (i)
    # Result: (N, config.size, config.size, config.clues_max_num)
    col_grid = np.tile(np.expand_dims(col_clues, axis=1), (1, config.size, 1, 1))

    # Stack into the final 5D tensor
    return np.stack([row_grid, col_grid], axis=3)


def convert_subset(set_name, config: DataProcessConfig):
    # Load and process clues into your requested 5D shape
    inputs, labels = load_files(set_name, config)

    # Generate dataset with indexing
    num_augments = config.num_aug if set_name == "train" else 0
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for i in tqdm(range(len(inputs)), desc=f"Processing {set_name}"):
        # Flatten to 1D sequence for TRM architecture
        results["inputs"].append(inputs[i].flatten())
        results["labels"].append(labels[i].flatten())

        example_id += 1
        puzzle_id += 1

        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)

        # Push group boundary (tracks original + all its augments)
        results["group_indices"].append(puzzle_id)

    # Final conversion and +1 vocab shift (to allow for empty spaces)
    def _inputs_to_numpy(seq):
        arr = np.array(seq, dtype=np.int32)

        if not np.all((arr >= 0) & (arr <= config.size)):
            min_val, max_val = arr.min(), arr.max()
            raise ValueError(f"Input checking failed! Expected 0-{config.size}, but found range [{min_val}, {max_val}].")
        return arr + 1

    # Final conversion with no vocab shift (final results should be either 0 or 1)
    def _labels_to_numpy(seq):
        arr = np.array(seq, dtype=np.int32)

        if not np.all((arr >= 0) & (arr <= 1)):
            min_val, max_val = arr.min(), arr.max()
            raise ValueError(f"Label checking failed! Expected 0-1, but found range [{min_val}, {max_val}].")
        return arr

    results = {
        "inputs": _inputs_to_numpy(results["inputs"]),
        "labels": _labels_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=config.size * config.size,
        vocab_size=config.size + 3,  # All possible clues numbers + PAD + Full + Empty
        pad_id=0,
        ignore_label_id=IGNORE_LABEL_ID,
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
    # If size dir already exists, remove it
    rm_dir(config.processed_dataset_path)

    print(("Creating dir: ", config.processed_dataset_path), flush=True)
    os.makedirs(config.processed_dataset_path, exist_ok=False)
    os.makedirs(os.path.join(config.processed_dataset_path, "before_subsets"), exist_ok=False)

    convert_subset("train", config)
    convert_subset("test", config)

    rm_dir(os.path.join(config.processed_dataset_path, "before_subsets"))
    rm_dir(config.dataset_path)


if __name__ == "__main__":
    cli()