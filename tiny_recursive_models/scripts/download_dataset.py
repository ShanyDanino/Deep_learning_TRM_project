import numpy as np
import zipfile
import os

# Configuration
SOURCE_DIR = "/content/NonoDataset"
OUTPUT_DIR = "uniform_dataset"
RANDOM_STATE = 42
TRAIN_SAMPLES = 100000
TEST_SAMPLES = 20000

def load_npy_data(path):
    """Robustly loads data from .npy, .npz, or .zip files."""
    print(f"Loading {path}...")

    if path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as z:
            file_list = z.namelist()

            # Case 1: Look for .npy inside zip
            npy_files = [n for n in file_list if n.endswith('.npy')]
            if npy_files:
                with z.open(npy_files[0]) as f:
                    return np.load(f)

            # Case 2: Look for .npz inside zip (Extract -> Load -> Delete)
            npz_files = [n for n in file_list if n.endswith('.npz')]
            if npz_files:
                print(f"  -> Found inner .npz: {npz_files[0]}")
                z.extract(npz_files[0], "/tmp")
                temp_path = os.path.join("/tmp", npz_files[0])
                try:
                    with np.load(temp_path) as data:
                        key = list(data.keys())[0]
                        return data[key]
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            # Case 3: Error - print contents for debugging
            raise ValueError(f"No .npy or .npz found in {path}. Contents: {file_list}")

    elif path.endswith('.npz'):
        with np.load(path) as data:
            # Return the first array found (usually 'arr_0' or similar)
            key = list(data.keys())[0]
            return data[key]

    elif path.endswith('.npy'):
        return np.load(path)

    else:
        raise ValueError(f"Unsupported format: {path}")

def save_split(size_name, x_data, y_data):
    """Splits data and saves to the uniform structure."""
    print(f"Processing {size_name}: X shape {x_data.shape}, Y shape {y_data.shape}")

    assert len(x_data) >= TRAIN_SAMPLES + 2 * TEST_SAMPLES, f"Mismatch in {size_name}: X({len(x_data)}) vs Y({len(y_data)})"

    # Shuffle everything together first
    rng = np.random.RandomState(RANDOM_STATE)
    indices = rng.permutation(len(x_data))
    x_data = x_data[indices]
    y_data = y_data[indices]

    # Manual Slice
    x_train = x_data[:TRAIN_SAMPLES]
    y_train = y_data[:TRAIN_SAMPLES]
    x_valid = x_data[TRAIN_SAMPLES : TRAIN_SAMPLES + TEST_SAMPLES]
    y_valid = y_data[TRAIN_SAMPLES : TRAIN_SAMPLES + TEST_SAMPLES]
    x_test = x_data[TRAIN_SAMPLES + TEST_SAMPLES : TRAIN_SAMPLES + 2 * TEST_SAMPLES]
    y_test = y_data[TRAIN_SAMPLES + TEST_SAMPLES : TRAIN_SAMPLES + 2 * TEST_SAMPLES]

    # Define paths
    base_out = os.path.join(OUTPUT_DIR, size_name)
    train_dir = os.path.join(base_out, "train")
    valid_dir = os.path.join(base_out, "valid")
    test_dir = os.path.join(base_out, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Save
    np.save(os.path.join(train_dir, "x_dataset.npy"), x_train)
    np.save(os.path.join(train_dir, "y_dataset.npy"), y_train)
    np.save(os.path.join(valid_dir, "x_dataset.npy"), x_valid)
    np.save(os.path.join(valid_dir, "y_dataset.npy"), y_valid)
    np.save(os.path.join(test_dir, "x_dataset.npy"), x_test)
    np.save(os.path.join(test_dir, "y_dataset.npy"), y_test)
    print(f"Saved {size_name} -> Train: {len(x_train)}, Validation: {len(x_valid)}, Test: {len(x_test)}")

def rearrange_dataset():
    # --- Process 5x5 ---
    # x is 'train_combined', y is 'target_combined'
    x_5 = load_npy_data(os.path.join(SOURCE_DIR, "5x5", "train_combined.zip"))
    y_5 = load_npy_data(os.path.join(SOURCE_DIR, "5x5", "target_combined.npz"))
    save_split("5x5", x_5, y_5)

    # --- Process 10x10 ---
    # Merge existing train/test files first, then resplit
    x_10_train = load_npy_data(os.path.join(SOURCE_DIR, "10x10", "x_train_dataset.npz"))
    x_10_test = load_npy_data(os.path.join(SOURCE_DIR, "10x10", "x_test_dataset.npz"))
    x_10_full = np.concatenate([x_10_train, x_10_test])

    y_10_train = load_npy_data(os.path.join(SOURCE_DIR, "10x10", "y_train_dataset.npz"))
    y_10_test = load_npy_data(os.path.join(SOURCE_DIR, "10x10", "y_test_dataset.npz"))
    y_10_full = np.concatenate([y_10_train, y_10_test])

    save_split("10x10", x_10_full, y_10_full)

    # --- Process 15x15 ---
    # Mixed formats: .zip for train, .npz for test
    x_15_train = load_npy_data(os.path.join(SOURCE_DIR, "15x15", "x_train_15x15_ok.zip"))
    x_15_test = load_npy_data(os.path.join(SOURCE_DIR, "15x15", "x_test_15x15_ok.npz"))
    x_15_full = np.concatenate([x_15_train, x_15_test])

    y_15_train = load_npy_data(os.path.join(SOURCE_DIR, "15x15", "y_train_15x15_ok.zip"))
    y_15_test = load_npy_data(os.path.join(SOURCE_DIR, "15x15", "y_test_15x15_ok.npz"))
    y_15_full = np.concatenate([y_15_train, y_15_test])

    save_split("15x15", x_15_full, y_15_full)

    print("\nProcessing complete. Check 'uniform_dataset/' folder.")

