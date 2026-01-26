# Recursive Nonogram Reasoner – "NonoReason"
​
**NonoReason** explores the potential of compact AI models to solve Nonograms, a logic puzzle, through recursive reasoning.

> **Note:** This project is based on the **TRM (Tiny Recursive Model)** architecture and utilizes the original codebase presented in the paper:
> **A. Jolicoeur-Martineau**, *"Less is more: Recursive reasoning with tiny networks"*, arXiv preprint arXiv:2510.04871 (2025).

## 📌 Project Overview
The goal of this project is to develop a recursive deep learning model capable of solving Nonograms—logic puzzles defined by row and column constraints.  NonoReason utilizes a **recursive reasoning architecture**. It solves the grid step-by-step, constantly cross-referencing constraints to verify and deduce the next move.
![Training_process](https://github.com/user-attachments/assets/ff26ad8c-eb5a-45d4-9338-0206d07076c1)

## 📊 Dataset
We utilize the Nonogram dataset originally published in the paper:
> **Rubio, José María Buades, et al.**, *"Solving nonograms using neural networks"*, Entertainment Computing 50 (2024): 100652.

The original dataset includes puzzles of various dimensions:
* **5x5:** Contains all possible combinations of 5x5 grids (Total: 33,554,432 puzzles).
* **10x10:** 361,094 training grids, 15,274 test grids.
* **15x15:** 361,094 training grids, 15,274 test grids.

For this project, we utilized **sub-datasets of 5x5 and 10x10 puzzles** containing a smaller number of samples. Each sub-dataset was split into three sets: **Train, Test, and Validation**.

> **Note:** The project includes a Jupyter Notebook (`.ipynb`) that handles the execution flow, including **automatically downloading the dataset**. See the [Usage](#-usage) section below for instructions.

## 📂 Files in the repository
As mentioned, this code is based on the original TRM implementation. The main adjustments and custom logic we implemented are primarily located in the following files:

| File name | Purpose |
| :--- | :--- |
| `Project_run.ipynb` | **Main Application:** Jupyter Notebook containing the full pipeline (Pulling from git directories, Dataset download, Training, Testing). |
| `download_dataset.py` | **Dataset Restructuring:** Handles loading mixed file formats (.zip, .npz, .npy) from the source dataset we mentioned above. |
| `speedrun.sh` | **Automation Script:** A bash script that handles the entire pipeline: building the dataset, training the model, and running evaluations. |
| `build_nonogram_dataset.py` | **Data Preprocessing:** Converts raw Nonogram puzzles into the specific tensor format required by the model. |
| `trm.py` | Contains the class definitions for the **TRM (Tiny Recursive Model)** architecture, modified for Nonograms. |
| `train.py` | **Training Script:** The main entry point for training. Handles **Hydra** configuration, distributed training, **WandB** logging, etc. |
| `evaluator.py` | **Evaluation & Visualization:** Handles the evaluation loop, calculates metrics, and generates **visual plots** of the Nonogram boards (with clues) to display in WandB. |
| `losses.py` | **Loss Functions:** Defines Binary Cross Entropy logic. Computes the reconstruction loss and accuracy metrics. |

## 💻 Usage

There are two ways to run the project: using the provided Jupyter Notebook (recommended for Google Colab) or running the automation script directly in a terminal.

### 🛠️ Prerequisites
Before running, ensure you have:
1.  **Weights & Biases Account:** The project logs training metrics to [WandB](https://wandb.ai/). You will need your API key.
2.  **GPU Support:** The training script automatically detects CUDA devices. At least one GPU is required.

---

### Option 1: Using the Notebook (Recommended)
The file `Project_run.ipynb` is designed to handle the entire setup pipeline, including fetching the dataset and setting up the environment.

1.  Open `Project_run.ipynb` in Google Colab or a Jupyter environment.
2.  **Set Secrets:** If using Colab, add the following to your "Secrets" (key icon):
    * `GITHUB_TOKEN`: Your GitHub personal access token (to clone the repo).
    * `WANDB_API_KEY`: Your Weights & Biases API key.
3.  **Run All Cells:** The notebook will:
    * Clone the repository.
    * Install dependencies via `uv`.
    * Download the dataset (`NonoDataset`).
    * Execute the `speedrun.sh` script to build, train, and evaluate the model.

---

### Option 2: Using the Automation Script (`speedrun.sh`)
For advanced users or local terminal execution, `speedrun.sh` provides a "one-click" solution with customizable parameters.

**Basic Command:**
```bash
chmod +x speedrun.sh
./speedrun.sh [TASK] [SIZE] [TRAIN_NUM] [TEST_NUM] [EPOCHS] [BATCH] [LR] [EVAL_INT] [RAW_DATA_PATH] [PROC_DATA_PATH]


