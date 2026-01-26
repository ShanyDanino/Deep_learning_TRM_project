# NonoReason: Recursive Reasoning for Nonograms

**NonoReason** explores the potential of compact AI models to solve Nonograms, a logic puzzle, through recursive reasoning.

> **Note:** This project is based on the **TRM (Tiny Recursive Model)** architecture and utilizes the original codebase presented in the paper:
> **A. Jolicoeur-Martineau**, *"Less is more: Recursive reasoning with tiny networks"*, arXiv preprint arXiv:2510.04871 (2025).

## 📌 Project Overview
The goal of this project is to develop a recursive deep learning model capable of solving Nonograms—logic puzzles defined by row and column constraints.  NonoReason utilizes a **recursive reasoning architecture**. It solves the grid step-by-step, constantly cross-referencing constraints to verify and deduce the next move.

## 📊 Dataset
We utilize the Nonogram dataset originally published in the paper:
> **Rubio, José María Buades, et al.**, *"Solving nonograms using neural networks"*, Entertainment Computing 50 (2024): 100652.

The original dataset includes puzzles of various dimensions:
* **5x5:** Contains all possible combinations of 5x5 grids (Total: 33,554,432 puzzles).
* **10x10:** 361,094 training grids, 15,274 test grids.
* **15x15:** 361,094 training grids, 15,274 test grids.

For this project, we utilized **sub-datasets of 5x5 and 10x10 puzzles** containing a smaller number of samples. Each sub-dataset was split into three sets: **Train, Test, and Validation**.

> **Note:** The project includes a Jupyter Notebook (`.ipynb`) that handles the execution flow, including **automatically downloading the dataset**. See the [Usage](#-usage) section below for instructions.

