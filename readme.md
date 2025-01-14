# Temporal LM: Benchmarking LMs for TKG Link Prediction <br>

## Introduction

The repository contains code for benchmarking various Language Models (LMs) in the task of temporal
Knowledge Graph Completion (KGC) expanding the efforts of the TEMT paper. The
conducted experiments compare various NLP models along with normal and zero encoding base-
lines to assess the importance of the choice of an appropriate embedding model for the task of link
prediction.

## Associated Work and Datasets

Below the links for original dataset given by the authors of the paper, the dataset preprocessed by us for the user's convinence and the link to the original TEMT author's repository:

## References and Resources

- [Repository of TEMT authors](https://github.com/duyguislakoglu/TEMT)  
- [Original Dataset](https://drive.google.com/drive/folders/1lfxUw8sRuL5qDYlX42Z-AFsgbgIVQhyJ)  
- [Dataset after initial splitting](https://drive.google.com/drive/folders/1uP66nWurssE9Wn-tZdbl4OTECC6o2_ro?usp=sharing)

## Repository 

The Repository Structure is as Follows:

```
root/
├── results/                # Directory for storing results (e.g., logs, outputs)
├── TEMT/                   # Main module directory (add description if needed)
├── notebooks/              # Jupyter notebooks for exploratory analysis and prototyping
├── pipeline-visualization/ # Visuals and assets for pipeline representation
├── .gitignore              # Git configuration file for ignored files
├── data_processor.py       # Script for data preprocessing and manipulation
├── dataset.py              # Dataset-related utilities and classes
├── Encoders.py             # Encoders implementation for handling data encoding
├── link_prediction.py      # Script for link prediction task
├── models.py               # Contains model architectures and related functions
├── models_dict.py          # Dictionary or module for managing models
├── readme.md               # Repository documentation (this file)
├── requirements.txt        # Python dependencies
├── results_logging.py      # Utility functions for logging results
├── run_experiments.py      # Script for running experiments
└── time_encoder.py         # Time encoding utilities
```

## Explanation of Key Components

### Directories
- **`data/`**: This folder holds input data files used for experiments or training.
- **`results/`**: The results generated during experiments (e.g., logs, predictions, performance metrics).
- **`notebooks/`**: Contains interactive Jupyter notebooks for experimenting and visualizing results.
- **`pipeline-visualization/`**: Visuals and assets for pipeline representation.

### Configuration and Dependencies
- **`.gitignore`**: Specifies files and folders that should be ignored by Git.
- **`requirements.txt`**: A file listing Python libraries and their versions needed to run the code.

### Scripts
- **`data_processor.py`**: Handles data preprocessing and manipulation.
- **`dataset.py`**: Utilities for loading and managing the datasets.
- **`Encoders.py`**: Implements encoding methods for data.
- **`link_prediction.py`**: Focuses on the link prediction experiment.
- **`models.py`**: Contains deep learning model's definition and architecture.
- **`models_dict.py`**: Dictionary of avilable embedding models.
- **`run_experiments.py`**: Main script for executing all experiments.
- **`results_logging.py`**: Utilities to log and save experiment results.
- **`time_encoder.py`**: Methods for handling time-based data encoding.

### Documentation
- **`readme.md`**: Repository documentation (this file).

## How to Run

1. Download the dataset from the [Dataset after initial splitting](https://drive.google.com/drive/folders/1uP66nWurssE9Wn-tZdbl4OTECC6o2_ro?usp=sharing) link in the **References and Resources** section.

2. If you wish to modify the embedding models, check the following files:
   - **`Encoders.py`**
   - **`models_dict.py`**

3. To run all experiments, use the following command:

```bash
python run_experiments.py --do_train --do_test --data_dir ./data/YAGO11k --dataset_name yago --epochs 50 --sampling 0.01 --results_save_dir ./Experiments_Results --tensorboard True --batch 1024 --n_temporal_neg 1 --lr 0.001 --min_time 0 --max_time 100 --margin 1.0 --save_to_dir Trained_Models --use_descriptions
```

4. --Or-- to train a single model run this command:

```bash
python link_prediction.py --data_dir "./data/inductive/all-triples/YAGO11k" --do_train --epochs 5 --batch_size 1024 --do_test --lr 0.001 --save_model --save_to "ind_yago11k_tp_model.pth" --use_descriptions --min_time -453 --max_time 2844 --n_temporal_neg 1 --embedding_model "all_mpnet_base_v2" --sampling 0.05 --results_save_dir "./yago_results/all_mpnet_base_v2" --tensorboard_log_dir "./yago_mpnet_logs"
```

5. The results should be present in the specified directories (logs for tensorboard --and-- .txt file for the model training and testing metrics and their values)