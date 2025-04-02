# Tartan Federer MIA Implementation for MIDST Challenge (Membership Inference over Diffusion-models-based Synthetic Tabular Data) - SaTML 2025

This repository contains the code implementation from Tartan Federer's group for the MIDST Challenge. Most of the code is built upon the official challenge repository: [MIDSTModels](https://github.com/VectorInstitute/MIDSTModels).

**Group member:** 
[Xiaoyu (Nicholas) Wu](https://nicholas0228.github.io/), [Yifei Pang](https://2020pyfcrawl.github.io/), [Terrance Liu](https://terranceliu.github.io/), [Steven Wu](https://zstevenwu.com/).


See our White paper: **Winning the MIDST Challenge: New Membership Inference Attacks on Diffusion Models for Tabular Data Synthesis** https://arxiv.org/abs/2503.12008 for more details.

## Repository Structure

- **`midst_models`**: This directory closely follows the original implementation, with additional functions added to each model to support loss value extraction via hijection.
  
- **`MIA`**: This directory contains the main pipeline for executing membership inference attacks. Specifically, it includes:

  - `Single_Table_Black_Box_MIA.ipynb`: Black-box MIA pipeline for the **TabDDPM** scenario.
  - `Multi_Table_Black_Box_MIA.ipynb`: Black-box MIA pipeline for the **ClavaDDPM** scenario.
  - `Single_Table_White_Box_MIA.ipynb`: White-box MIA pipeline for the **TabDDPM** scenario.
  - `Multi_Table_White_Box_MIA.ipynb`: White-box MIA pipeline for the **ClavaDDPM** scenario.
  - `train_single_table_DM.py`: Shadow model training script for **TabDDPM**. (Note: it takes about 30 mins to train one model on A100 GPU)
  - `train_multi_table_DM.py`: Shadow model training script for **ClavaDDPM**. (Note: it takes about 45 mins to train one model on A100 GPU)



We do **not** implement attacks targeting **TabSyn**.


## Getting Started

The initial setup follows the official challenge repository. Most of the following section is a copy of the original challenge repository:

To get started with this repository, follow these steps:

1. Clone this repository to your machine.
2. Activate your python environment. You can create a new environment using the following command:

```bash
pip install --upgrade pip poetry
poetry env use [name of your python] #python3.9
source $(poetry env info --path)/bin/activate
poetry install --with "tabsyn, clavaddpm"
# If your system is not compatible with pykeops, you can uninstall it using the following command
pip uninstall pykeops
# Install the kernel for jupyter (only need to do it once)
ipython kernel install --user --name=midst_models
```

3. Run the notebook under `MIA`, starting with `download_updated_files.ipynb` to download the model checkpoints for the competition. Then, execute the corresponding attack notebook.

## License

This project is licensed under the terms of the [LICENSE] file located in the root directory of this repository.
