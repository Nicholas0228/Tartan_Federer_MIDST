# this file is used to train shadow models using the data in train.csv (by default)

import json
import os
import pandas as pd
import shutil

from midst_models.single_table_TabDDPM.complex_pipeline import (
    clava_clustering,
    clava_training,
    clava_load_pretrained,
    clava_synthesizing,
    load_configs,
)
from midst_models.single_table_TabDDPM.pipeline_modules import load_multi_table

# use with id format to train the model, trans and account id is the row index (this will be dropped later in data preprocessing)
TRAIN_BASE_PATH = "./tabddpm_black_box/train/tabddpm_1" 

# prepare data from "trans_synthetic.csv" in "train_with_id.csv" format, save to "train.csv"
def data_prepare(path):
    TRAIN_DATA_PATH = os.path.join(TRAIN_BASE_PATH, "train.csv")
    SYN_DATA_PATH = os.path.join(path, "trans_synthetic.csv")
    df1 = pd.read_csv(TRAIN_DATA_PATH)
    df2 = pd.read_csv(SYN_DATA_PATH)

    df1.iloc[:, 2:] = df2

    TRAIN_DATA_PATH = os.path.join(path, "train.csv")
    df1.to_csv(TRAIN_DATA_PATH, index=False)

# some json file needed
TRANS_DEMO_PATH = "../midst_models/single_table_TabDDPM/configs/trans_demo.json"
DATASET_META_PATH = "../midst_models/single_table_TabDDPM/configs/dataset_meta.json"
TRANS_DOMAIN_PATH = "../midst_models/single_table_TabDDPM/configs/trans_domain.json"


with open(TRANS_DEMO_PATH, "r") as f:
    configs = json.load(f)

# for final phase
# nums = list(range(61, 71)) + list(range(101, 111))

# for dev phase
# nums = list(range(51, 61)) + list(range(91, 101))

# for train phase
nums = list(range(1,31))
for i in nums:
    
    # change model phase "train/dev/final" and corresponding numbers to train each model in their folder
    MODEL_PATH = "./tabddpm_black_box/train/tabddpm_" + str(i)
    print(MODEL_PATH)

    # prepare three json files for training
    config_path = os.path.join(MODEL_PATH, "trans_demo.json")
    configs["general"]["data_dir"] = MODEL_PATH
    configs["general"]["workspace_dir"] = os.path.join(MODEL_PATH, "workspace")
    configs["general"]["test_data_dir"] = MODEL_PATH
    with open(config_path, "w") as f:
        json.dump(configs, f, indent=4)
    
    shutil.copy(DATASET_META_PATH, os.path.join(MODEL_PATH, "dataset_meta.json"))
    shutil.copy(TRANS_DOMAIN_PATH, os.path.join(MODEL_PATH, "trans_domain.json"))
    
    # prepare train data
    data_prepare(MODEL_PATH)

    configs, save_dir = load_configs(config_path)

    # load table
    tables, relation_order, dataset_meta = load_multi_table(configs["general"]["data_dir"])
    print("")

    # Clustering on the multi-table dataset
    tables, all_group_lengths_prob_dicts = clava_clustering(
        tables, relation_order, save_dir, configs
    )

    # Launch training from scratch
    models = clava_training(tables, relation_order, save_dir, configs)
    print("training completed")
    print("----------------------------------------------------------")

    # uncomment these if you want to generate synthetic data using new model
    # cleaned_tables, synthesizing_time_spent, matching_time_spent = clava_synthesizing(
    #     tables,
    #     relation_order,
    #     save_dir,
    #     all_group_lengths_prob_dicts,
    #     models,
    #     configs,
    #     sample_scale=1 if "debug" not in configs else configs["debug"]["sample_scale"],
    # )

    # print("generation done!!!")
    # print("----------------------------------------------------------")