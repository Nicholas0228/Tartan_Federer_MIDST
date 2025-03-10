# this file is used to train shadow models using the synthetic data

import json
import os
import pandas as pd
import shutil

from midst_models.multi_table_ClavaDDPM.complex_pipeline import (
    clava_clustering,
    clava_training,
    clava_load_pretrained,
    clava_load_synthesized_data,
    clava_load_synthesized_data_updated,
    clava_synthesizing,
    clava_eval,
    load_configs,
)
from midst_models.multi_table_ClavaDDPM.pipeline_modules import load_multi_table
from midst_models.multi_table_ClavaDDPM.report_utils import get_multi_metadata
import warnings
warnings.filterwarnings("ignore")

# use with id format to train the model, trans and account id is the row index (this will be dropped later in data preprocessing)
TRAIN_BASE_PATH = "./clavaddpm_black_box/train/clavaddpm_1" 
BASE_JSON_PATH = "../midst_models/multi_table_ClavaDDPM/configs/domain_files/" 

# prepare data from "trans_synthetic.csv" in "train_with_id.csv" format, save to "train.csv"
def data_prepare(path):
    # use synthetic data to train
    shutil.copy(os.path.join(path, "trans_synthetic.csv"), 
        os.path.join(path, "trans.csv"))
    shutil.copy(os.path.join(path, "account_synthetic.csv"), 
        os.path.join(path, "account.csv"))
    shutil.copy(os.path.join(MODEL_PATH, "card_synthetic.csv"), 
        os.path.join(MODEL_PATH, "card.csv"))
    shutil.copy(os.path.join(MODEL_PATH, "client_synthetic.csv"), 
        os.path.join(MODEL_PATH, "client.csv"))
    shutil.copy(os.path.join(MODEL_PATH, "disp_synthetic.csv"), 
        os.path.join(MODEL_PATH, "disp.csv"))
    shutil.copy(os.path.join(MODEL_PATH, "district_synthetic.csv"), 
        os.path.join(MODEL_PATH, "district.csv"))
    shutil.copy(os.path.join(MODEL_PATH, "loan_synthetic.csv"), 
        os.path.join(MODEL_PATH, "loan.csv"))
    shutil.copy(os.path.join(MODEL_PATH, "order_synthetic.csv"), 
        os.path.join(MODEL_PATH, "order.csv"))

    # perpare the json file
    shutil.copy(os.path.join(BASE_JSON_PATH, "account_domain.json"), 
        os.path.join(path, "account_domain.json"))
    shutil.copy(os.path.join(BASE_JSON_PATH, "trans_domain.json"), 
        os.path.join(path, "trans_domain.json"))
    shutil.copy(os.path.join(BASE_JSON_PATH, "order_domain.json"), 
        os.path.join(path, "order_domain.json"))
    shutil.copy(os.path.join(BASE_JSON_PATH, "loan_domain.json"), 
        os.path.join(path, "loan_domain.json"))
    shutil.copy(os.path.join(BASE_JSON_PATH, "district_domain.json"), 
        os.path.join(path, "district_domain.json"))
    shutil.copy(os.path.join(BASE_JSON_PATH, "disp_domain.json"), 
        os.path.join(path, "disp_domain.json"))
    shutil.copy(os.path.join(BASE_JSON_PATH, "client_domain.json"), 
        os.path.join(path, "client_domain.json"))
    shutil.copy(os.path.join(BASE_JSON_PATH, "card_domain.json"), 
        os.path.join(path, "card_domain.json"))

    shutil.copy(os.path.join(BASE_JSON_PATH, "dataset_meta.json"), 
        os.path.join(path, "dataset_meta.json"))
    

CONFIG_PATH = "../midst_models/multi_table_ClavaDDPM/configs/berka.json"


# for final phase
# nums = list(range(61, 71)) + list(range(101, 111))

# for dev phase
# nums = list(range(51, 61)) + list(range(91, 101))

# for train phase
# nums = list(range(1,31))

nums = list(range(1,31)) + list(range(51, 61)) + list(range(91, 101)) + list(range(61, 71)) + list(range(101, 111))

for i in nums:
    if i in list(range(1,31)):
        current_phase = 'train'
    elif i in list(range(51, 61)) + list(range(91, 101)):
        current_phase = 'dev'
    else:
        current_phase = 'final'
    
    MODEL_PATH = f"./clavaddpm_black_box/{current_phase}/clavaddpm_" + str(i)
    print(MODEL_PATH)

    configs, save_dir = load_configs(CONFIG_PATH)
    # prepare three json files for training
    configs["general"]["data_dir"] = MODEL_PATH
    configs["general"]["workspace_dir"] = os.path.join(MODEL_PATH, "workspace2")
    configs["general"]["test_data_dir"] = MODEL_PATH
    save_dir = os.path.join(MODEL_PATH, "workspace2")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
       
    # prepare train data
    data_prepare(MODEL_PATH)

    # load table
    tables, relation_order, dataset_meta = load_multi_table(configs["general"]["data_dir"])
    print("")

    # Clustering on the multi-table dataset
    # we only care this model, train others may not help but costs too much time
    relation_order = [['account', 'trans']] 
    tables, all_group_lengths_prob_dicts = clava_clustering(
        tables, relation_order, save_dir, configs
    )

    # Launch training from scratch
    models = clava_training(tables, relation_order, save_dir, configs)
    print("training completed:", MODEL_PATH)
    print("---------------------------------------------------------\n\n")
    
    # uncomment these if you want to generate synthetic data using new model
    # # synthetic using new model
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

from midst_models.multi_table_ClavaDDPM.pipeline_modules import *
from midst_models.multi_table_ClavaDDPM.pipeline_utils import *
from midst_models.multi_table_ClavaDDPM.complex_pipeline import *

def clava_training(tables, relation_order, save_dir, configs):
    models = {}
    for parent, child in relation_order:
        print(f"Training {parent} -> {child} model from scratch")
        df_with_cluster = tables[child]["df"]
        id_cols = [col for col in df_with_cluster.columns if "_id" in col]
        df_without_id = df_with_cluster.drop(columns=id_cols)
        result = child_training(
            df_without_id, tables[child]["domain"], parent, child, configs
        )
        models[(parent, child)] = result
        pickle.dump(
            result,
            open(os.path.join(save_dir, f"models/{parent}_{child}_ckpt.pkl"), "wb"),
        )

    return models

def child_training(
    child_df_with_cluster, child_domain_dict, parent_name, child_name, configs
):
    if parent_name is None:
        y_col = "placeholder"
        child_df_with_cluster["placeholder"] = list(range(len(child_df_with_cluster)))
    else:
        y_col = f"{parent_name}_{child_name}_cluster"
    child_info = get_table_info(child_df_with_cluster, child_domain_dict, y_col)
    child_model_params = get_model_params(
        {
            "d_layers": configs["diffusion"]["d_layers"],
            "dropout": configs["diffusion"]["dropout"],
        }
    )
    child_T_dict = get_T_dict()

    child_result = train_model(
        child_df_with_cluster,
        child_info,
        child_model_params,
        child_T_dict,
        configs["diffusion"]["iterations"],
        configs["diffusion"]["batch_size"],
        configs["diffusion"]["model_type"],
        configs["diffusion"]["gaussian_loss_type"],
        configs["diffusion"]["num_timesteps"],
        configs["diffusion"]["scheduler"],
        configs["diffusion"]["lr"],
        configs["diffusion"]["weight_decay"],
    )

    if parent_name is None:
        child_result["classifier"] = None
    elif configs["classifier"]["iterations"] > 0:
        child_classifier = train_classifier(
            child_df_with_cluster,
            child_info,
            child_model_params,
            child_T_dict,
            configs["classifier"]["iterations"],
            configs["classifier"]["batch_size"],
            configs["diffusion"]["gaussian_loss_type"],
            configs["diffusion"]["num_timesteps"],
            configs["diffusion"]["scheduler"],
            cluster_col=y_col,
            d_layers=configs["classifier"]["d_layers"],
            dim_t=configs["classifier"]["dim_t"],
            lr=configs["classifier"]["lr"],
        )
        child_result["classifier"] = child_classifier

    child_result["df_info"] = child_info
    child_result["model_params"] = child_model_params
    child_result["T_dict"] = child_T_dict
    return child_result

def train_classifier(
    df,
    df_info,
    model_params,
    T_dict,
    classifier_steps,
    batch_size,
    gaussian_loss_type,
    num_timesteps,
    scheduler,
    device="cuda",
    cluster_col="cluster",
    d_layers=None,
    dim_t=128,
    lr=0.0001,
):
    T = Transformations(**T_dict)
    dataset, label_encoders, column_orders = make_dataset_from_df(
        df,
        T,
        is_y_cond=model_params["is_y_cond"],
        ratios=[0.99, 0.005, 0.005],
        df_info=df_info,
        std=0,
    )
    print(dataset.n_features)
    train_loader = prepare_fast_dataloader(
        dataset, split="train", batch_size=batch_size, y_type="long"
    )
    val_loader = prepare_fast_dataloader(
        dataset, split="val", batch_size=batch_size, y_type="long"
    )
    test_loader = prepare_fast_dataloader(
        dataset, split="test", batch_size=batch_size, y_type="long"
    )

    eval_interval = 5
    log_interval = 10

    K = np.array(dataset.get_category_sizes("train"))
    if len(K) == 0 or T_dict["cat_encoding"] == "one-hot":
        K = np.array([0])
    print(K)

    num_numerical_features = (
        dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0
    )
    if model_params["is_y_cond"] == "concat":
        num_numerical_features -= 1

    # print(df[cluster_col])
    classifier = Classifier(
        d_in=num_numerical_features,
        d_out=int(max(df[cluster_col].values) + 1),
        dim_t=dim_t,
        hidden_sizes=d_layers,
    ).to(device)

    classifier_optimizer = optim.AdamW(classifier.parameters(), lr=lr)

    empty_diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=None,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device,
    )
    empty_diffusion.to(device)

    schedule_sampler = create_named_schedule_sampler("uniform", empty_diffusion)

    classifier.train()
    resume_step = 0
    for step in range(classifier_steps):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * batch_size,
        )
        numerical_forward_backward_log(
            classifier,
            classifier_optimizer,
            train_loader,
            dataset,
            schedule_sampler,
            empty_diffusion,
            prefix="train",
        )

        classifier_optimizer.step()
        if not step % eval_interval:
            with torch.no_grad():
                classifier.eval()
                numerical_forward_backward_log(
                    classifier,
                    classifier_optimizer,
                    val_loader,
                    dataset,
                    schedule_sampler,
                    empty_diffusion,
                    prefix="val",
                )
                classifier.train()

        if not step % log_interval:
            logger.dumpkvs()

    # # test classifier
    # classifier.eval()

    # correct = 0
    # for step in range(3000):
    #     test_x, test_y = next(test_loader)
    #     test_y = test_y.long().to(device)
    #     if model_params["is_y_cond"] == "concat":
    #         test_x = test_x[:, 1:].to(device)
    #     else:
    #         test_x = test_x.to(device)
    #     with torch.no_grad():
    #         pred = classifier(test_x, timesteps=torch.zeros(test_x.shape[0]).to(device))
    #         correct += (pred.argmax(dim=1) == test_y).sum().item()

    # acc = correct / (3000 * batch_size)
    # print(acc)

    return classifier