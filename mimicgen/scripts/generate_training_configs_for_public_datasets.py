# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Script to generate json configs for use with robomimic to reproduce the
policy learning results in the MimicGen paper.
"""
import os
import argparse
import robomimic
from robomimic.config import config_factory, Config
from robomimic.scripts.generate_paper_configs import modify_config_for_default_low_dim_exp, modify_config_for_default_image_exp

import mimicgen
from mimicgen import DATASET_REGISTRY


def set_obs_config(config, obs_modality):
    """
    Sets specific config settings related to running low-dim or image experiments.

    Args:
        config (BCConfig instance): config to modify

        obs_modality (str): observation modality (either low-dim or image)
    """
    assert obs_modality in ["low_dim", "image"]

    # use identical settings to robomimic study, with exception of low-dim learning rate
    if obs_modality == "low_dim":
        config = modify_config_for_default_low_dim_exp(config)
        with config.algo.values_unlocked():
            # we found a higher learning rate of 1e-3 (instead of 1e-4) to work well
            config.algo.optim_params.policy.learning_rate.initial = 1e-3
    else:
        config = modify_config_for_default_image_exp(config)
        with config.algo.values_unlocked():
            # standard learning rate
            config.algo.optim_params.policy.learning_rate.initial = 1e-4
            # image runs use higher RNN hidden dim size
            config.algo.rnn.hidden_dim = 1000

    return config


def set_rnn_config(config):
    """
    Sets RNN settings in config.

    Args:
        config (BCConfig instance): config to modify

        obs_modality (str): observation modality (either low-dim or image)
    """
    with config.train.values_unlocked():
        # make sure RNN is enabled with sequence length 10
        config.train.seq_length = 10

    with config.algo.values_unlocked():
        # make sure RNN is enabled with sequence length 10
        config.algo.rnn.enabled = True
        config.algo.rnn.horizon = 10

        # base parameters that may get modified
        config.algo.actor_layer_dims = ()                                       # no MLP layers between rnn layer and output
        config.algo.gmm.enabled = True                                          # enable GMM
        config.algo.rnn.hidden_dim = 400                                        # rnn dim 400

    return config


def modify_config_for_dataset(config, dataset_type, task_name, obs_modality, base_dataset_dir):
    """
    Modifies a Config object with experiment, training, and observation settings to
    correspond to experiment settings for the dataset of type @dataset_type collected on @task_name. This 
    mostly just sets the rollout horizon.

    Args:
        config (Config instance): config to modify

        dataset_type (str): identifies the type of dataset (e.g. source human data, 
            core experiment data, object transfer data)

        task_name (str): identify task that dataset was collected on

        obs_modality (str): observation modality (either low-dim or image)

        base_dataset_dir (str): path to directory where datasets are on disk.
            Directory structure is expected to be consistent with the output
            of @make_dataset_dirs in the download_datasets.py script.
    """
    assert dataset_type in DATASET_REGISTRY, \
        "dataset type {} not found in dataset registry!".format(dataset_type)
    assert task_name in DATASET_REGISTRY[dataset_type], \
        "task {} not found in dataset registry under dataset type {}!".format(task_name, dataset_type)

    with config.experiment.values_unlocked():
        # look up rollout evaluation horizon in registry and set it
        config.experiment.rollout.horizon = DATASET_REGISTRY[dataset_type][task_name]["horizon"]

        # no validation
        config.experiment.validate = False

    with config.train.values_unlocked():
        # set dataset path
        file_name = "{}.hdf5".format(task_name)
        config.train.data = os.path.join(base_dataset_dir, dataset_type, file_name)
        config.train.hdf5_filter_key = None
        config.train.hdf5_validation_filter_key = None

    return config


def generate_experiment_config(
    base_exp_name,
    base_config_dir,
    base_dataset_dir,
    base_output_dir,
    dataset_type,
    task_name,
    obs_modality,
):
    """
    Helper function to generate a config for a particular experiment.

    Args:
        base_exp_name (str): name that identifies this set of experiments

        base_config_dir (str): base directory to place generated configs

        base_dataset_dir (str): path to directory where datasets are on disk.
            Directory structure is expected to be consistent with the output
            of @make_dataset_dirs in the download_datasets.py script.

        base_output_dir (str): directory to save training results to. If None, will use the directory
            from the default algorithm configs.

        dataset_type (str): identifies the type of dataset (e.g. source human data, 
            core experiment data, object transfer data)

        task_name (str): identify task that dataset was collected on

        obs_modality (str): observation modality (either low-dim or image)
    """

    # start with BC config
    config = config_factory(algo_name="bc")

    # set RNN settings
    config = set_rnn_config(config=config)

    # set low-dim / image settings
    config = set_obs_config(config=config, obs_modality=obs_modality)

    # add in config based on the dataset
    config = modify_config_for_dataset(
        config=config,
        dataset_type=dataset_type,
        task_name=task_name,
        obs_modality=obs_modality,
        base_dataset_dir=base_dataset_dir,
    )

    # set experiment name
    with config.experiment.values_unlocked():
        config.experiment.name = "{}_{}_{}".format(base_exp_name, task_name, obs_modality)
    # set output folder
    with config.train.values_unlocked():
        if base_output_dir is None:
            base_output_dir = config.train.output_dir
        config.train.output_dir = os.path.join(base_output_dir, base_exp_name, task_name, obs_modality, "trained_models")
    
    # save config to json file
    dir_to_save = os.path.join(base_config_dir, base_exp_name, task_name, obs_modality)
    os.makedirs(dir_to_save, exist_ok=True)
    json_path = os.path.join(dir_to_save, "bc_rnn.json")
    config.dump(filename=json_path)

    return config, json_path


def generate_all_configs(
    base_config_dir, 
    base_dataset_dir, 
    base_output_dir, 
):
    """
    Helper function to generate all configs.

    Args:
        base_config_dir (str): base directory to place generated configs

        base_dataset_dir (str): path to directory where datasets are on disk.
            Directory structure is expected to be consistent with the output
            of @make_dataset_dirs in the download_datasets.py script.

        base_output_dir (str): directory to save training results to. If None, will use the directory
            from the default algorithm configs.

        algo_to_config_modifier (dict): dictionary that maps algo name to a function that modifies configs 
            to add algo hyperparameter settings, given the task, dataset, and hdf5 types.
    """
    json_paths = Config() # use for convenient nested dict
    for dataset_type in DATASET_REGISTRY:
        for task in DATASET_REGISTRY[dataset_type]:
            for obs_modality in ["low_dim", "image"]:

                # generate config for this experiment
                config, json_path = generate_experiment_config(
                    base_exp_name=dataset_type,
                    base_config_dir=base_config_dir,
                    base_dataset_dir=base_dataset_dir,
                    base_output_dir=base_output_dir,
                    dataset_type=dataset_type,
                    task_name=task,
                    obs_modality=obs_modality,
                )

                # save json path into dict
                json_paths[dataset_type][task][obs_modality] = json_path

    return json_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Directory where generated configs will be placed
    parser.add_argument(
        "--config_dir",
        type=str,
        default=os.path.join(mimicgen.__path__[0], "exps/paper"),
        help="Directory where generated configs will be placed. Defaults to 'paper' subfolder in exps folder of repository",
    )

    # directory where released datasets are located
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=os.path.join(mimicgen.__path__[0], "../datasets"),
        help="Base dataset directory for released datasets. Defaults to datasets folder in repository.",
    )

    # output directory for training runs (will be written to configs)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(mimicgen.__path__[0], "../training_results"),
        help="Base output directory for all training runs that will be written to generated configs. Defaults to training_results folder in repository",
    )

    args = parser.parse_args()

    # read args
    generated_configs_base_dir = args.config_dir
    datasets_base_dir = args.dataset_dir
    output_base_dir = args.output_dir

    # generate configs for all experiments
    print("\nwriting configs to {}...".format(generated_configs_base_dir))
    print("\ndatasets expected at {}".format(datasets_base_dir))
    print("\ntraining outputs will be written to {}\n".format(output_base_dir))
    config_json_paths = generate_all_configs(
        base_config_dir=generated_configs_base_dir, 
        base_dataset_dir=datasets_base_dir, 
        base_output_dir=output_base_dir, 
    )

    # write output shell scripts
    for exp_name in config_json_paths:
        shell_path = os.path.join(generated_configs_base_dir, "{}.sh".format(exp_name))
        with open(shell_path, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("# " + "=" * 10 + exp_name + "=" * 10 + "\n")
            train_script_loc = os.path.join(robomimic.__path__[0], "scripts/train.py")

            for task in config_json_paths[exp_name]:
                for obs_modality in config_json_paths[exp_name][task]:
                    f.write("\n")
                    f.write("#  task: {}\n".format(task))
                    f.write("#    obs modality: {}\n".format(obs_modality))
                    exp_json_path = config_json_paths[exp_name][task][obs_modality]
                    cmd = "python {} --config {}\n".format(train_script_loc, exp_json_path)
                    f.write(cmd)
            f.write("\n")
