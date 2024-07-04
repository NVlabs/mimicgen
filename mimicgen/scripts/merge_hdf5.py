# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Script to merge all hdf5s if scripts/generate_dataset.py is incomplete, 
and doesn't make it to the line that merges all the hdf5s.
"""

import os
import shutil
import json
import h5py
import argparse
import imageio

import numpy as np

import mimicgen
import mimicgen.utils.file_utils as MG_FileUtils
from mimicgen.configs import config_factory
from mimicgen.scripts.generate_dataset import make_dataset_video, postprocess_motion_planning_dataset


def merge_hdf5s(args):
    """
    Main function to collect a new dataset using trajectory transforms from
    an existing dataset.
    """
    have_config = (args.config is not None)
    have_folder = (args.folder is not None)
    assert have_config or have_folder
    assert not (have_config and have_folder)

    folder_path = args.folder
    if have_config:
        # get folder path from config

        # load config object
        with open(args.config, "r") as f:
            ext_cfg = json.load(f)
            # config generator from robomimic generates this part of config unused by MimicGen
            if "meta" in ext_cfg:
                del ext_cfg["meta"]
        mg_config = config_factory(ext_cfg["name"], config_type=ext_cfg["type"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base config
        with mg_config.values_unlocked():
            mg_config.update(ext_cfg)

        base_folder = os.path.expandvars(os.path.expanduser(mg_config.experiment.generation.path)) # path where new folder will be generated
        new_dataset_folder_name = mg_config.experiment.name # name of folder to generate
        folder_path = os.path.join(
            base_folder,
            new_dataset_folder_name,
        )

    path_to_hdf5s = os.path.join(folder_path, "tmp")
    path_to_new_hdf5 = os.path.join(folder_path, "demo.hdf5")
    path_to_failed_hdf5s = os.path.join(folder_path, "tmp_failed")
    path_to_new_failed_hdf5 = os.path.join(folder_path, "demo_failed.hdf5")

    assert os.path.exists(path_to_hdf5s)
    merge_failures = os.path.exists(path_to_failed_hdf5s)

    # merge all new created files
    num_success = MG_FileUtils.merge_all_hdf5(
        folder=path_to_hdf5s,
        new_hdf5_path=path_to_new_hdf5,
        delete_folder=args.delete,
        dry_run=args.count,
    )
    print("NUM SUCCESS: {}".format(num_success))
    if merge_failures:
        num_failures = MG_FileUtils.merge_all_hdf5(
            folder=path_to_failed_hdf5s,
            new_hdf5_path=path_to_new_failed_hdf5,
            delete_folder=args.delete,
            dry_run=args.count,
        )
        print("NUM FAILURE: {}".format(num_failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to json for dataset generation, used to find dataset folder",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="path to dataset folder that is generated by scripts/generate_dataset.py",
    )
    parser.add_argument(
        "--count",
        action='store_true',
        help="if provided just count the number of demos instead of merging all of them",
    )
    parser.add_argument(
        "--delete",
        action='store_true',
        help="if provided, delete the tmp directories instead of saving them",
    )

    args = parser.parse_args()
    merge_hdf5s(args)