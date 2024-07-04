# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Helper script to report source dataset information. It verifies that the dataset has a 
"datagen_info" field for the first episode and prints its structure.
"""
import h5py
import argparse

import mimicgen
import mimicgen.utils.file_utils as MG_FileUtils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
        required=True,
    )
    args = parser.parse_args()

    dataset_path = args.dataset

    # get first demonstration
    first_demo_key = MG_FileUtils.get_all_demos_from_dataset(
        dataset_path=dataset_path,
        filter_key=None,
        start=None,
        n=1,
    )[0]
    f = h5py.File(dataset_path, "r")
    ep_grp = f["data/{}".format(first_demo_key)]

    # verify datagen info exists
    assert "datagen_info" in ep_grp, "Could not find MimicGen metadata in dataset {}. Ensure you have run prepare_src_dataset.py on this hdf5".format(dataset_path)
    
    # environment interface information
    env_interface_name = ep_grp["datagen_info"].attrs["env_interface_name"]
    env_interface_type = ep_grp["datagen_info"].attrs["env_interface_type"]

    print("\nEnvironment Interface: {}".format(env_interface_name))
    print("Environment Interface Type: {}".format(env_interface_type))

    # structure of datagen info
    ep_datagen_info = ep_grp["datagen_info"]

    print("\nStructure of datagen_info in episode {}:".format(first_demo_key))
    for k in ep_datagen_info:
        if k in ["object_poses", "subtask_term_signals"]:
            print("  {}:".format(k))
            for k2 in ep_datagen_info[k]:
                print("    {}: shape {}".format(k2, ep_datagen_info[k][k2].shape))
        else:
            print("  {}: shape {}".format(k, ep_datagen_info[k].shape))
    print("")

    f.close()
