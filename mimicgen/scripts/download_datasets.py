# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Script to download datasets packaged with the repository.
"""
import os
import argparse

import mimicgen
import mimicgen.utils.file_utils as FileUtils
from mimicgen import DATASET_REGISTRY, HF_REPO_ID


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # directory to download datasets to
    parser.add_argument(
        "--download_dir",
        type=str,
        default=None,
        help="Base download directory. Created if it doesn't exist. Defaults to datasets folder in repository.",
    )

    # dataset type to download datasets for
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="core",
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset type to download datasets for (e.g. source, core, object, robot, large_interpolation). Defaults to core.",
    )

    # tasks to download datasets for
    parser.add_argument(
        "--tasks",
        type=str,
        nargs='+',
        default=["square_d0"],
        help="Tasks to download datasets for. Defaults to square_d0 task. Pass 'all' to download all tasks\
            for the provided dataset type or directly specify the list of tasks.",
    )

    # dry run - don't actually download datasets, but print which datasets would be downloaded
    parser.add_argument(
        "--dry_run",
        action='store_true',
        help="set this flag to do a dry run to only print which datasets would be downloaded"
    )

    args = parser.parse_args()

    # set default base directory for downloads
    default_base_dir = args.download_dir
    if default_base_dir is None:
        default_base_dir = os.path.join(mimicgen.__path__[0], "../datasets")

    # load args
    download_dataset_type = args.dataset_type
    download_tasks = args.tasks
    if "all" in download_tasks:
        assert len(download_tasks) == 1, "all should be only tasks argument but got: {}".format(args.tasks)
        download_tasks = list(DATASET_REGISTRY[download_dataset_type].keys())
    else:
        for task in download_tasks:
            assert task in DATASET_REGISTRY[download_dataset_type], "got unknown task {} for dataset type {}. Choose one of {}".format(task, download_dataset_type, list(DATASET_REGISTRY[download_dataset_type].keys()))

    # download requested datasets
    for task in download_tasks:
        download_dir = os.path.abspath(os.path.join(default_base_dir, download_dataset_type))
        download_path = os.path.join(download_dir, "{}.hdf5".format(task))
        print("\nDownloading dataset:\n    dataset type: {}\n    task: {}\n    download path: {}"
            .format(download_dataset_type, task, download_path))
        url = DATASET_REGISTRY[download_dataset_type][task]["url"]
        if args.dry_run:
            print("\ndry run: skip download")
        else:
            # Make sure path exists and create if it doesn't
            os.makedirs(download_dir, exist_ok=True)
            print("")
            FileUtils.download_file_from_hf(
                repo_id=HF_REPO_ID,
                filename=url,
                download_dir=download_dir,
                check_overwrite=True,
            )
        print("")
