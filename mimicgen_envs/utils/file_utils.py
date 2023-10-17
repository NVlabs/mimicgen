# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
A collection of utility functions for working with files.
"""
import os
import tempfile
import gdown

from robomimic.utils.file_utils import url_is_alive


def download_url_from_gdrive(url, download_dir, check_overwrite=True):
    """
    Downloads a file at a URL from Google Drive.

    Example usage:
        url = https://drive.google.com/file/d/1DABdqnBri6-l9UitjQV53uOq_84Dx7Xt/view?usp=drive_link
        download_dir = "/tmp"
        download_url_from_gdrive(url, download_dir, check_overwrite=True)

    Args:
        url (str): url string
        download_dir (str): path to directory where file should be downloaded
        check_overwrite (bool): if True, will sanity check the download fpath to make sure a file of that name
            doesn't already exist there
    """
    assert url_is_alive(url), "@download_url_from_gdrive got unreachable url: {}".format(url)

    with tempfile.TemporaryDirectory() as td:
        # HACK: Change directory to temp dir, download file there, and then move the file to desired directory.
        #       We do this because we do not know the name of the file beforehand.
        cur_dir = os.getcwd()
        os.chdir(td)
        fpath = gdown.download(url, quiet=False, fuzzy=True)
        fname = os.path.basename(fpath)
        file_to_write = os.path.join(download_dir, fname)
        if check_overwrite and os.path.exists(file_to_write):
            user_response = input(f"Warning: file {file_to_write} already exists. Overwrite? y/n\n")
            assert user_response.lower() in {"yes", "y"}, f"Did not receive confirmation. Aborting download."
        os.rename(fpath, file_to_write)
        os.chdir(cur_dir)
