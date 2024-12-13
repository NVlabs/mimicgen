# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if (('.png' not in x) and ('.gif' not in x))]
long_description = ''.join(lines)

setup(
    name="mimicgen",
    packages=[
        package for package in find_packages() if package.startswith("mimicgen")
    ],
    install_requires=[
        "numpy>=1.13.3",
        "h5py",
        "tqdm",
        "imageio",
        "imageio-ffmpeg",
        "gdown",
        "chardet",
        "huggingface_hub",
    ],
    eager_resources=['*'],
    include_package_data=True,
    python_requires='>=3',
    description="MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations",
    author="Ajay Mandlekar",
    url="https://github.com/NVlabs/mimicgen",
    author_email="amandlekar@nvidia.com",
    version="1.0.0",
    long_description=long_description,
    long_description_content_type='text/markdown'
)
