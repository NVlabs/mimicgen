# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Helpful script to generate example config files, one per config class. These should be re-generated
when new config options are added, or when default settings in the config classes are modified.
"""
import os
import json

import mimicgen
from mimicgen.configs.config import get_all_registered_configs


def main():
    # store template config jsons in this directory
    target_dir = os.path.join(mimicgen.__path__[0], "exps/templates/")

    # iterate through registered config classes
    all_configs = get_all_registered_configs()
    for config_type in all_configs:
        # store config json by config type
        target_type_dir = os.path.join(target_dir, config_type)
        os.makedirs(target_type_dir, exist_ok=True)
        for name in all_configs[config_type]:
            # make config class to dump it to json
            c = all_configs[config_type][name]()
            assert name == c.name
            assert config_type == c.type
            # dump to json
            json_path = os.path.join(target_type_dir, "{}.json".format(name))
            c.dump(filename=json_path)


if __name__ == '__main__':
    main()