# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from mimicgen.configs.config import MG_Config, get_all_registered_configs, config_factory
from mimicgen.configs.task_spec import MG_TaskSpec
from mimicgen.configs.robosuite import *

from mimicgen.configs.robocasa.single_stage.config_doors import *
from mimicgen.configs.robocasa.single_stage.config_drawer import *
from mimicgen.configs.robocasa.single_stage.config_pnp import *
from mimicgen.configs.robocasa.single_stage.config_sink import *
from mimicgen.configs.robocasa.single_stage.config_stove import *
from mimicgen.configs.robocasa.single_stage.config_coffee import *
from mimicgen.configs.robocasa.single_stage.config_microwave import *