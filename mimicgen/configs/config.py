# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Base MG_Config object for mimicgen data generation.
"""
import six
from copy import deepcopy

import robomimic
from robomimic.config.config import Config


# global dictionary for remembering name - class mappings
REGISTERED_CONFIGS = {}


def get_all_registered_configs():
    """
    Give access to dictionary of all registered configs for external use.
    """
    return deepcopy(REGISTERED_CONFIGS)


def config_factory(name, config_type, dic=None):
    """
    Creates an instance of a config from the algo name. Optionally pass
    a dictionary to instantiate the config from the dictionary.
    """
    if (config_type not in REGISTERED_CONFIGS) or (name not in REGISTERED_CONFIGS[config_type]):
        raise Exception("Config for name {} and type {} not found. Make sure it is a registered config among: {}".format(
            name, config_type, ', '.join(REGISTERED_CONFIGS)))
    return REGISTERED_CONFIGS[config_type][name](dict_to_load=dic)


class ConfigMeta(type):
    """
    Define a metaclass for constructing a config class.
    It registers configs into the global registry.
    """
    def __new__(meta, name, bases, class_dict):
        cls = super(ConfigMeta, meta).__new__(meta, name, bases, class_dict)
        if cls.__name__ != "MG_Config":
            if cls.TYPE not in REGISTERED_CONFIGS:
                REGISTERED_CONFIGS[cls.TYPE] = dict()
            REGISTERED_CONFIGS[cls.TYPE][cls.NAME] = cls
        return cls


@six.add_metaclass(ConfigMeta)
class MG_Config(Config):
    def __init__(self, dict_to_load=None):
        if dict_to_load is not None:
            super(MG_Config, self).__init__(dict_to_load)
            return

        super(MG_Config, self).__init__()

        # store name class property in the config (must be implemented by subclasses)
        self.name = type(self).NAME
        self.type = type(self).TYPE

        self.experiment_config()
        self.obs_config()
        self.task_config()

        # After init, new keys cannot be added to the config, except under nested
        # attributes that have called @do_not_lock_keys
        self.lock_keys()

    @property
    @classmethod
    def NAME(cls):
        # must be specified by subclasses
        raise NotImplementedError

    @property
    @classmethod
    def TYPE(cls):
        # must be specified by subclasses
        raise NotImplementedError

    def experiment_config(self):
        """
        This function populates the `config.experiment` attribute of the config, 
        which has general settings related to the dataset generation (e.g.
        which environment, robot, and gripper to use for generation, how many 
        demonstrations to try collecting, etc).
        """

        # set the name of the experiment - which will be used to name the dataset folder that is generated
        self.experiment.name = "demo"

        # settings related to source dataset
        self.experiment.source.dataset_path = None  # path to source hdf5 dataset
        self.experiment.source.filter_key = None    # filter key, to select a subset of trajectories in the source hdf5 dataset
        self.experiment.source.n = None             # if provided, use only the first @n trajectories in source hdf5 dataset
        self.experiment.source.start = None         # if provided, exclude the first @start trajectories in source hdf5 dataset

        # settings related to data generation
        self.experiment.generation.path = None                                  # path where new dataset folder will be created
        self.experiment.generation.guarantee = False                            # whether to keep running data collection until we have @num_trials successful trajectories
        self.experiment.generation.keep_failed = True                           # whether to keep failed trajectories as well
        self.experiment.generation.num_trials = 10                              # number of attempts to collect new data

        # if True, select a different source demonstration for each subtask during data generation, else 
        # keep the same one for the entire episode
        self.experiment.generation.select_src_per_subtask = False
        
        # if True, each subtask segment will consist of the first robot pose and the target poses instead of just the target poses. 
        # Can sometimes help improve data generation quality as the interpolation segment will interpolate to where the robot 
        # started in the source segment instead of the first target pose. Note that the first subtask segment of each episode 
        # will always include the first robot pose, regardless of this argument.
        self.experiment.generation.transform_first_robot_pose = False

        # if True, each interpolation segment will start from the last target pose in the previous subtask segment, instead 
        # of the current robot pose. Can sometimes improve data generation quality.
        self.experiment.generation.interpolate_from_last_target_pose = True

        # settings related to task used for data generation
        self.experiment.task.name = None                                    # if provided, override the env name in env meta to collect data on a different environment from the one in source data
        self.experiment.task.robot = None                                   # if provided, override the robot name in env meta to collect data on a different robot from the one in source data
        self.experiment.task.gripper = None                                 # if provided, override the gripper in env meta to collect data on a different robot gripper from the one in source data
        self.experiment.task.env_meta_update_kwargs = Config()              # if provided, override the arguments passed to the environment constructor, possibly to collect data on a different environment from the one in source data
        self.experiment.task.env_meta_update_kwargs.do_not_lock_keys()
        self.experiment.task.interface = None                               # if provided, override the environment interface class to use for this task to use a different one from the one in source data
        self.experiment.task.interface_type = None                          # if provided, specify environment interface type (usually one per simulator) to use a different one from the one in source data

        # general settings
        self.experiment.max_num_failures = 50           # maximum number of failure demos to save
        self.experiment.render_video = True             # whether to render some generated demos to video
        self.experiment.num_demo_to_render = 50         # maxumum number of demos to render to video
        self.experiment.num_fail_demo_to_render = 50    # maxumum number of failure demos to render to video
        self.experiment.log_every_n_attempts = 50       # logs important info every N generation attempts

        # random seed for generation
        self.experiment.seed = 1

    def obs_config(self):
        """
        This function populates the `config.obs` attribute of the config, which has
        setings for which observations to collect during data generation.
        """
        self.obs.collect_obs = True     # whether to collect observations
        self.obs.camera_names = []      # which cameras to render observations from
        self.obs.camera_height = 84     # camera height
        self.obs.camera_width = 84      # camera width

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has settings for each object-centric subtask in a task.
        """
        raise NotImplementedError
