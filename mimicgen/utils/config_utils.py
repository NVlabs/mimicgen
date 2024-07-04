# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
A collection of utilities for working with config generators. These generators 
are re-used from robomimic (https://robomimic.github.io/docs/tutorials/hyperparam_scan.html)
"""
import json
from collections.abc import Iterable


def set_debug_settings(
    generator,
    group,
):
    """
    Sets config generator parameters for a quick debug run.

    Args:
        generator (robomimic ConfigGenerator instance): config generator object
        group (int): parameter group for these settings
    """
    generator.add_param(
        key="experiment.generation.guarantee",
        name="", 
        group=group, 
        values=[False],
    )
    generator.add_param(
        key="experiment.generation.num_trials",
        name="", 
        group=group, 
        values=[2],
    )
    return generator


def set_basic_settings(
    generator,
    group,
    source_dataset_path,
    source_dataset_name,
    generation_path,
    guarantee,
    num_traj,
    num_src_demos=None,
    max_num_failures=25,
    num_demo_to_render=10,
    num_fail_demo_to_render=25,
    verbose=False,
):
    """
    Sets config generator parameters for some basic data generation settings.

    Args:
        generator (robomimic ConfigGenerator instance): config generator object
        group (int): parameter group for these settings
        source_dataset_path (str): path to source dataset
        source_dataset_name (str): name to give source dataset in experiment name
        generation_path (str): folder for generated data
        guarantee (bool): whether to ensure @num_traj successes
        num_traj (int): number of trajectories for generation
        num_src_demos (int or None): number of source demos to take from @source_dataset_path
        max_num_failures (int): max failures to keep
        num_demo_to_render (int): max demos to render to video
        num_fail_demo_to_render (int): max fail demos to render to video
        verbose (bool): if True, make experiment name verbose using the passed settings
    """

    # set source dataset
    generator.add_param(
        key="experiment.source.dataset_path",
        name="src" if source_dataset_name is not None else "",
        group=group,
        values=[source_dataset_path],
        value_names=[source_dataset_name],
    )

    # set number of demos to use from source dataset
    generator.add_param(
        key="experiment.source.n",
        name="n_src" if verbose else "",
        group=group,
        values=[num_src_demos],
    )

    # set generation settings
    generator.add_param(
        key="experiment.generation.path",
        name="", 
        group=group, 
        values=[generation_path],
    )
    generator.add_param(
        key="experiment.generation.guarantee",
        name="gt" if verbose else "", 
        group=group, 
        values=[guarantee],
        value_names=["t" if guarantee else "f"],
    )
    generator.add_param(
        key="experiment.generation.num_trials",
        name="nt" if verbose else "", 
        group=group, 
        values=[num_traj],
    )
    generator.add_param(
        key="experiment.max_num_failures",
        name="", 
        group=group, 
        values=[max_num_failures],
    )
    generator.add_param(
        key="experiment.num_demo_to_render",
        name="", 
        group=group, 
        values=[num_demo_to_render],
    )
    generator.add_param(
        key="experiment.num_fail_demo_to_render",
        name="", 
        group=group, 
        values=[num_fail_demo_to_render],
    )

    return generator


def set_obs_settings(
    generator,
    group,
    collect_obs,
    camera_names,
    camera_height,
    camera_width,
):
    """
    Sets config generator parameters for collecting observations.
    """
    generator.add_param(
        key="obs.collect_obs",
        name="", 
        group=group, 
        values=[collect_obs],
    )
    generator.add_param(
        key="obs.camera_names",
        name="", 
        group=group, 
        values=[camera_names],
    )
    generator.add_param(
        key="obs.camera_height",
        name="", 
        group=group, 
        values=[camera_height],
    )
    generator.add_param(
        key="obs.camera_width",
        name="", 
        group=group, 
        values=[camera_width],
    )
    return generator


def set_subtask_settings(
    generator,
    group,
    base_config_file,
    select_src_per_subtask,
    subtask_term_offset_range=None,
    selection_strategy=None,
    selection_strategy_kwargs=None,
    action_noise=None,
    num_interpolation_steps=None,
    num_fixed_steps=None,
    verbose=False,
):
    """
    Sets config generator parameters for each subtask.

    Args:
        generator (robomimic ConfigGenerator instance): config generator object
        group (int): parameter group for these settings
        base_config_file (str): path to base config file being used for generating configs
        select_src_per_subtask (bool): whether to select src demo for each subtask
        subtask_term_offset_range (list or None): if provided, should be list of 2-tuples, one
            entry per subtask, with the last entry being None
        selection_strategy (str or None): src demo selection strategy
        selection_strategy_kwargs (dict or None): kwargs for selection strategy
        action_noise (float or list or None): action noise for all subtasks
        num_interpolation_steps (int or list or None): interpolation steps for all subtasks
        num_fixed_steps (int or list or None): interpolation steps for all subtasks
        verbose (bool): if True, make experiment name verbose using the passed settings
    """

    # get number of subtasks
    with open(base_config_file, 'r') as f:
        config = json.load(f)
        num_subtasks = len(config["task"]["task_spec"])

    # whether to select a different source demonstration for each subtask
    generator.add_param(
        key="experiment.generation.select_src_per_subtask",
        name="select_src_per_subtask" if verbose else "",
        group=group,
        values=[select_src_per_subtask],
        value_names=["t" if select_src_per_subtask else "f"],
    )

    # settings for each subtask

    # offset range
    if subtask_term_offset_range is not None:
        assert len(subtask_term_offset_range) == num_subtasks
        for i in range(num_subtasks):
            if (i == num_subtasks - 1):
                assert subtask_term_offset_range[i] is None
            else:
                assert (subtask_term_offset_range[i] is None) or (len(subtask_term_offset_range[i]) == 2)
            generator.add_param(
                key="task.task_spec.subtask_{}.subtask_term_offset_range".format(i + 1),
                name="offset" if (verbose and (i == 0)) else "", 
                group=group,
                values=[subtask_term_offset_range[i]],
            )

    # selection strategy
    if selection_strategy is not None:
        for i in range(num_subtasks):
            generator.add_param(
                key="task.task_spec.subtask_{}.selection_strategy".format(i + 1),
                name="ss" if (verbose and (i == 0)) else "", 
                group=group,
                values=[selection_strategy],
            )

    # selection kwargs
    if selection_strategy_kwargs is not None:
        for i in range(num_subtasks):
            generator.add_param(
                key="task.task_spec.subtask_{}.selection_strategy_kwargs".format(i + 1),
                name="", 
                group=group,
                values=[selection_strategy_kwargs],
            )

    # action noise
    if action_noise is not None:
        if not isinstance(action_noise, Iterable):
            action_noise = [action_noise for _ in range(num_subtasks)]
        assert len(action_noise) == num_subtasks
        for i in range(num_subtasks):
            generator.add_param(
                key="task.task_spec.subtask_{}.action_noise".format(i + 1),
                name="noise" if (verbose and (i == 0)) else "", 
                group=group,
                values=[action_noise[i]],
            )

    # interpolation
    if num_interpolation_steps is not None:
        if not isinstance(num_interpolation_steps, Iterable):
            num_interpolation_steps = [num_interpolation_steps for _ in range(num_subtasks)]
        assert len(num_interpolation_steps) == num_subtasks
        for i in range(num_subtasks):
            generator.add_param(
                key="task.task_spec.subtask_{}.num_interpolation_steps".format(i + 1),
                name="ni" if (verbose and (i == 0)) else "", 
                group=group,
                values=[num_interpolation_steps[i]],
            )
    if num_fixed_steps is not None:
        if not isinstance(num_fixed_steps, Iterable):
            num_fixed_steps = [num_fixed_steps for _ in range(num_subtasks)]
        assert len(num_fixed_steps) == num_subtasks
        for i in range(num_subtasks):
            generator.add_param(
                key="task.task_spec.subtask_{}.num_fixed_steps".format(i + 1),
                name="ni" if (verbose and (i == 0)) else "", 
                group=group,
                values=[num_fixed_steps[i]],
            )

    return generator


def set_learning_settings_for_bc_rnn(
    generator,
    group,
    modality,
    seq_length=10,
    low_dim_keys=None,
    image_keys=None,
    crop_size=None,
):
    """
    Sets config generator parameters for robomimic BC-RNN training runs.

    Args:
        generator (robomimic ConfigGenerator instance): config generator object
        group (int): parameter group for these settings
        modality (str): whether this is a low-dim or image observation run
        seq_length (int): BC-RNN context length
        low_dim_keys (list or None): if provided, set low-dim observation keys, else use defaults
        image_keys (list or None): if provided, set image observation keys, else use defaults
        crop_size (tuple or None): if provided, size of crop to use for pixel shift augmentation
    """
    supported_modalities = ["low_dim", "image"]
    assert modality in supported_modalities, "got modality {} not in supported modalities {}".format(modality, supported_modalities)

    # setup RNN with GMM and desired seq_length
    generator.add_param(
        key="train.seq_length",
        name="",
        group=group,
        values=[seq_length],
    )
    generator.add_param(
        key="algo.rnn.horizon",
        name="",
        group=group,
        values=[seq_length],
    )
    generator.add_param(
        key="algo.rnn.enabled",
        name="",
        group=group,
        values=[True],
    )
    generator.add_param(
        key="algo.gmm.enabled",
        name="",
        group=group,
        values=[True],
    )
    actor_layer_dims = []
    generator.add_param(
        key="algo.actor_layer_dims",
        name="",
        group=group,
        values=[actor_layer_dims],
    )

    # 4 data workers and low-dim cache mode seems to work well for both low-dim and image observations
    generator.add_param(
        key="train.num_data_workers",
        name="",
        group=group,
        values=[4],
    )
    generator.add_param(
        key="train.hdf5_cache_mode",
        name="",
        group=group,
        values=["low_dim"],
    )

    # modality-specific defaults
    if modality == "image":

        # epoch settings
        epoch_every_n_steps = 500
        validation_epoch_every_n_steps = 50
        eval_rate = 20

        # learning settings
        num_epochs = 600
        batch_size = 16
        policy_lr = 1e-4
        rnn_hidden_dim = 1000

        # observation settings
        if low_dim_keys is None:
            low_dim_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        if image_keys is None:
            image_keys = ["agentview_image", "robot0_eye_in_hand_image"]
        if crop_size is None:
            crop_size = (76, 76)

        generator.add_param(
            key="observation.encoder.rgb", 
            name="", 
            group=group, 
            values=[{
                "core_class": "VisualCore",
                "core_kwargs": {
                    "feature_dimension": 64,
                    "flatten": True,
                    "backbone_class": "ResNet18Conv",
                    "backbone_kwargs": {
                        "pretrained": False,
                        "input_coord_conv": False,
                    },
                    "pool_class": "SpatialSoftmax",
                    "pool_kwargs": {
                        "num_kp": 32,
                        "learnable_temperature": False,
                        "temperature": 1.0,
                        "noise_std": 0.0,
                        "output_variance": False,
                    },
                },
                "obs_randomizer_class": "CropRandomizer",
                "obs_randomizer_kwargs": {
                    "crop_height": crop_size[0],
                    "crop_width": crop_size[1],
                    "num_crops": 1,
                    "pos_enc": False,
                },
            }],
        )

    else:

        # epoch settings
        epoch_every_n_steps = 100
        validation_epoch_every_n_steps = 10
        eval_rate = 50

        # learning settings
        num_epochs = 2000
        batch_size = 100
        policy_lr = 1e-3
        rnn_hidden_dim = 400

        # observation settings
        if low_dim_keys is None:
            low_dim_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
        if image_keys is None:
            image_keys = []

    generator.add_param(
        key="observation.modalities.obs.low_dim", 
        name="", 
        group=group, 
        values=[low_dim_keys],
    )
    generator.add_param(
        key="observation.modalities.obs.rgb", 
        name="", 
        group=group, 
        values=[image_keys],
    )

    # epoch settings
    generator.add_param(
        key="experiment.epoch_every_n_steps",
        name="",
        group=group,
        values=[epoch_every_n_steps],
    )
    generator.add_param(
        key="experiment.validation_epoch_every_n_steps",
        name="",
        group=group,
        values=[validation_epoch_every_n_steps],
    )
    generator.add_param(
        key="experiment.save.every_n_epochs",
        name="",
        group=group,
        values=[eval_rate],
    )
    generator.add_param(
        key="experiment.rollout.rate",
        name="",
        group=group,
        values=[eval_rate],
    )

    # learning settings
    generator.add_param(
        key="train.num_epochs",
        name="",
        group=group,
        values=[num_epochs],
    )
    generator.add_param(
        key="train.batch_size",
        name="",
        group=group,
        values=[batch_size],
    )
    generator.add_param(
        key="algo.optim_params.policy.learning_rate.initial",
        name="",
        group=group,
        values=[policy_lr],
    )
    generator.add_param(
        key="algo.rnn.hidden_dim",
        name="",
        group=group,
        values=[rnn_hidden_dim],
    )

    return generator
