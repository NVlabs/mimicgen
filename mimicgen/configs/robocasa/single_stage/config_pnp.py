"""
Some configs for robosuite.
"""
import mimicgen
from mimicgen.configs.config import MG_Config

class PnPCounterToCab_Config(MG_Config):
    NAME = "PnPCounterToCab"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has task settings such as the task specification (the
        stages of each task, the amount of noise to apply during each stage, etc).
        """
        self.task.task_spec.stage_1 = dict(
            object_ref="obj", 
            subtask_term_signal="stage_contact_obj", 
            subtask_term_offset_range=(10, 20),
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        self.task.task_spec.stage_2 = dict(
            object_ref="cab", 
            subtask_term_signal=None, 
            subtask_term_offset_range=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        self.task.task_spec.do_not_lock_keys() # allow downstream code to completely replace the task spec


class PnPCabToCounter_Config(MG_Config):
    NAME = "PnPCabToCounter"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has task settings such as the task specification (the
        stages of each task, the amount of noise to apply during each stage, etc).
        """
        self.task.task_spec.stage_1 = dict(
            object_ref="obj", 
            subtask_term_signal="stage_contact_obj", 
            subtask_term_offset_range=(10, 20),
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        self.task.task_spec.stage_2 = dict(
            object_ref="obj", 
            subtask_term_signal=None, 
            subtask_term_offset_range=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        self.task.task_spec.do_not_lock_keys() # allow downstream code to completely replace the task spec


class PnPCounterToSink_Config(MG_Config):
    NAME = "PnPCounterToSink"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has task settings such as the task specification (the
        stages of each task, the amount of noise to apply during each stage, etc).
        """
        self.task.task_spec.stage_1 = dict(
            object_ref="obj", 
            subtask_term_signal="stage_contact_obj", 
            subtask_term_offset_range=(10, 20),
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        self.task.task_spec.stage_2 = dict(
            object_ref="sink", 
            subtask_term_signal=None, 
            subtask_term_offset_range=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        self.task.task_spec.do_not_lock_keys() # allow downstream code to completely replace the task spec


class PnPSinkToCounter_Config(MG_Config):
    NAME = "PnPSinkToCounter"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has task settings such as the task specification (the
        stages of each task, the amount of noise to apply during each stage, etc).
        """
        self.task.task_spec.stage_1 = dict(
            object_ref="obj", 
            subtask_term_signal="stage_contact_obj", 
            subtask_term_offset_range=(20, 30),
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        self.task.task_spec.stage_2 = dict(
            object_ref="container", 
            subtask_term_signal=None, 
            subtask_term_offset_range=None,
            action_noise=0.05,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        self.task.task_spec.do_not_lock_keys() # allow downstream code to completely replace the task spec


class PnPCounterToMicrowave_Config(MG_Config):
    NAME = "PnPCounterToMicrowave"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has task settings such as the task specification (the
        stages of each task, the amount of noise to apply during each stage, etc).
        """
        self.task.task_spec.stage_1 = dict(
            object_ref="obj", 
            subtask_term_signal="stage_contact_obj", 
            subtask_term_offset_range=(10, 20),
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        self.task.task_spec.stage_2 = dict(
            object_ref="microwave", 
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        self.task.task_spec.do_not_lock_keys() # allow downstream code to completely replace the task spec
        

class PnPMicrowaveToCounter_Config(MG_Config):
    NAME = "PnPMicrowaveToCounter"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has task settings such as the task specification (the
        stages of each task, the amount of noise to apply during each stage, etc).
        """
        self.task.task_spec.stage_1 = dict(
            object_ref="obj", 
            subtask_term_signal="stage_contact_obj", 
            subtask_term_offset_range=(10, 20),
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        self.task.task_spec.stage_2 = dict(
            object_ref="container", 
            subtask_term_signal=None, 
            subtask_term_offset_range=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        self.task.task_spec.do_not_lock_keys() # allow downstream code to completely replace the task spec


class PnPCounterToStove_Config(MG_Config):
    NAME = "PnPCounterToStove"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has task settings such as the task specification (the
        stages of each task, the amount of noise to apply during each stage, etc).
        """
        self.task.task_spec.stage_1 = dict(
            object_ref="obj", 
            subtask_term_signal="stage_contact_obj", 
            subtask_term_offset_range=(10, 20),
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        self.task.task_spec.stage_2 = dict(
            object_ref="cookware", 
            subtask_term_signal=None, 
            subtask_term_offset_range=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        self.task.task_spec.do_not_lock_keys() # allow downstream code to completely replace the task spec


class PnPStoveToCounter_Config(MG_Config):
    NAME = "PnPStoveToCounter"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has task settings such as the task specification (the
        stages of each task, the amount of noise to apply during each stage, etc).
        """
        self.task.task_spec.stage_1 = dict(
            object_ref="obj", 
            subtask_term_signal="stage_contact_obj", 
            subtask_term_offset_range=(10, 20),
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        self.task.task_spec.stage_2 = dict(
            object_ref="container", 
            subtask_term_signal=None, 
            subtask_term_offset_range=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            selection_strategy="nearest_neighbor_interpolation",
            selection_strategy_kwargs=dict(nn_k=5),
        )
        self.task.task_spec.do_not_lock_keys() # allow downstream code to completely replace the task spec