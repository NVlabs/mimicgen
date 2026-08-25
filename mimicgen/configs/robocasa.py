"""MimicGen task config for robocasa's WashAndReturnPlates (one-plate chain).

9 subtasks matching the demo_0.hdf5 subtask_label boundaries and
WashAndReturnPlatesInterface.SUBTASK_LABELS (8 of the 9 - the final one
never needs a term signal). Each object_ref must be a key
WashAndReturnPlatesInterface.get_object_poses() actually returns
(plate_1, sponge, cabinet, sink).
"""
from mimicgen.configs.config import MG_Config


class WashAndReturnPlates_Config(MG_Config):
    """Corresponds to robocasa's WashAndReturnPlates task (one-plate chain)."""

    NAME = "wash_and_return_plates"
    TYPE = "robosuite"

    def task_config(self):
        common = dict(
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )

        self.task.task_spec.subtask_1 = dict(
            object_ref="cabinet",
            subtask_term_signal="1_open_cabinet",
            subtask_term_offset_range=(5, 10),
            **common,
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref="sponge",
            subtask_term_signal="2_acquire_sponge",
            subtask_term_offset_range=(5, 10),
            **common,
        )
        self.task.task_spec.subtask_3 = dict(
            object_ref="sink",
            subtask_term_signal="3_turn_on_faucet",
            subtask_term_offset_range=(5, 10),
            **common,
        )
        self.task.task_spec.subtask_4 = dict(
            object_ref="plate_1",
            subtask_term_signal="4_pick_plate_1",
            subtask_term_offset_range=(5, 10),
            **common,
        )
        self.task.task_spec.subtask_5 = dict(
            object_ref="plate_1",
            subtask_term_signal="5_wash_plate_1",
            subtask_term_offset_range=(5, 10),
            **common,
        )
        self.task.task_spec.subtask_6 = dict(
            object_ref="plate_1",
            subtask_term_signal="6_store_plate_1",
            subtask_term_offset_range=(5, 10),
            **common,
        )
        self.task.task_spec.subtask_7 = dict(
            object_ref="sponge",
            subtask_term_signal="7_return_sponge_only",
            subtask_term_offset_range=(5, 10),
            **common,
        )
        self.task.task_spec.subtask_8 = dict(
            object_ref="sink",
            subtask_term_signal="8_turn_off_faucet_only",
            subtask_term_offset_range=(5, 10),
            **common,
        )
        self.task.task_spec.subtask_9 = dict(
            object_ref="cabinet",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            **common,
        )
        self.task.task_spec.do_not_lock_keys()
