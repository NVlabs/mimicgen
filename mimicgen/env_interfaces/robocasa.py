"""MimicGen environment interfaces for robocasa's AgiBotG2 whole-body robot.

BimanualRobosuiteInterface: generic bimanual coordination.
WashAndReturnPlatesInterface: the one task on it.
"""
from typing import Any

import numpy as np

import robosuite
import robosuite.utils.transform_utils as T

import mimicgen.utils.pose_utils as PoseUtils
from mimicgen.env_interfaces.robosuite import RobosuiteInterface
from mimicgen.datagen.waypoint import WaypointSequence, WaypointTrajectory

import robocasa  # registers WashAndReturnPlates etc. with robosuite
from robocasa.environments.kitchen.composite.bimanual.wash_and_return_plates import (
    DIRTY_PLATE_NAMES,
    SPONGE_NAME,
)


class BimanualRobosuiteInterface(RobosuiteInterface):
    """Bimanual coordination for AgiBotG2 (or any composite-controller 2-arm
    robot). RobosuiteInterface assumes env.robots[0].controller (single-arm);
    G2 uses env.robots[0].composite_controller (7 parts: right/left arm+gripper,
    torso, head, base).

    A subclass must declare:
        SUBTASK_ORDER (tuple[str]): every subtask, in generation order.
        SUBTASK_PRIMARY_ARM (dict[str, str]): "right"/"left" per subtask - the
            actively-driven arm; the other holds still (gripper keeps the grasp).
        BIMANUAL_SUBTASKS (tuple[str]): subtasks needing both arms driven at
            once (see _build_secondary_trajectory).
        get_object_poses() (from RobosuiteInterface): still task-specific.
    """

    RIGHT_EEF_SITE = "gripper0_right_grip_site"
    LEFT_EEF_SITE = "gripper0_left_grip_site"
    # G2 gripper qpos at fully open/closed - hardware constants, not per-task.
    GRIPPER_OPEN_REF = -0.785
    GRIPPER_CLOSED_REF = -0.124

    def __init__(self, env: Any):
        """Init subtask-boundary tracking + generation-side hook targets."""
        super().__init__(env)
        boundary_names = self.SUBTASK_ORDER[:-1]  # final subtask needs no term signal
        self._boundary_names = boundary_names
        self._seen_boundary = {name: False for name in boundary_names}
        self._past_boundary = {name: False for name in boundary_names}
        self._current_label = None  # set by get_datagen_info(), source-parsing only

        self._subtask_ind = -1  # set by data_generator.py; -1 means source-demo parsing
        self._current_src_segment = None  # set by data_generator.py; (DatagenInfo, (start, end))
        self._primary_traj_len = None  # set by data_generator.py; primary arm's step count this subtask
        self._primary_num_interp_steps = None  # set by data_generator.py; primary's own bridge length this subtask
        self._primary_action_noise = None  # set by data_generator.py; same action_noise the primary's real segment gets

        # Secondary-arm trajectory (BIMANUAL_SUBTASKS), built lazily per subtask.
        self._secondary_traj = None
        self._secondary_traj_subtask_ind = None
        self._secondary_step = 0

    def _eef_site(self, arm: str) -> str:
        """Gripper site name for arm ("right"/"left")."""
        return self.RIGHT_EEF_SITE if arm == "right" else self.LEFT_EEF_SITE

    def _primary_arm_for_label(self, label) -> str:
        """SUBTASK_PRIMARY_ARM lookup for a raw (possibly bytes) ground-truth label."""
        if isinstance(label, bytes):
            label = label.decode()
        return self.SUBTASK_PRIMARY_ARM.get(label, "right")

    def get_robot_eef_pose(self) -> np.ndarray:
        """Current primary arm's eef pose (source-frame arm during parsing, tracked arm during generation)."""
        if self._subtask_ind < 0:
            arm = self._primary_arm_for_label(self._current_label)
        else:
            arm = self._infer_current_primary_arm()
        return self.get_object_pose(obj_name=self._eef_site(arm), obj_type="site")

    def _infer_current_primary_arm(self) -> str:
        """Primary arm for the current live-generation subtask (ground truth, via _subtask_ind)."""
        idx = min(max(self._subtask_ind, 0), len(self.SUBTASK_ORDER) - 1)
        return self.SUBTASK_PRIMARY_ARM[self.SUBTASK_ORDER[idx]]

    def _arm_target_pose(self, action: np.ndarray, arm: str) -> np.ndarray:
        """Target eef pose for arm, decoded from action's [right_pos+aa, left_pos+aa, ...] layout."""
        offset = 0 if arm == "right" else 6
        target_pos = np.asarray(action[offset : offset + 3], dtype=float)
        target_axisangle = np.asarray(action[offset + 3 : offset + 6], dtype=float)
        target_rot = T.quat2mat(T.axisangle2quat(target_axisangle))
        return PoseUtils.make_pose(target_pos, target_rot)

    def action_to_target_pose(self, action: np.ndarray, relative: bool = True) -> np.ndarray:
        """Source-side only (prepare_src_dataset.py's replay) - ground truth arm via the current label."""
        arm = self._primary_arm_for_label(self._current_label)
        return self._arm_target_pose(action, arm)

    def action_to_gripper_action(self, action: np.ndarray) -> np.ndarray:
        """Binarizes to the controller's [-1, 1] open/close convention (not a raw joint angle)."""
        close_threshold = self.GRIPPER_CLOSED_REF + 0.2 * (  # 20% short of closed, not the midpoint
            self.GRIPPER_OPEN_REF - self.GRIPPER_CLOSED_REF
        )
        raw = np.asarray(action[12:14], dtype=float)
        return np.where(raw > close_threshold, 1.0, -1.0)

    def get_datagen_info(self, action=None, label=None):
        """Bimanual: captures both arms' target poses into datagen_info, for the
        secondary arm's motion during generation (see _build_secondary_trajectory)."""
        self._current_label = label
        datagen_info = super().get_datagen_info(action=action, label=label)
        if action is not None:
            datagen_info.right_target_pose = self._arm_target_pose(action, "right")
            datagen_info.left_target_pose = self._arm_target_pose(action, "left")
        return datagen_info

    def get_subtask_term_signals(self) -> dict:
        """Ground-truth per-subtask completion flags from the current label."""
        label = self._current_label.decode() if isinstance(self._current_label, bytes) else self._current_label
        signals = {}
        for name in self._boundary_names:
            if label == name:
                self._seen_boundary[name] = True
            elif self._seen_boundary[name] and label != name:
                self._past_boundary[name] = True
            signals[name] = int(self._past_boundary[name])
        return signals

    def _secondary_arm(self, primary_arm: str) -> str:
        """The other arm."""
        return "left" if primary_arm == "right" else "right"

    def _delta_action(self, arm: str, target_pose: np.ndarray) -> np.ndarray:
        """Delta-pose-to-action math against arm's own part controller. The
        OSC controller's input_ref_frame is "base" (G2 is mobile, base isn't
        world-aligned) - deltas must be rotated world->base before clipping,
        unlike a fixed-base arm where the two frames coincide."""
        target_pos, target_rot = PoseUtils.unmake_pose(target_pose)
        curr_pose = self.get_object_pose(obj_name=self._eef_site(arm), obj_type="site")
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)
        part = self.env.robots[0].composite_controller.part_controllers[arm]
        max_dpos = part.output_max[0]
        max_drot = part.output_max[3]
        base_rot = part.origin_ori
        delta_position_world = target_pos - curr_pos
        delta_position = np.clip(base_rot.T.dot(delta_position_world) / max_dpos, -1.0, 1.0)
        delta_rot_mat = target_rot.dot(curr_rot.T)
        delta_quat = T.mat2quat(delta_rot_mat)
        delta_axisangle_world = T.quat2axisangle(delta_quat)
        delta_rotation = np.clip(base_rot.T.dot(delta_axisangle_world) / max_drot, -1.0, 1.0)
        return np.concatenate([delta_position, delta_rotation])

    def _build_secondary_trajectory(self, subtask_ind: int, secondary_arm: str):
        """Precompute the secondary arm's trajectory for a BIMANUAL_SUBTASKS entry."""
        # REPLAY (raw source poses) - only valid if live/source frame-0 poses match.
        src_ep_datagen_info, (start, end) = self._current_src_segment
        src_poses = (
            src_ep_datagen_info.right_target_pose if secondary_arm == "right" else src_ep_datagen_info.left_target_pose
        )[start:end]
        gripper_col = 0 if secondary_arm == "right" else 1
        src_grippers = src_ep_datagen_info.gripper_action[start:end, gripper_col : gripper_col + 1]

        traj = WaypointTrajectory()
        cur_pose = self.get_object_pose(
            obj_name=self._eef_site(secondary_arm), obj_type="site"
        )
        init_seq = WaypointSequence.from_poses(
            poses=cur_pose[None], gripper_actions=src_grippers[0:1], action_noise=0.0
        )
        traj.add_waypoint_sequence(init_seq)

        real_traj = WaypointTrajectory()
        real_traj.add_waypoint_sequence(
            WaypointSequence.from_poses(
                poses=src_poses, gripper_actions=src_grippers,
                action_noise=self._primary_action_noise or 0.0,
            )
        )
        # Match the primary's own bridge length (not hardcoded) - a mismatch
        # desyncs the two arms' real-segment start by the difference every trial.
        bridge_steps = self._primary_num_interp_steps or 30
        traj.merge(real_traj, num_steps_interp=bridge_steps, num_steps_fixed=0, action_noise=0.0)
        traj.pop_first()

        self._secondary_traj = traj
        self._secondary_traj_subtask_ind = subtask_ind
        self._secondary_step = 0

    def target_pose_to_action(self, target_pose: np.ndarray, relative: bool = True) -> np.ndarray:
        """Primary arm's delta action, plus the secondary arm's replayed motion for BIMANUAL_SUBTASKS."""
        assert robosuite.__version__.split(".")[0] == "1"

        arm = self._infer_current_primary_arm()
        idx = min(max(self._subtask_ind, 0), len(self.SUBTASK_ORDER) - 1)
        label = self.SUBTASK_ORDER[idx]

        cc = self.env.robots[0].composite_controller
        action_dict = {
            name: np.zeros(ctrl.control_dim) for name, ctrl in cc.part_controllers.items()
        }
        action_dict[arm] = self._delta_action(arm, target_pose)

        if label in self.BIMANUAL_SUBTASKS:
            secondary_arm = self._secondary_arm(arm)
            if self._secondary_traj_subtask_ind != idx:
                self._build_secondary_trajectory(idx, secondary_arm)
            # 1 real waypoint per tick, same cadence primary gets from
            # WaypointTrajectory.execute() - the earlier proportional-frac
            # pacing could round-skip a waypoint, which cost this arm the
            # brief handle-catch contact window (measured: left succeeded far
            # less often than right on the identical delta-action code).
            # Holds at the final waypoint if this trajectory is shorter than
            # the primary's - same safe idle-hold semantics used elsewhere.
            step = min(self._secondary_step, len(self._secondary_traj) - 1)
            waypoint = self._secondary_traj[step]
            self._secondary_step += 1
            action_dict[secondary_arm] = self._delta_action(secondary_arm, waypoint.pose)

        full_vec = cc.create_action_vector(action_dict)
        return full_vec[:-2]  # gripper dims: action_to_gripper_action supplies these


class WashAndReturnPlatesInterface(BimanualRobosuiteInterface):
    """MimicGen environment interface for robocasa's WashAndReturnPlates task."""

    SUBTASK_ORDER = (
        "1_open_cabinet",
        "2_acquire_sponge",
        "3_turn_on_faucet",
        "4_pick_plate_1",
        "5_wash_plate_1",
        "6_store_plate_1",
        "7_return_sponge_only",
        "8_turn_off_faucet_only",
        "9_close_cabinet_and_home",
    )
    # Arm assignment grounded in direct measurement, not assumed.
    SUBTASK_PRIMARY_ARM = {
        "1_open_cabinet": "right",  # BIMANUAL: see BIMANUAL_SUBTASKS
        "2_acquire_sponge": "right",  # INFERRED: sponge held by RIGHT in wash_plate_1
        "3_turn_on_faucet": "left",  # VERIFIED: left hand 9cm from sink, right 54cm+
        "4_pick_plate_1": "left",  # INFERRED: plate held by LEFT in wash_plate_1
        "5_wash_plate_1": "right",  # OBSERVED: right holds sponge, left holds plate
        "6_store_plate_1": "left",  # INFERRED: plate held by LEFT in wash_plate_1
        "7_return_sponge_only": "right",  # INFERRED: sponge held by RIGHT in wash_plate_1
        "8_turn_off_faucet_only": "left",  # ASSUMED: mirrors 3_turn_on_faucet
        "9_close_cabinet_and_home": "right",  # BIMANUAL: see BIMANUAL_SUBTASKS
    }
    # REPLAY valid: live post-reset handle pose == source frame-0 pose (0.0 delta, measured).
    BIMANUAL_SUBTASKS = ("1_open_cabinet", "9_close_cabinet_and_home")

    def get_object_poses(self) -> dict:
        """4 object refs: plate_1, sponge, cabinet, sink."""
        # Every body (sampled object or fixture) registers with a "_main" suffix.
        return {
            "plate_1": self.get_object_pose(obj_name=f"{DIRTY_PLATE_NAMES[0]}_main", obj_type="body"),
            "sponge": self.get_object_pose(obj_name=f"{SPONGE_NAME}_main", obj_type="body"),
            "cabinet": self.get_object_pose(obj_name=f"{self.env.cabinet.name}_main", obj_type="body"),
            "sink": self.get_object_pose(obj_name=f"{self.env.sink.name}_main", obj_type="body"),
        }


class AcquireDishBrushInterface(BimanualRobosuiteInterface):
    """Pilot: isolated single-subtask generation for the standalone
    AcquireDishBrush task (scripts/mimicgen_seed/demo_0_AcquireDishBrush.hdf5).
    Tests whether splitting generation per-subtask avoids the full-chains
    compounding drift (only subtask 1 succeeds there)."""

    SUBTASK_ORDER = ("2_acquire_sponge",)
    SUBTASK_PRIMARY_ARM = {"2_acquire_sponge": "right"}  # INFERRED: sponge held by RIGHT in wash_plate_1
    BIMANUAL_SUBTASKS = ()

    def get_object_poses(self) -> dict:
        """1 object ref: sponge."""
        return {"sponge": self.get_object_pose(obj_name=f"{SPONGE_NAME}_main", obj_type="body")}


class OpenWashCabinetInterface(BimanualRobosuiteInterface):
    """Isolated single-subtask generation for the standalone OpenWashCabinet
    task (scripts/mimicgen_seed/demo_0_OpenWashCabinet.hdf5). Bimanual -
    single arm produces zero door motion on this cabinet, confirmed
    empirically (see wash_and_return_plates.py)."""

    SUBTASK_ORDER = ("1_open_cabinet",)
    SUBTASK_PRIMARY_ARM = {"1_open_cabinet": "right"}
    BIMANUAL_SUBTASKS = ("1_open_cabinet",)

    def get_object_poses(self) -> dict:
        """1 object ref: cabinet."""
        return {"cabinet": self.get_object_pose(obj_name=f"{self.env.cabinet.name}_main", obj_type="body")}
