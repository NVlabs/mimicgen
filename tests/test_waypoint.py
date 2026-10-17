# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Tests for mimicgen.datagen.waypoint.

Run with: python -m pytest tests/test_waypoint.py (from the repo root)
"""

import numpy as np

from mimicgen.datagen.waypoint import Waypoint, WaypointTrajectory
import mimicgen.utils.pose_utils as PoseUtils


def _make_pose(pos=(0., 0., 0.5)):
    """
    Helper to make a 4x4 homogeneous pose with identity rotation.
    """
    return PoseUtils.make_pose(
        pos=np.array(pos),
        rot=np.eye(3),
    )


def test_add_waypoint_sequence_for_target_pose_skip_interpolation():
    """
    Adding fixed (non-interpolated) waypoint sequences should repeat the target
    pose and gripper action @num_steps times, without crashing.

    Regression test for https://github.com/NVlabs/mimicgen/issues/39: this path
    used an undefined name `gripper` instead of the `gripper_action` argument,
    raising NameError whenever a task spec set `num_fixed_steps > 0`.
    """
    traj = WaypointTrajectory()

    # first sequence must be non-interpolated (nothing to interpolate from)
    traj.add_waypoint_sequence_for_target_pose(
        pose=_make_pose(),
        gripper_action=np.array([1.0]),
        num_steps=3,
        skip_interpolation=True,
    )

    # a second fixed segment with a different gripper action
    traj.add_waypoint_sequence_for_target_pose(
        pose=_make_pose(pos=(0.1, 0., 0.5)),
        gripper_action=np.array([-1.0]),
        num_steps=2,
        skip_interpolation=True,
    )

    assert len(traj) == 5

    # all poses in each segment should equal the segment's target pose
    for ind in range(len(traj)):
        waypoint = traj[ind]
        assert isinstance(waypoint, Waypoint)
        assert waypoint.pose.shape == (4, 4)
        if ind < 3:
            assert waypoint.gripper_action.shape == (1,)
            np.testing.assert_allclose(waypoint.gripper_action, [1.0])
            np.testing.assert_allclose(waypoint.pose, _make_pose())
        else:
            assert waypoint.gripper_action.shape == (1,)
            np.testing.assert_allclose(waypoint.gripper_action, [-1.0])
            np.testing.assert_allclose(waypoint.pose, _make_pose(pos=(0.1, 0., 0.5)))


def test_add_waypoint_sequence_for_target_pose_interpolation():
    """
    Interpolated waypoint sequences should keep pose and gripper action
    lengths consistent (both @num_steps + 1 after dropping the first element).
    """
    traj = WaypointTrajectory()
    start_pose = _make_pose()

    traj.add_waypoint_sequence_for_target_pose(
        pose=start_pose,
        gripper_action=np.array([1.0]),
        num_steps=2,
        skip_interpolation=True,
    )

    traj.add_waypoint_sequence_for_target_pose(
        pose=_make_pose(pos=(0.2, 0., 0.5)),
        gripper_action=np.array([-1.0]),
        num_steps=4,
        skip_interpolation=False,
    )

    # 2 from the fixed segment + (num_steps + 1) interpolated targets (first is dropped)
    assert len(traj) == 2 + 4 + 1

    for ind in range(2, len(traj)):
        waypoint = traj[ind]
        assert waypoint.gripper_action.shape == (1,)
        np.testing.assert_allclose(waypoint.gripper_action, [-1.0])

    # last pose should be the requested target pose
    np.testing.assert_allclose(traj[len(traj) - 1].pose, _make_pose(pos=(0.2, 0., 0.5)))


def test_merge_with_fixed_steps():
    """
    Merging with @num_steps_fixed > 0 (the path data_generator.py uses for
    task specs with num_fixed_steps > 0) should create a constant-pose
    segment repeating the first waypoint of @other, without crashing.
    """
    traj = WaypointTrajectory()
    traj.add_waypoint_sequence_for_target_pose(
        pose=_make_pose(),
        gripper_action=np.array([1.0]),
        num_steps=2,
        skip_interpolation=True,
    )

    other = WaypointTrajectory()
    other.add_waypoint_sequence_for_target_pose(
        pose=_make_pose(pos=(0.3, 0., 0.5)),
        gripper_action=np.array([-1.0]),
        num_steps=2,
        skip_interpolation=True,
    )

    traj.merge(
        other,
        num_steps_interp=3,
        num_steps_fixed=2,
        action_noise=0.,
    )

    # 2 original + 4 interpolated (num_steps_interp + 1) + 2 fixed + 1 rest of other
    assert len(traj) == 2 + 4 + 2 + 1

    # the fixed segment (indices 6 and 7) repeats other's first target pose
    fixed_first = traj[6]
    np.testing.assert_allclose(fixed_first.pose, _make_pose(pos=(0.3, 0., 0.5)))
    np.testing.assert_allclose(fixed_first.gripper_action, [-1.0])
    np.testing.assert_allclose(traj[7].pose, fixed_first.pose)
    np.testing.assert_allclose(traj[7].gripper_action, fixed_first.gripper_action)
