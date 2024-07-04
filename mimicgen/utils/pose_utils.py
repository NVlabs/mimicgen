# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
A collection of utilities for working with poses.
"""

import math
import collections
import numpy as np

import robosuite
import robosuite.utils.transform_utils as T


def make_pose(pos, rot):
    """
    Make homogenous pose matrices from a set of translation vectors and rotation matrices.

    Args:
        pos (np.array): batch of position vectors with last dimension of 3
        rot (np.array): batch of rotation matrices with last 2 dimensions of (3, 3)

    Returns:
        pose (np.array): batch of pose matrices with last 2 dimensions of (4, 4)
    """
    assert pos.shape[:-1] == rot.shape[:-2]
    assert pos.shape[-1] == rot.shape[-2] == rot.shape[-1] == 3
    pose = np.zeros(pos.shape[:-1] + (4, 4))
    pose[..., :3, :3] = rot
    pose[..., :3, 3] = pos
    pose[..., 3, 3] = 1.
    return pose


def unmake_pose(pose):
    """
    Split homogenous pose matrices back into translation vectors and rotation matrices.

    Args:
        pose (np.array): batch of pose matrices with last 2 dimensions of (4, 4)

    Returns:
        pos (np.array): batch of position vectors with last dimension of 3
        rot (np.array): batch of rotation matrices with last 2 dimensions of (3, 3)
    """
    return pose[..., :3, 3], pose[..., :3, :3]


def pose_inv(pose):
    """
    Computes the inverse of homogenous pose matrices.

    Note that the inverse of a pose matrix is the following:
    [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    Args:
        pose (np.array): batch of pose matrices with last 2 dimensions of (4, 4)


    Returns:
        inv_pose (np.array): batch of inverse pose matrices with last 2 dimensions of (4, 4)
    """
    num_axes = len(pose.shape)
    assert num_axes >= 2

    inv_pose = np.zeros_like(pose)

    # gymnastics to take transpose of last 2 dimensions
    inv_pose[..., :3, :3] = np.transpose(pose[..., :3, :3], tuple(range(num_axes - 2)) + (num_axes - 1, num_axes - 2))

    # note: numpy matmul wants shapes [..., 3, 3] x [..., 3, 1] -> [..., 3, 1] so we add a dimension and take it away after
    inv_pose[..., :3, 3] = np.matmul(-inv_pose[..., :3, :3], pose[..., :3, 3:4])[..., 0]
    inv_pose[..., 3, 3] = 1.0
    return inv_pose


def pose_in_A_to_pose_in_B(pose_in_A, pose_A_in_B):
    """
    Converts homogenous matrices corresponding to a point C in frame A
    to homogenous matrices corresponding to the same point C in frame B.

    Args:
        pose_in_A (np.array): batch of homogenous matrices corresponding to the pose of C in frame A
        pose_A_in_B (np.array): batch of homogenous matrices corresponding to the pose of A in frame B

    Returns:
        pose_in_B (np.array): batch of homogenous matrices corresponding to the pose of C in frame B
    """
    return np.matmul(pose_A_in_B, pose_in_A)


def quat2axisangle(quat):
    """
    Converts (x, y, z, w) quaternion to axis-angle format.
    Returns a unit vector direction and an angle.

    NOTE: this differs from robosuite's function because it returns
          both axis and angle, not axis * angle.
    """

    # conversion from axis-angle to quaternion:
    #   qw = cos(theta / 2); qx, qy, qz = u * sin(theta / 2)

    # normalize qx, qy, qz by sqrt(qx^2 + qy^2 + qz^2) = sqrt(1 - qw^2)
    # to extract the unit vector

    # clipping for scalar with if-else is orders of magnitude faster than numpy
    if quat[3] > 1.:
        quat[3] = 1.
    elif quat[3] < -1.:
        quat[3] = -1.

    den = np.sqrt(1. - quat[3] * quat[3])
    if math.isclose(den, 0.):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3), 0.

    return quat[:3] / den, 2. * math.acos(quat[3])


def axisangle2quat(axis, angle):
    """
    Converts axis-angle to (x, y, z, w) quat.

    NOTE: this differs from robosuite's function because it accepts
          both axis and angle as arguments, not axis * angle.
    """

    # handle zero-rotation case
    if math.isclose(angle, 0.):
        return np.array([0., 0., 0., 1.])

    # make sure that axis is a unit vector
    assert math.isclose(np.linalg.norm(axis), 1., abs_tol=1e-3)

    q = np.zeros(4)
    q[3] = np.cos(angle / 2.)
    q[:3] = axis * np.sin(angle / 2.)
    return q


def quat_slerp(q1, q2, tau):
    """
    Adapted from robosuite.
    """
    if tau == 0.0:
        return q1
    elif tau == 1.0:
        return q2
    d = np.dot(q1, q2)
    if abs(abs(d) - 1.0) < np.finfo(float).eps * 4.:
        return q1
    if d < 0.0:
        # invert rotation
        d = -d
        q2 *= -1.0
    angle = math.acos(np.clip(d, -1, 1))
    if abs(angle) < np.finfo(float).eps * 4.:
        return q1
    isin = 1.0 / math.sin(angle)
    q1 = q1 * math.sin((1.0 - tau) * angle) * isin
    q2 = q2 * math.sin(tau * angle) * isin
    q1 = q1 + q2
    return q1


def interpolate_rotations(R1, R2, num_steps, axis_angle=True):
    """
    Interpolate between 2 rotation matrices. If @axis_angle, interpolate the axis-angle representation
    of the delta rotation, else, use slerp.

    NOTE: I have verified empirically that both methods are essentially equivalent, so pick your favorite.
    """
    if axis_angle:
        # delta rotation expressed as axis-angle
        delta_rot_mat = R2.dot(R1.T)
        delta_quat = T.mat2quat(delta_rot_mat)
        delta_axis, delta_angle = quat2axisangle(delta_quat)

        # fix the axis, and chunk the angle up into steps
        rot_step_size = delta_angle / num_steps

        # convert into delta rotation matrices, and then convert to absolute rotations
        if delta_angle < 0.05:
            # small angle - don't bother with interpolation
            rot_steps = np.array([R2 for _ in range(num_steps)])
        else:
            delta_rot_steps = [T.quat2mat(axisangle2quat(delta_axis, i * rot_step_size)) for i in range(num_steps)]
            rot_steps = np.array([delta_rot_steps[i].dot(R1) for i in range(num_steps)])
    else:
        q1 = T.mat2quat(R1)
        q2 = T.mat2quat(R2)
        rot_steps = np.array([T.quat2mat(quat_slerp(q1, q2, tau=(float(i) / num_steps))) for i in range(num_steps)])
    
    # add in endpoint
    rot_steps = np.concatenate([rot_steps, R2[None]], axis=0)

    return rot_steps


def interpolate_poses(pose_1, pose_2, num_steps=None, step_size=None, perturb=False):
    """
    Linear interpolation between two poses.

    Args:
        pose_1 (np.array): 4x4 start pose
        pose_2 (np.array): 4x4 end pose
        num_steps (int): if provided, specifies the number of desired interpolated points (not excluding
            the start and end points). Passing 0 corresponds to no interpolation, and passing None
            means that @step_size must be provided to determine the number of interpolated points.
        step_size (float): if provided, will be used to infer the number of steps, by taking the norm
            of the delta position vector, and dividing it by the step size
        perturb (bool): if True, randomly move all the interpolated position points in a uniform, non-overlapping grid.
    
    Returns:
        pose_steps (np.array): array of shape (N + 2, 3) corresponding to the interpolated pose path, where N is @num_steps
        num_steps (int): the number of interpolated points (N) in the path
    """
    assert step_size is None or num_steps is None

    pos1, rot1 = unmake_pose(pose_1)
    pos2, rot2 = unmake_pose(pose_2)

    if num_steps == 0:
        # skip interpolation
        return np.concatenate([pos1[None], pos2[None]], axis=0), np.concatenate([rot1[None], rot2[None]], axis=0), num_steps

    delta_pos = pos2 - pos1
    if num_steps is None:
        assert np.linalg.norm(delta_pos) > 0
        num_steps = math.ceil(np.linalg.norm(delta_pos) / step_size)

    num_steps += 1  # include starting pose
    assert num_steps >= 2

    # linear interpolation of positions
    pos_step_size = delta_pos / num_steps
    grid = np.arange(num_steps).astype(np.float64)
    if perturb:
        # move the interpolation grid points by up to a half-size forward or backward
        perturbations = np.random.uniform(
            low=-0.5,
            high=0.5,
            size=(num_steps - 2,),
        )
        grid[1:-1] += perturbations
    pos_steps = np.array([pos1 + grid[i] * pos_step_size for i in range(num_steps)])

    # add in endpoint
    pos_steps = np.concatenate([pos_steps, pos2[None]], axis=0)

    # interpolate the rotations too
    rot_steps = interpolate_rotations(R1=rot1, R2=rot2, num_steps=num_steps, axis_angle=True)

    pose_steps = make_pose(pos_steps, rot_steps)
    return pose_steps, num_steps - 1


def transform_source_data_segment_using_object_pose(
    obj_pose,
    src_eef_poses,
    src_obj_pose,
):
    """
    Transform a source data segment (object-centric subtask segment from source demonstration) such that 
    the relative poses between the target eef pose frame and the object frame are preserved. Recall that 
    each object-centric subtask segment corresponds to one object, and consists of a sequence of 
    target eef poses.

    Args:
        obj_pose (np.array): 4x4 object pose in current scene
        src_eef_poses (np.array): pose sequence (shape [T, 4, 4]) for the sequence of end effector control poses 
            from the source demonstration
        src_obj_pose (np.array): 4x4 object pose from the source demonstration

    Returns:
        transformed_eef_poses (np.array): transformed pose sequence (shape [T, 4, 4])
    """

    # transform source end effector poses to be relative to source object frame

    # convert these eef poses from frame A (world frame) to frame B (source object frame)
    src_eef_poses_rel_obj = pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses,
        pose_A_in_B=pose_inv(src_obj_pose[None]),
    )

    # apply relative poses to current object frame to obtain new target eef poses

    # convert these eef poses from frame A (new object frame) to frame B (world frame)
    transformed_eef_poses = pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses_rel_obj,
        pose_A_in_B=obj_pose[None],
    )
    return transformed_eef_poses
