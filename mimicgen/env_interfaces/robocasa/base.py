"""
Base class for robosuite environments used by mimicgen. This includes a few
specific implementations for functions required by the general mimicgen base
env class, and a specific metaclass that registers environments both
into the mimicgen registry and into the robosuite registry.
"""
import six
import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.environments.base import register_env, EnvMeta
from robocasa.environments.kitchen.kitchen import KitchenEnvMeta

from mimicgen.envs.base import MG_Env, register_env_class, MG_EnvMeta


class RobosuiteObjectType:
    """
    Enum for object types in robosuite / mujoco.
    """
    BODY = 1
    GEOM = 2
    SITE = 3


class RobosuiteObject:
    """
    Simple interface for objects in robosuite / mujoco.
    """
    def __init__(self, obj_name, obj_type, obj_joint=None):
        self.name = obj_name
        self.type = obj_type
        self.joint = obj_joint
        assert self.type in [RobosuiteObjectType.BODY, RobosuiteObjectType.GEOM, RobosuiteObjectType.SITE], "invalid object type"

    def get_id(self, sim):
        """get internal mujoco id"""
        if self.type == RobosuiteObjectType.BODY:
            return sim.model.body_name2id(self.name)
        if self.type == RobosuiteObjectType.GEOM:
            return sim.model.geom_name2id(self.name)
        if self.type == RobosuiteObjectType.SITE:
            return sim.model.site_name2id(self.name)

    def get_pos(self, sim):
        """3-dim position"""
        obj_id = self.get_id(sim)
        if self.type == RobosuiteObjectType.BODY:
            return np.array(sim.data.body_xpos[obj_id])
        if self.type == RobosuiteObjectType.GEOM:
            return np.array(sim.data.geom_xpos[obj_id])
        if self.type == RobosuiteObjectType.SITE:
            return np.array(sim.data.site_xpos[obj_id])

    def get_rot(self, sim):
        """3x3 rotation matrix"""
        obj_id = self.get_id(sim)
        if self.type == RobosuiteObjectType.BODY:
            return np.array(sim.data.body_xmat[obj_id].reshape(3, 3))
        if self.type == RobosuiteObjectType.GEOM:
            return np.array(sim.data.geom_xmat[obj_id].reshape(3, 3))
        if self.type == RobosuiteObjectType.SITE:
            return np.array(sim.data.site_xmat[obj_id].reshape(3, 3))

    def set_pose(self, sim, pos, rot):
        """Set pose of object"""

        # convert rotation to quat in wxyz
        quat = T.convert_quat(T.mat2quat(rot), to="wxyz")

        assert self.type == RobosuiteObjectType.BODY
        if self.joint is None:
            # object is fixture - set pose with model
            obj_id = self.get_id(sim)
            sim.model.body_pos[obj_id] = np.array(pos)
            sim.model.body_xmat[obj_id] = quat
        else:
            sim.data.set_joint_qpos(self.joint, np.concatenate([pos, quat]))
        sim.forward()


# NOTE: we subclass both robosuite meta and MG meta here because MG_Robocasa_Env
#       is a subclass of MG_Env, whose metaclass is MG meta, while downstream
#       robosuite environments will be a subclass of the robosuite base class,
#       whose meta is the robosuite meta. For class inheritance to work out,
#       this metaclass must be a subclass of both metaclasses. Basically,
#       if we have class A and class B and class C which inherits from A and B,
#       then the metaclass for class C must be a subclass of the metaclass for
#       both A and B.
class MG_Robocasa_EnvMeta(MG_EnvMeta, KitchenEnvMeta):
    """
    Register environment class in robosuite registry and our own global registry.
    """
    def __new__(meta, name, bases, class_dict):
        # will register env in robosuite registry
        cls = KitchenEnvMeta.__new__(meta, name, bases, class_dict)
        # now register in our registry
        register_env_class(cls)
        return cls


@six.add_metaclass(MG_Robocasa_EnvMeta)
class MG_Robocasa_Env(MG_Env):
    """
    Env interface that robosuite mimicgen environments should conform to.
    """
    def get_controller_robot_pose(self):
        """
        Helper function to get controller robot pose from current state of environment.
        """
        eef_site_name = self.robots[0].controller["right"].ref_name
        curr_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(eef_site_name)])
        curr_rot = np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(eef_site_name)].reshape([3, 3]))
        return curr_pos, curr_rot
    
    def get_controller_mount_pose(self):        
        mount_pos = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id("robot0_link0")])
        root_body_name = self.robots[0].robot_model.root_body
        mount_rot = np.array(self.sim.data.body_xmat[self.sim.model.body_name2id(root_body_name)].reshape([3, 3]))
        mount_rot = np.matmul(
            np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
            mount_rot
        )
        return mount_pos, mount_rot
    
    def get_controller_base_pose(self):        
        # base_pos = np.array(self.sim.data.geom_xpos[self.sim.model.geom_name2id("mount0_pedestal_feet_col")])
        # root_body_name = self.robots[0].robot_model.root_body
        # base_rot = np.array(self.sim.data.body_xmat[self.sim.model.body_name2id(root_body_name)].reshape([3, 3]))
        # base_rot = np.matmul(
        #     np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        #     base_rot
        # )
        base_pos, base_rot = self.robots[0].composite_controller.get_controller_base_pose("right")
        return base_pos, base_rot

    @staticmethod
    def poses_to_action(
            start_pos, target_pos,
            start_rot=None, target_rot=None,
            base_pos=None, base_rot=None,
            max_dpos=None, max_drot=None
        ):
        """
        Takes a starting eef pose and target controller pose and returns a normalized action that 
        corresponds to the desired controller target.

        NOTE: assumes robosuite v1.2, for the convention used to translate a delta rotation action into absolute
              rotation target
        """
        
        delta_position = target_pos - start_pos
        delta_position = np.clip(delta_position / max_dpos, -1., 1.)

        if target_rot is None:
            return delta_position

        # version check for robosuite - must be v1.2, so that we're using the correct controller convention
        assert (robosuite.__version__.split(".")[0] == "1")
        assert (robosuite.__version__.split(".")[1] in ["2", "3", "4", "5"])

        # use the OSC controller's convention for delta rotation
        delta_rot_mat = target_rot.dot(start_rot.T)
        delta_quat = T.mat2quat(delta_rot_mat)
        delta_rotation = T.quat2axisangle(delta_quat)
        delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)

        # convert deltas relative to base rotation
        if base_rot is not None:
            # TOOD: convert base_rot to base_angle
            base_angle = T.quat2axisangle(T.mat2quat(base_rot))[2]
            x_w = delta_position[0]
            y_w = delta_position[1]
            x_r = np.cos(base_angle) * x_w + np.sin(base_angle) * y_w
            y_r = -np.sin(base_angle) * x_w + np.cos(base_angle) * y_w
            delta_position[0] = x_r
            delta_position[1] = y_r

            roll_w = delta_rotation[0]
            pitch_w = delta_rotation[1]
            roll_r = np.cos(base_angle) * roll_w + np.sin(base_angle) * pitch_w
            pitch_r = -np.sin(base_angle) * roll_w + np.cos(base_angle) * pitch_w
            delta_rotation[0] = roll_r
            delta_rotation[1] = pitch_r

        return np.concatenate([delta_position, delta_rotation])

    def pose_target_to_action(self, target_pos, target_rot=None):
        """
        Take a controller target pose and return a normalized action (usually a normalized
        delta pose action) that corresponds to setting the controller target pose to the
        input value. To compute this, the current eef pose will be read from the env.

        Args:
            target_pos (np.array): target position
            target_rot (np.array): target rotation

        Returns:
            (np.array) pose action
        """
        max_dpos = self.robots[0].controller["right"].output_max[0]
        max_drot = None if target_rot is None else self.robots[0].controller["right"].output_max[3]

        curr_pos, curr_rot = self.get_controller_robot_pose()
        curr_base_pos, curr_base_rot = self.get_controller_base_pose()
        action_pos = MG_Robocasa_Env.poses_to_action(
            base_pos=curr_base_pos,
            base_rot=curr_base_rot,
            start_pos=curr_pos,
            target_pos=target_pos,
            start_rot=None if target_rot is None else curr_rot,
            target_rot=target_rot,
            max_dpos=max_dpos,
            max_drot=max_drot
        )
        return action_pos

    def action_to_pose_target(self, action):
        """
        Convert env action (and current robot eef pose) to a target controller robot eef pose. Inverse
        of @pose_target_to_action. Usually used to back out a sequence of target controller poses
        from a demonstration trajectory.

        Args:
            action (np.array): environment action
        """

        # version check for robosuite - must be v1.2, so that we're using the correct controller convention
        assert (robosuite.__version__.split(".")[0] == "1")
        assert (robosuite.__version__.split(".")[1] in ["2", "3", "4", "5"])

        # unscale actions
        max_dpos = self.robots[0].controller["right"].output_max[0]
        max_drot = self.robots[0].controller["right"].output_max[3]
        delta_position = action[:3] * max_dpos
        delta_rotation = action[3:6] * max_drot

        # get reference
        curr_pos, curr_rot = self.get_controller_robot_pose()

        # get pose target
        target_pos = curr_pos + delta_position
        delta_quat = T.axisangle2quat(delta_rotation)
        delta_rot_mat = T.quat2mat(delta_quat)
        target_rot = delta_rot_mat.dot(curr_rot)

        return target_pos, target_rot
    
    def get_datagen_info(self, action=None):
        datagen_info = super(MG_Robocasa_Env, self).get_datagen_info(action=action)

        base_pos, base_rot = self.get_controller_base_pose()
        datagen_info["base_pos"] = base_pos
        datagen_info["base_rot"] = base_rot

        mount_pos, mount_rot = self.get_controller_mount_pose()
        datagen_info["mount_pos"] = mount_pos
        datagen_info["mount_rot"] = mount_rot

        return datagen_info
    
    def get_object_info(self):
        """
        Get all information about objects in the scene (usually just poses) as
        a dictionary.
        """
        objects = self._get_objects()
        info = dict()
        for name in objects:
            info["obj_{}_pos".format(name)] = objects[name].get_pos(self.sim)
            info["obj_{}_rot".format(name)] = objects[name].get_rot(self.sim)

        return info
