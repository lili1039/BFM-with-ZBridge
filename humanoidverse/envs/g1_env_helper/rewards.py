import abc
import dataclasses
import re
from typing import Optional

import mujoco
import numpy as np
from dm_control.utils import rewards

COORD_TO_INDEX = {"x": 0, "y": 1, "z": 2}
ALIGNMENT_BOUNDS = {"x": (-0.1, 0.1), "z": (0.9, float("inf")), "y": (-0.1, 0.1)}


def add_visual_arrow(renderer, point1, point2, rgba):
    """Adds an arrow to an mjvScene."""
    if renderer is None:
        return
    if not isinstance(rgba, np.ndarray):
        rgba = np.array(rgba).astype(np.float32)
    if not isinstance(point1, np.ndarray):
        point1 = np.array(point1).astype(np.float32)
    if not isinstance(point2, np.ndarray):
        point2 = np.array(point2).astype(np.float32)
    scene = renderer.scene
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.02,
        point1,
        point2,
    )


def add_arrow_from_xpos_to_direction(renderer, xpos, vector, rgba):
    point1 = xpos
    point2 = xpos + vector
    add_visual_arrow(renderer, point1, point2, rgba)


def rot2eul(R: np.ndarray):
    beta = -np.arcsin(R[2, 0])
    alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
    gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
    return np.array((alpha, beta, gamma))


def get_xpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert index > -1
    xpos = data.xpos[index].copy()
    return xpos


def get_xmat(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert index > -1
    xmat = data.xmat[index].reshape((3, 3)).copy()
    return xmat


def get_center_of_mass_linvel(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    chest_subtree_linvel_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_link_subtreelinvel")  # in global coordinate
    start = model.sensor_adr[chest_subtree_linvel_index]
    end = start + model.sensor_dim[chest_subtree_linvel_index]
    center_of_mass_velocity = data.sensordata[start:end].copy()
    return center_of_mass_velocity


def get_sensor_data(model: mujoco.MjModel, data: mujoco.MjData, name: str):
    chest_gyro_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)  # in global coordinate
    assert chest_gyro_index > -1
    start = model.sensor_adr[chest_gyro_index]
    end = start + model.sensor_dim[chest_gyro_index]
    sensord = data.sensordata[start:end].copy()
    return sensord


class RewardFunction(abc.ABC):
    @abc.abstractmethod
    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float: ...

    @staticmethod
    @abc.abstractmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]: ...

    def __call__(
        self,
        model: mujoco.MjModel,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
    ):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = ctrl
        mujoco.mj_forward(model, data)
        return self.compute(model, data)


@dataclasses.dataclass
class ZeroReward(RewardFunction):
    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        return 0.0

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        if name.lower() in ["none", "zero", "rewardfree"]:
            return ZeroReward()
        return None


@dataclasses.dataclass
class LocomotionReward(RewardFunction):
    move_speed: float = 5
    # Head height of G1 robot after "Default" reset is 1.22
    stand_height: float = 0.5
    move_angle: float = 0
    egocentric_target: bool = True
    stay_low: bool = False

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        root_height = get_xpos(model, data, "pelvis")[-1]
        # head_height = data.geom("head").xpos[-1]
        # torso_upright = get_torso_upright(model, data)
        # base_quat = data.qpos[3:7].copy().reshape(1, -1)
        # v = np.array([[0, 0, -1]])
        # gravity = quat_rotate_inverse_numpy(base_quat, v).ravel()
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        if self.move_angle is not None:
            move_angle = np.deg2rad(self.move_angle)
        if self.egocentric_target:
            pelvis_xmat = get_xmat(model, data, name="pelvis")
            euler = rot2eul(pelvis_xmat)
            move_angle = move_angle + euler[-1]

        if self.stay_low:
            standing = rewards.tolerance(
                root_height,
                bounds=(self.stand_height*0.95, self.stand_height*1.05),
                margin=self.stand_height / 2,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        else:
            standing = rewards.tolerance(
                root_height,
                bounds=(self.stand_height, float("inf")),
                margin=self.stand_height,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        # upright = rewards.tolerance(
        #             torso_upright,
        #             bounds=(-0.1, 0.1),
        #             margin=0.8,
        #             value_at_margin=0,
        #             sigmoid="linear",
        #         )
        # gravity_upright = rewards.tolerance(
        #     -gravity[-1],
        #     bounds=(0.9, float("inf")),
        #     sigmoid="linear",
        #     margin=1.9,
        #     value_at_margin=0,
        # )
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(
            np.sum(np.square(upvector_torso - np.array([0.073, 0.0, 1.0]))),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * cost_orientation
        # small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        # small_control = (4 + small_control) / 5
        small_control = 1.0
        if 0<= self.move_speed <= 0.01:
            horizontal_velocity = center_of_mass_velocity[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=0.2).mean()
            angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
            dont_rotate = rewards.tolerance(angular_velocity, margin=0.1).mean()
            return small_control * stand_reward * dont_move * dont_rotate
        else:
            vel = center_of_mass_velocity[[0, 1]]
            com_velocity = np.linalg.norm(vel)
            move = rewards.tolerance(
                com_velocity,
                bounds=(
                    self.move_speed - 0.1 * self.move_speed,
                    self.move_speed + 0.1 * self.move_speed,
                ),
                margin=self.move_speed / 2,
                value_at_margin=0.5,
                sigmoid="gaussian",
            )
            move = (5 * move + 1) / 6
            # move in a specific direction
            if np.isclose(com_velocity, 0.0) or move_angle is None:
                angle_reward = 1.0
            else:
                direction = vel / (com_velocity + 1e-6)
                target_direction = np.array([np.cos(move_angle), np.sin(move_angle)])
                dot = target_direction.dot(direction)
                angle_reward = (dot + 1.0) / 2.0
            reward = small_control * stand_reward * move * angle_reward
            return reward

    def render(self, renderer, model, data):
        if renderer is None:
            return
        pelvis_xpos = get_xpos(model, data, "pelvis")
        com_velocity = get_center_of_mass_linvel(model, data)
        com_velocity[2] = 0  # ignore vertical velocity

        move_angle = np.deg2rad(self.move_angle)
        if self.egocentric_target:
            pelvis_xmat = get_xmat(model, data, name="pelvis")
            euler = rot2eul(pelvis_xmat)
            move_angle = move_angle + euler[-1]
        target_direction = np.array([np.cos(move_angle), np.sin(move_angle), 0])
        target_direction = target_direction * self.move_speed

        # Visualize center of mass velocity
        add_arrow_from_xpos_to_direction(renderer, pelvis_xpos, com_velocity, (0, 1, 0, 1))

        # Visualize target direction
        add_arrow_from_xpos_to_direction(renderer, pelvis_xpos, target_direction, (1, 0, 0, 1))

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^move-ego-(-?\d+\.*\d*)-(-?\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            move_angle, move_speed = float(match.group(1)), float(match.group(2))
            return LocomotionReward(move_angle=move_angle, move_speed=move_speed)
        pattern = r"^move-ego-low(-?\d+\.*\d*)-(-?\d+\.*\d*)-(-?\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            stand_height, move_angle, move_speed = float(match.group(1)), float(match.group(2)), float(match.group(3))
            return LocomotionReward(move_angle=move_angle, move_speed=move_speed, stay_low=True, stand_height=stand_height)
        return None


@dataclasses.dataclass
class RotationReward(RewardFunction):
    axis: str = "x"
    target_ang_velocity: float = 5.0
    # Note: pelvis height is 0.8 exactly after reset with default pose
    stand_pelvis_height: float = 0.8

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        pelvis_height = get_xpos(model, data, name="pelvis")[-1]
        pelvis_xmat = get_xmat(model, data, name="pelvis")
        torso_rotation = pelvis_xmat[2, :].ravel()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")

        height_reward = rewards.tolerance(
            pelvis_height,
            bounds=(self.stand_pelvis_height, float("inf")),
            margin=self.stand_pelvis_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        direction = np.sign(self.target_ang_velocity)

        small_control = rewards.tolerance(data.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean()
        small_control = (4 + small_control) / 5
        small_control = 1

        targ_av = np.abs(self.target_ang_velocity)
        move = rewards.tolerance(
            direction * angular_velocity[COORD_TO_INDEX[self.axis]],
            bounds=(targ_av, targ_av + 5),
            margin=targ_av / 2,
            value_at_margin=0,
            sigmoid="linear",
        )

        aligned = rewards.tolerance(
            torso_rotation[COORD_TO_INDEX[self.axis]],
            bounds=ALIGNMENT_BOUNDS[self.axis],
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )

        reward = move * height_reward * small_control * aligned
        return reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^rotate-(x|y|z)-(-?\d+\.*\d*)-(\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            axis, target_ang_velocity, stand_pelvis_height = (
                match.group(1),
                float(match.group(2)),
                float(match.group(3)),
            )
            return RotationReward(
                axis=axis,
                target_ang_velocity=target_ang_velocity,
                stand_pelvis_height=stand_pelvis_height,
            )
        return None
