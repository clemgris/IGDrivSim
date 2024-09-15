from abc import ABC, abstractmethod
import dataclasses
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Any

def linear_clip_scale(v, v_max, min_value, max_value):
    return v.clip(0, v_max) * ((max_value - min_value) / v_max) + min_value

class ObsMask(ABC):

    @abstractmethod
    def mask_obs(self, *args: Any, **kwds: Any) -> Any:
        pass

    @abstractmethod
    def mask_fun(self, *args: Any, **kwds: Any) -> Any:
        pass

    @abstractmethod
    def plot_mask_fun(self, *args: Any, **kwds: Any) -> None:
        pass

@dataclass
class RandomMasking(ObsMask):
    """
    Applies random masking to observations based on a specified probability. This
    probabilistically invalidates certain observations.

    Attributes:
        prob (float): The probability threshold for masking observations.
        mask_per_step (bool): Boolean indicating if the random key has to be
                              initialised at every timestep. If False, the random
                              key will be initialised at every trajectory.
    """
    prob: float
    mask_per_step: bool = True

    def mask_obs(self, state, obs, rng):
        valid = obs.trajectory.valid
        visible_obj = self.mask_fun(valid, rng)
        visible_obj = visible_obj | state.object_metadata.is_sdc[:, None, ... ,None] # Cannot mask SDC

        trajectory_limited = dataclasses.replace(obs.trajectory,
                                                 valid=visible_obj & valid)

        limited_obs = dataclasses.replace(obs,
                                          trajectory=trajectory_limited)

        return limited_obs

    def mask_fun(self, valid, rng):

        mask = jax.random.uniform(rng, valid.shape) >= self.prob

        return mask

    def plot_mask_fun(self, ax, center=(0, 0), color='b') -> None:
        pass

@dataclass
class GaussianNoise(ObsMask):
    """
    Adds Gaussian noise to observations of the other cars' position and road points'
    positions as if the SDC position were corrupted (i.e. the same pertubation is applied to the
    cars' position and road points).

    Attributes:
        sigma_ (float): The standard deviation of the Gaussian noise.
        mask_per_step (bool): Boolean indicating if the random key has to be
                              initialised at every timestep. If False, the random
                              key will be initialised at every trajectory.
    """
    sigma: float
    mask_per_step: bool = False

    def mask_obs(self, state, obs, rng):
        noisy_xy, noisy_point_xy = self.mask_fun(state, obs, rng) # Same noise for the entire trajectory

        # Noise
        trajectory_limited = dataclasses.replace(obs.trajectory,
                                                 x=noisy_xy[..., 0],
                                                 y=noisy_xy[..., 1])

        roadgraph_limited = dataclasses.replace(obs.roadgraph_static_points,
                                                x=noisy_point_xy[..., 0],
                                                y=noisy_point_xy[..., 1])

        obs_limited = dataclasses.replace(obs,
                                          trajectory=trajectory_limited,
                                          roadgraph_static_points=roadgraph_limited)
        return obs_limited

    def mask_fun(self, state, obs, rng):

        B = obs.trajectory.shape[0]

        xy = obs.trajectory.xy
        point_xy = obs.roadgraph_static_points.xy

        sdc_mask = state.object_metadata.is_sdc[:, None, :, None, None].repeat(2, axis=-1)

        gaussian_noise = jax.random.normal(rng, (B, 2)) * self.sigma

        noisy_xy = xy + gaussian_noise[:, None, None, None, :].repeat(xy.shape[2], axis=2) * (1 - sdc_mask)
        noisy_point_xy = point_xy + gaussian_noise[:, None, None, :].repeat(point_xy.shape[2] ,axis=2)

        return (noisy_xy, noisy_point_xy)

    def plot_mask_fun(self, ax, center=(0, 0), color='b') -> None:
        pass

@dataclass
class DistanceObsMask(ObsMask):
    """
    Masks observations based on a specified radius. This class filters out objects
    (both cars and road points) that are beyond the defined radius from the origin.

    Attributes:
        radius (float): The radius within which objects are considered visible.
        mask_per_step (bool): Boolean indicating if the random key has to be
                              initialised at every timestep. If False, the random
                              key will be initialised at every trajectory.
    """
    radius: float
    mask_per_step: bool = True

    def mask_obs(self, state, obs, rng):

        # Mask cars
        x = obs.trajectory.x
        y = obs.trajectory.y

        visible_obj = self.mask_fun(x, y)

        trajectory_limited = dataclasses.replace(obs.trajectory,
                                                valid=visible_obj & obs.trajectory.valid)

        limited_obs = dataclasses.replace(obs,
                                          trajectory=trajectory_limited)

        # Mask road
        point_x = obs.roadgraph_static_points.x
        point_y = obs.roadgraph_static_points.y

        visible_point = self.mask_fun(point_x, point_y)

        roadgraph_limited = dataclasses.replace(obs.roadgraph_static_points,
                                                valid=visible_point & obs.roadgraph_static_points.valid)

        limited_obs = dataclasses.replace(limited_obs,
                                          roadgraph_static_points=roadgraph_limited)

        return limited_obs

    def mask_fun(self, obj_x, obj_y, eps=1e-3):
        is_center = (-eps <= obj_x) & (obj_x <= eps) & \
            (-eps <= obj_y) & (obj_y <= eps)

        squared_distance = obj_x**2 + obj_y**2

        return (squared_distance <= self.radius**2) | is_center

    def plot_mask_fun(self, ax, center=(0, 0)) -> None:
        theta = jnp.linspace(0, 2 * jnp.pi, 100)

        x = center[0] + self.radius * jnp.cos(theta)
        y = center[1] + self.radius * jnp.sin(theta)

        ax.plot(x, y)

@dataclass
class ConicObsMask(ObsMask):
    """
    Masks observations based on a specified radius and angle. This class filters out objects
    (both cars and road points) that are beyond a conic field of view defined by radius from
    the origin (SDC position) and an opening angle.

    Attributes:
        radius (float): The radius within which objects are considered visible.
        angle (float): The opening angle of the conic field of view.
        mask_per_step (bool): Boolean indicating if the random key has to be
                              initialised at every timestep. If False, the random
                              key will be initialised at every trajectory.
    """
    radius: float
    angle: float
    mask_per_step: bool = True

    def mask_obs(self, state, obs, rng):

        # Mask cars
        x = obs.trajectory.x
        y = obs.trajectory.y

        visible_obj = self.mask_fun(x, y)

        trajectory_limited = dataclasses.replace(obs.trajectory,
                                                 valid=visible_obj & obs.trajectory.valid)

        limited_obs = dataclasses.replace(obs,
                                          trajectory=trajectory_limited)

        # Mask road
        point_x = obs.roadgraph_static_points.x
        point_y = obs.roadgraph_static_points.y

        visible_point = self.mask_fun(point_x, point_y)

        roadgraph_limited = dataclasses.replace(obs.roadgraph_static_points,
                                                valid=visible_point & obs.roadgraph_static_points.valid)

        limited_obs = dataclasses.replace(limited_obs,
                                          roadgraph_static_points=roadgraph_limited)

        # Mask road
        point_x = obs.roadgraph_static_points.x[..., None]
        point_y = obs.roadgraph_static_points.y[..., None]

        visible_point = self.mask_fun(point_x, point_y).squeeze(-1)

        roadgraph_limited = dataclasses.replace(obs.roadgraph_static_points,
                                                valid=visible_point & obs.roadgraph_static_points.valid)

        limited_obs = dataclasses.replace(limited_obs,
                                          roadgraph_static_points=roadgraph_limited)

        return limited_obs

    def mask_fun(self, obj_x, obj_y, eps=1e-3):

        angle = jnp.array([self.angle])

        is_center = (-eps <= obj_x) & (obj_x <= eps) & \
            (-eps <= obj_y) & (obj_y <= eps)

        squared_distance = obj_x**2 + obj_y**2

        obj_angle = jnp.arctan2(obj_y, obj_x)
        angle_condition = (- angle[:, None, None] / 2 <= obj_angle) &\
            (obj_angle <= angle[:, None, None] / 2)

        radius_condition = squared_distance <= self.radius**2

        return (angle_condition & radius_condition)| is_center

    def plot_mask_fun(self, ax, center=(0, 0), color='b') -> None:
        theta = jnp.linspace(- self.angle/2, self.angle/2, 100)

        x = center[0] + self.radius * jnp.cos(theta)
        y = center[1] + self.radius * jnp.sin(theta)

        x1 = center[0] + self.radius * jnp.cos(- self.angle/2)
        y1 = center[1] + self.radius * jnp.sin(- self.angle/2)

        x2 = center[0] + self.radius * jnp.cos(self.angle/2)
        y2 = center[1] + self.radius * jnp.sin(self.angle/2)

        ax.plot([0, x1], [0, y1], c=color)
        ax.plot([0, x2], [0, y2], c=color)
        ax.plot(x, y, c=color)


@dataclass
class BlindSpotObsMask(ObsMask):
    """
    Masks observations based on a specified radius and angle to create a blind spot effect.
    This class filters out objects (both cars and road points) that are beyond the defined
    radius and within the blind spot angle from the origin.

    Attributes:
        radius (float): The radius within which objects are considered visible.
        angle (float): The angle defining the blind spot area.
        mask_per_step (bool): Boolean indicating if the random key has to be
                              initialised at every timestep. If False, the random
                              key will be initialised at every trajectory.
    """
    radius: float
    angle: float
    mask_per_step: bool = True

    def mask_obs(self, state, obs, rng):

        # Mask cars
        x = obs.trajectory.x
        y = obs.trajectory.y

        visible_obj = self.mask_fun(x, y)

        trajectory_limited = dataclasses.replace(obs.trajectory,
                                                valid=visible_obj & obs.trajectory.valid)

        limited_obs = dataclasses.replace(obs,
                                          trajectory=trajectory_limited)

        # Mask road
        point_x = obs.roadgraph_static_points.x
        point_y = obs.roadgraph_static_points.y

        visible_point = self.mask_fun(point_x, point_y)

        roadgraph_limited = dataclasses.replace(obs.roadgraph_static_points,
                                                valid=visible_point & obs.roadgraph_static_points.valid)

        limited_obs = dataclasses.replace(limited_obs,
                                          roadgraph_static_points=roadgraph_limited)

        return limited_obs

    def mask_fun(self, obj_x, obj_y, eps=1e-3):
        assert(self.angle <= jnp.pi / 2)
        is_center = (-eps <= obj_x) & (obj_x <= eps) & \
            (-eps <= obj_y) & (obj_y <= eps)

        visible_angle = jnp.pi - 2 * self.angle
        squared_distance = obj_x**2 + obj_y**2

        obj_angle = jnp.arctan2(obj_y, obj_x)
        angle_condition_front = (- jnp.pi / 2 <= obj_angle) & (obj_angle <= jnp.pi / 2)
        angle_condition_back = (- (visible_angle / 2 + 2 * self.angle) >= obj_angle) | (obj_angle >= (visible_angle / 2 + 2 * self.angle))
        radius_condition = squared_distance <= self.radius**2

        return ((angle_condition_front | angle_condition_back) & radius_condition) | is_center

    def plot_mask_fun(self, ax, center=(0, 0), color='b') -> None:

        # Front
        theta = jnp.linspace(- jnp.pi / 2, jnp.pi / 2, 100)

        x = center[0] + self.radius * jnp.cos(theta)
        y = center[1] + self.radius * jnp.sin(theta)

        x1 = center[0] + self.radius * jnp.cos(- jnp.pi / 2)
        y1 = center[1] + self.radius * jnp.sin(- jnp.pi / 2)

        x2 = center[0] + self.radius * jnp.cos(jnp.pi / 2)
        y2 = center[1] + self.radius * jnp.sin(jnp.pi / 2)

        ax.plot([center[0], x1], [center[1], y1], c=color)
        ax.plot([center[0], x2], [center[1], y2], c=color)
        ax.plot(x, y, c=color)

        # Back
        visible_angle = jnp.pi - 2 * self.angle
        theta = jnp.linspace(visible_angle / 2, - visible_angle / 2, 100)

        x = center[0] - self.radius * jnp.cos(theta)
        y = center[1] - self.radius * jnp.sin(theta)

        x1 = center[0] + self.radius * jnp.cos(- (visible_angle / 2 + 2 * self.angle))
        y1 = center[1] + self.radius * jnp.sin(- (visible_angle / 2 + 2 * self.angle))

        x2 = center[0] + self.radius * jnp.cos(visible_angle / 2 + 2 * self.angle)
        y2 = center[1] + self.radius * jnp.sin(visible_angle / 2 + 2 * self.angle)

        ax.plot([center[0], x1], [center[1], y1], c=color)
        ax.plot([center[0], x2], [center[1], y2], c=color)
        ax.plot(x, y, c=color)

@dataclass
class SpeedConicObsMask(ObsMask):
    """
    Masks observations based on a conical region defined by speed, radius, and angles.
    This class dynamically adjusts the visible region based on the speed of a specific
    vehicle (SDC) to create a conical mask that varies with speed.

    Attributes:
        radius (float): The maximum distance within which objects are considered visible.
        angle_max (float): The maximum angle of the conical mask when the vehicle is at rest.
        angle_min (float): The minimum angle of the conical mask at the maximum speed.
        v_max (float): The maximum speed of the vehicle to calculate the dynamic angle.
        mask_per_step (bool): Boolean indicating if the random key has to be
                              initialised at every timestep. If False, the random
                              key will be initialised at every trajectory.
    """
    radius: float
    angle_max: float
    angle_min: float
    v_max: float
    mask_per_step: bool = True

    def mask_obs(self, state, obs, rng):
        _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
        sdc_v = jnp.take_along_axis(obs.trajectory.speed, sdc_idx[..., None, None], axis=-2)

        # Mask cars
        x = obs.trajectory.x
        y = obs.trajectory.y

        visible_obj = self.mask_fun(x, y, sdc_v.squeeze())

        trajectory_limited = dataclasses.replace(obs.trajectory,
                                                 valid=visible_obj & obs.trajectory.valid)
        limited_obs = dataclasses.replace(obs,
                                          trajectory=trajectory_limited)
        # Mask road
        point_x = obs.roadgraph_static_points.x[..., None]
        point_y = obs.roadgraph_static_points.y[..., None]

        visible_point = self.mask_fun(point_x, point_y, sdc_v.squeeze()).squeeze(-1)

        roadgraph_limited = dataclasses.replace(obs.roadgraph_static_points,
                                                valid=visible_point & obs.roadgraph_static_points.valid)

        limited_obs = dataclasses.replace(limited_obs,
                                          roadgraph_static_points=roadgraph_limited)

        return limited_obs

    def mask_fun(self, obj_x, obj_y, sdc_v, eps=1e-3):

        angle = (sdc_v.clip(0, self.v_max) * (self.angle_min - self.angle_max) / self.v_max + self.angle_max)[..., None]

        return ConicObsMask(self.radius, angle).mask_fun(obj_x, obj_y, eps=eps)

    def plot_mask_fun(self, ax, sdc_v, center=(0, 0), color='b') -> None:
        angle = sdc_v.clip(0, self.v_max) * (self.angle_min - self.angle_max) / self.v_max + self.angle_max

        ConicObsMask(self.radius, angle.squeeze()).plot_mask_fun(ax, center=center, color=color)

@dataclass
class SpeedGaussianNoise(ObsMask):
    """
    Adds Gaussian noise to observations based on the speed of a specific vehicle (SDC).
    The noise level varies with speed, introducing more noise as the vehicle speed
    increases up to a maximum threshold.

    Attributes:
        v_max (float): The maximum speed of the vehicle to calculate the dynamic noise level.
        sigma_max (float): The maximum standard deviation of the Gaussian noise at the maximum speed.
        sigma_min (float): The minimum standard deviation of the Gaussian noise at zero speed.
        mask_per_step (bool): Boolean indicating if the random key has to be
                              initialised at every timestep. If False, the random
                              key will be initialised at every trajectory.
    """
    v_max: float
    sigma_max: float
    sigma_min: float
    mask_per_step: bool = True

    def mask_obs(self, state, obs, rng):
        noisy_xy = self.mask_fun(state, obs, rng)

        trajectory_limited = dataclasses.replace(obs.trajectory,
                                                 x=noisy_xy[..., 0],
                                                 y=noisy_xy[..., 1])
        obs_limited = dataclasses.replace(obs,
                                          trajectory=trajectory_limited)
        return obs_limited

    def mask_fun(self, state, obs, rng):

        xy = obs.trajectory.xy

        _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
        sdc_v = jnp.take_along_axis(obs.trajectory.speed, sdc_idx[..., None, None], axis=-2)

        xy = obs.trajectory.xy
        is_obj = 1 - state.object_metadata.is_sdc
        sigma = jnp.where(is_obj[:, None, ..., None, None],
                          linear_clip_scale(sdc_v, self.v_max, self.sigma_min, self.sigma_max)[..., None] * jnp.ones_like(xy),
                          jnp.zeros_like(xy))

        gaussian_noise = jax.random.normal(rng, xy.shape) * sigma

        noisy_xy = xy + gaussian_noise

        return noisy_xy

    def plot_mask_fun(self, ax, center=(0, 0), color='b') -> None:
        pass


@dataclass
class SpeedUniformNoise(ObsMask):
    """
    Adds uniform noise to observations based on the speed of a specific vehicle (SDC).
    The noise range varies with speed, introducing wider noise bounds as the vehicle speed
    increases up to a maximum threshold.

    Attributes:
        v_max (float): The maximum speed of the vehicle to calculate the dynamic noise bounds.
        bound_max (float): The maximum bound for the uniform noise at the maximum speed.
        bound_min (float): The minimum bound for the uniform noise at zero speed.
        mask_per_step (bool): Boolean indicating if the random key has to be
                              initialised at every timestep. If False, the random
                              key will be initialised at every trajectory.
    """
    v_max: float
    bound_max: float
    bound_min: float
    mask_per_step: bool = True

    def mask_obs(self, state, obs, rng):
        noisy_xy = self.mask_fun(state, obs, rng)

        trajectory_limited = dataclasses.replace(obs.trajectory,
                                                 x=noisy_xy[..., 0],
                                                 y=noisy_xy[..., 1])
        obs_limited = dataclasses.replace(obs,
                                          trajectory=trajectory_limited)
        return obs_limited

    def mask_fun(self, state, obs, rng):

        _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
        sdc_v = jnp.take_along_axis(obs.trajectory.speed, sdc_idx[..., None, None], axis=-2)

        xy = obs.trajectory.xy
        is_obj = 1 - state.object_metadata.is_sdc
        bound = jnp.where(is_obj[:, None, ..., None, None],
                          linear_clip_scale(sdc_v, self.v_max, self.bound_min, self.bound_max)[..., None] * jnp.ones_like(xy),
                          jnp.zeros_like(xy))

        uniform_noise = jax.random.uniform(rng,
                                          minval=-bound,
                                          maxval=bound,
                                          shape=xy.shape)

        noisy_xy = xy + uniform_noise

        return noisy_xy

    def plot_mask_fun(self, ax, center=(0, 0), color='b') -> None:
        pass

##  DEBUGGING

@dataclass
class ZeroMask(ObsMask):
    """
    Masks observations by setting their coordinates to zero. This effectively removes
    all positional information from the observations.
    """

    def mask_obs(self, state, obs, rng):
        zero_xy = self.mask_fun(state, obs, rng)

        trajectory_limited = dataclasses.replace(obs.trajectory,
                                                 x=zero_xy[..., 0],
                                                 y=zero_xy[..., 1])
        obs_limited = dataclasses.replace(obs,
                                          trajectory=trajectory_limited)
        return obs_limited

    def mask_fun(self, state, obs, rng):

        xy = obs.trajectory.xy

        return jnp.zeros_like(xy)

    def plot_mask_fun(self, ax, center=(0, 0), color='b') -> None:
        pass