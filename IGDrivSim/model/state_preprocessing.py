from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Any, Dict

from waymax import datatypes
from waymax.datatypes import transform_trajectory

from IGWaymax.utils import last_sdc_observation_for_current_sdc_from_state

from .config import MAX_HEADING_RADIUS, XY_SCALING_FACTOR, SPEED_SCALING_FACTOR
from .model_utils import combine_two_object_pose_2d, radius_point_extra

def shuffle_features(features, valid, rng):

    rng, rng_permut = jax.random.split(rng)
    permuted_features = jax.random.permutation(rng_permut, features, axis=-3)
    permuted_valid = jax.random.permutation(rng_permut, valid, axis=-3)

    sorted_idx = jnp.argsort(1 - permuted_valid, axis=-3)

    sorted_permuted_features = jnp.take_along_axis(permuted_features, sorted_idx, axis=-3)
    return sorted_permuted_features

def extract_xy(state, obs, rng):
    """Extract the xy positions of the objects of the scene.

    Args:
        state: The current simulator state (unused).
        obs: The current observation of the SDC in its local
        referential.

    Returns:
        xy positions (trajectory of size 1) of the
        objects in the scene concatenated with validity tag.
    """
    xy = obs.trajectory.xy / XY_SCALING_FACTOR
    valid = obs.trajectory.valid[..., None]

    masked_xy = jnp.where(valid, xy, jnp.zeros_like(xy))

    output = jnp.concatenate((masked_xy, valid), axis=-1)

    valid_t0 = state.log_trajectory.valid[:, None, :, None][..., 0][..., None]
    output = shuffle_features(output, valid_t0, rng)

    return output

def extract_xyyaw(state, obs, rng):
    """Extract the xy positions and yaw of the objects of the scene.

    Args:
        state: The current simulator state (unused).
        obs: The current observation of the SDC in its local
        referential.

    Returns:
        xy positions (trajectory of size 1) of the
        objects in the scene concatenated with validity tag.
    """
    xy = obs.trajectory.xy / XY_SCALING_FACTOR
    yaw = obs.trajectory.yaw[..., None]
    cos_sin = jnp.concatenate((jnp.cos(yaw), jnp.sin(yaw)), axis=-1)
    valid = obs.trajectory.valid[..., None]

    masked_xy = jnp.where(valid, xy, jnp.zeros_like(xy))
    masked_cos_sin = jnp.where(valid, cos_sin, jnp.zeros_like(cos_sin))

    output = jnp.concatenate((masked_xy, masked_cos_sin, valid), axis=-1)

    valid_t0 = state.log_trajectory.valid[:, None, :, None][..., 0][..., None]
    output = shuffle_features(output, valid_t0, rng)

    return output

def extract_xyyawv(state, obs, rng):
    """Extract the xy positions, yaw and speed of the objects of the scene.

    Args:
        state: The current simulator state (unused).
        obs: The current observation of the SDC in its local
        referential.

    Returns:
        xy positions (trajectory of size 1) of the
        objects in the scene concatenated with validity tag.
    """
    xy = obs.trajectory.xy / XY_SCALING_FACTOR
    yaw = obs.trajectory.yaw[..., None]
    cos_sin = jnp.concatenate((jnp.cos(yaw), jnp.sin(yaw)), axis=-1)
    speed = obs.trajectory.speed[..., None] / SPEED_SCALING_FACTOR
    valid = obs.trajectory.valid[..., None]

    masked_xy = jnp.where(valid, xy, jnp.zeros_like(xy))
    masked_cos_sin = jnp.where(valid, cos_sin, jnp.zeros_like(cos_sin))
    masked_speed = jnp.where(valid, speed, jnp.zeros_like(speed))

    output = jnp.concatenate((masked_xy, masked_cos_sin, masked_speed, valid), axis=-1)

    valid_t0 = state.log_trajectory.valid[:, None, :, None][..., 0][..., None]
    output = shuffle_features(output, valid_t0, rng)

    return output

def extract_sdc_speed(state, obs, rng):
    """Extract the speed of the SDC.

    Args:
        state: The current simulator state (unused).
        obs: The current observation of the SDC in its local
        referential.

    Returns:
        Speed of the SDC.
    """
    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
    sdc_speed = jnp.take_along_axis(obs.trajectory.speed, sdc_idx[..., None, None], axis=-2) / SPEED_SCALING_FACTOR

    return sdc_speed

def extract_goal(state, obs, rng):
    """Generates the proxy goal as the last
    xy positoin of the SDC in the log trajectory.

    Args:
        state: The current simulator state.
        obs: The current observation of the SDC in its local referential (unused).

    Returns:
        Proxy goal of shape (..., 2). Note that the proxy
        goal coordinates are in the referential of the current
        SDC position.
    """
    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)

    last_sdc_obs = last_sdc_observation_for_current_sdc_from_state(state) # Last obs of the log in the current SDC pos referential
    last_sdc_xy = jnp.take_along_axis(last_sdc_obs.trajectory.xy[..., 0, :], sdc_idx[..., None, None], axis=-2)

    mask = jnp.any(state.object_metadata.is_sdc)[..., None, None] # Mask if scenario with no SDC
    proxy_goal = (last_sdc_xy * mask) / XY_SCALING_FACTOR

    return proxy_goal


def extract_noisy_goal(state, obs, rng, sigma):
    """Generates the proxy goal as the last
    xy positoin of the SDC in the log trajectory and add a Gaussian noise
    with mean 0 and std.

    Args:
        state: The current simulator state.
        obs: The current observation of the SDC in its local referential (unused).
        sigma: The standard deviation of the noise. Note that the mean is always 0.
        rng: Random key.

    Returns:
        Proxy goal of shape (..., 2). Note that the proxy
        goal coordinates are in the referential of the current
        SDC position.
    """
    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)

    last_sdc_obs = last_sdc_observation_for_current_sdc_from_state(state) # Last obs of the log in the current SDC pos referential
    last_sdc_xy = jnp.take_along_axis(last_sdc_obs.trajectory.xy[..., 0, :], sdc_idx[..., None, None], axis=-2)

    mask = jnp.any(state.object_metadata.is_sdc)[..., None, None] # Mask if scenario with no SDC
    proxy_goal = last_sdc_xy * mask

    noise = jax.random.normal(rng, proxy_goal.shape) * sigma

    noisy_proxy_goal = (noise + proxy_goal) / XY_SCALING_FACTOR

    return noisy_proxy_goal

def find_index_in_range(A, radius):
    """Return the first index i such that A[i] <= radius
    and A[i+1] >= radius.
    """
    conditions = (A[:-1] <= radius) & (A[1:] >= radius)
    index = jax.lax.cond(jnp.any(conditions),
                     lambda _: jnp.argmax(conditions),
                     lambda _: 0,
                     operand=None)
    return index

def extract_heading(state, obs, rng, radius):
    """Generates the heading for the SDC to move towards
    the log position radius meters away from its current
    position. If no such intersection exists try with a radius
    of MAX_HEADING_RADIUS and if still no intersection exists
    return (0,0).

    Args:
        state: Has shape (...,).
        radius: the considered distance in meters.

    Returns:
        Heading with shape (..., 2). Note that the heading is in
        the coordinate system of the current SDC position.
    """
    def proxy_heading(state: datatypes.simulator_state.SimulatorState,
                      radius: float
                      ) -> jnp.ndarray:
        _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)

        obj_xy = state.current_sim_trajectory.xy[..., 0, :]
        obj_yaw = state.current_sim_trajectory.yaw[..., 0]
        obj_valid = state.current_sim_trajectory.valid[..., 0]

        _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)

        current_sdc_xy = jnp.take_along_axis(obj_xy, sdc_idx[..., None], axis=-2)
        current_sdc_yaw = jnp.take_along_axis(obj_yaw, sdc_idx, axis=-1)
        current_sdc_valid = jnp.take_along_axis(obj_valid, sdc_idx, axis=-1)

        current_sdc_pose2d = datatypes.ObjectPose2D.from_center_and_yaw(
            xy=current_sdc_xy, yaw=current_sdc_yaw, valid=current_sdc_valid
        )

        traj = state.log_trajectory

        pose2d_shape = traj.shape
        # Global coordinates pose2d
        pose2d = datatypes.ObjectPose2D.from_center_and_yaw(
            xy=jnp.zeros(shape=pose2d_shape + (2,)),
            yaw=jnp.zeros(shape=pose2d_shape),
            valid=jnp.ones(shape=pose2d_shape, dtype=jnp.bool_),
        )
        pose = combine_two_object_pose_2d(
            src_pose=pose2d, dst_pose=current_sdc_pose2d
        )

        # Project log trajectory in the ref of the current SDC
        transf_traj = transform_trajectory(traj, pose)

        sdc_transf_traj_xy = jnp.take_along_axis(transf_traj.xy, sdc_idx[..., None, None], axis=0)
        sdc_transf_traj_yaw = jnp.take_along_axis(transf_traj.yaw, sdc_idx[..., None], axis=0)

        # Compute dist to the current SDC pos
        dist_matrix = jnp.linalg.norm(sdc_transf_traj_xy, axis=-1).squeeze()

        # Intersection btw the circle and log trajectory
        idx_radius_point = find_index_in_range(dist_matrix, radius)
        radius_point = jnp.take_along_axis(sdc_transf_traj_xy, idx_radius_point[None,None, None], axis=-2)

        inter_heading =  radius_point / jnp.linalg.norm(radius_point, axis=-1)

        # Intersection btw the circle and extrapolation of the log trajectory
        last_sdc_xy = sdc_transf_traj_xy[:, -1, :]
        last_sdc_yaw = sdc_transf_traj_yaw[:, -1]
        last_sdc_heading = jnp.stack((jnp.cos(last_sdc_yaw), jnp.sin(last_sdc_yaw)), axis=-1)

        extra_heading = radius_point_extra(last_sdc_xy, last_sdc_heading, jnp.zeros_like(last_sdc_xy), radius)

        current_sdc_heading = jnp.where(idx_radius_point == 0, extra_heading, inter_heading)

        # assert(jnp.any(current_sdc_heading != jnp.zeros((2,)), axis=-1))

        current_sdc_heading = jnp.where(jnp.all(current_sdc_heading == 0),
                                        current_sdc_heading,
                                        current_sdc_heading / jnp.linalg.norm(current_sdc_heading))

        return current_sdc_heading

    radius_heading = jax.vmap(proxy_heading, (0, None))(state, radius)
    max_radius_heading = jax.vmap(proxy_heading, (0, None))(state, MAX_HEADING_RADIUS)

    heading = jnp.where(jnp.all(radius_heading == 0, axis=-1)[..., None],
                        max_radius_heading,
                        radius_heading)

    return heading

def extract_roadgraph(state, obs, rng):
    """Extract the features (xy, dir_xy, type,valid) of the roadgraph
    points.

    Args:
        state: The current simulator state (unused).
        obs: The current observation of the SDC in its local
        referential.

    Returns:
        Masked roadgraph points features (trajectory of size 1).
    """
    valid = obs.roadgraph_static_points.valid[..., None]

    xy = obs.roadgraph_static_points.xy / XY_SCALING_FACTOR
    masked_xy = jnp.where(valid, xy, jnp.zeros_like(xy))

    dir = obs.roadgraph_static_points.dir_xy / XY_SCALING_FACTOR
    masked_dir = jnp.where(valid, dir, jnp.zeros_like(dir))

    type = obs.roadgraph_static_points.types
    type_one_hot = jax.nn.one_hot(type, 20)

    point_features = jnp.concatenate((masked_xy, masked_dir, type_one_hot, valid), axis=-1)

    return point_features

def extract_trafficlights(state, obs, rng):
    """Extract the features (xy positions, type) of the traffic
    lights present in the scene.

    Args:
        state: The current simulator state (unused).
        obs: The current observation of the SDC in its local
        referential.

    Returns:
        Masked traffic lights features (trajectory of size 1).
    """
    xy = obs.traffic_lights.xy / XY_SCALING_FACTOR
    valid = obs.traffic_lights.valid[..., None]

    masked_xy = jnp.where(valid, xy, jnp.zeros_like(xy))

    type = obs.traffic_lights.state
    type_one_hot = jax.nn.one_hot(type, 9)

    traffic_lights_features = jnp.concatenate((masked_xy, type_one_hot, valid), axis=-1)

    return traffic_lights_features

EXTRACTOR_DICT = {'xy': extract_xy,
                  'xyyaw': extract_xyyaw,
                  'xyyawv': extract_xyyawv,
                  'sdc_speed': extract_sdc_speed,
                  'proxy_goal': extract_goal,
                  'noisy_proxy_goal': extract_noisy_goal,
                  'heading': extract_heading,
                  'roadgraph_map': extract_roadgraph,
                  'traffic_lights': extract_trafficlights}

def init_dict(config, batch_size):
    return {'xy': jnp.zeros((1, batch_size, config['max_num_obj'], 3)),
            'xyyaw': jnp.zeros((1, batch_size, config['max_num_obj'], 5)),
            'xyyawv': jnp.zeros((1, batch_size, config['max_num_obj'], 6)),
            '[xyyawv, sdc_speed, heading]': jnp.zeros((1, batch_size, config['max_num_obj']*6 + 1 + 2)),
            '[xyyawv, sdc_speed]': jnp.zeros((1, batch_size, config['max_num_obj']*6 + 1)),
            'sdc_speed': jnp.zeros((1, batch_size, 1)),
            'proxy_goal': jnp.zeros((1, batch_size, 2)),
            'noisy_proxy_goal': jnp.zeros((1, batch_size, 2)),
            'heading': jnp.zeros((1, batch_size, 2)),
            'roadgraph_map': jnp.zeros((1, batch_size, config['roadgraph_top_k'], 25)),
            'traffic_lights': jnp.zeros((1, batch_size, 16, 12))
            }

class Extractor(ABC):

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    @abstractmethod
    def init_x(self):
        pass

@dataclass
class ExtractObs(Extractor):
    config: Dict

    def __call__(self, state, obs, rng):
        obs_features = {}

        for key in self.config['feature_extractor_kwargs']['keys']:
            if '[' in key:
                list_subkeys = [item.strip() for item in key.strip("[]").split(",")]
            else:
                list_subkeys = [key]
            for subkey in list_subkeys:
                if subkey in self.config['feature_extractor_kwargs']['kwargs']:
                    kwargs = self.config['feature_extractor_kwargs']['kwargs'][subkey]
                    obs_feature = EXTRACTOR_DICT[subkey](state, obs, rng, **kwargs)
                else:
                    obs_feature = EXTRACTOR_DICT[subkey](state, obs, rng)
                B = obs_feature.shape[0]
                obs_feature = jnp.squeeze(obs_feature)
                if B == 1:
                    obs_feature = obs_feature[jnp.newaxis, ...]
                obs_features[subkey] = obs_feature
        return obs_features

    def init_x(self, batch_size=None):
        init_obs = {}

        if batch_size is None:
            batch_size = self.config['num_envs']

        all_init_dict = init_dict(self.config, batch_size)
        for key in self.config['feature_extractor_kwargs']['keys']:
            if '[' in key:
                list_subkeys = [item.strip() for item in key.strip("[]").split(",")]
            else:
                list_subkeys = [key]
            for subkey in list_subkeys:
                init_obs[subkey] = all_init_dict[subkey]
        return  (init_obs,
                 jnp.zeros((1, batch_size), dtype=bool),
                 )