import chex
import jax
import jax.numpy as jnp
from typing import Any

from waymax import datatypes

def last_log_obs_from_state(state: datatypes.simulator_state.SimulatorState,
                            obs_num_steps: int = 1,
                            num_obj: int = 1,
                            ) -> datatypes.Observation:
    """Generates the last log observation in global coordinates.
    (Adapted from global_observation_from_state in waymax.observation.py)

    Args:
        state: Has shape (...,).
        obs_num_steps: Number of observation steps for trajectories and traffic
        lights state.
        num_obj: Used to tile the global observation for multiple objects.

    Returns:
        Observation with shape (..., num_obj). Note the same observation in
        global coordinates is tiled for num_obj when num_obj is larger than 1.
    """
    last_traj = datatypes.dynamic_slice(state.log_trajectory, start_index=91, slice_size=obs_num_steps, axis=-1)
    last_rg = state.roadgraph_points
    last_tls = datatypes.dynamic_slice(state.log_traffic_light, start_index=91, slice_size=obs_num_steps, axis=-1)
    metadata = state.object_metadata

    pose2d_shape = state.shape
    pose2d = datatypes.ObjectPose2D.from_center_and_yaw(
        xy=jnp.zeros(shape=pose2d_shape + (2,)),
        yaw=jnp.zeros(shape=pose2d_shape),
        valid=jnp.ones(shape=pose2d_shape, dtype=jnp.bool_),
    )

    # Agent-agnostic observation does not have SDC paths by default.
    sdc_paths_shape = state.shape + (1, 1)
    sdc_paths = datatypes.route.Paths(
        x=jnp.zeros(shape=sdc_paths_shape),
        y=jnp.zeros(shape=sdc_paths_shape),
        z=jnp.zeros(shape=sdc_paths_shape),
        ids=jnp.zeros(shape=sdc_paths_shape, dtype=jnp.int32),
        valid=jnp.zeros(shape=sdc_paths_shape, dtype=jnp.bool_),
        arc_length=jnp.zeros(shape=sdc_paths_shape),
        on_route=jnp.zeros(shape=sdc_paths_shape, dtype=jnp.bool_),
    )

    last_obs = datatypes.Observation(
        trajectory=last_traj,
        pose2d=pose2d,
        metadata=metadata,
        roadgraph_static_points=last_rg,
        traffic_lights=last_tls,
        sdc_paths=sdc_paths,
        is_ego=jnp.zeros(metadata.shape, dtype=jnp.bool_),  # Placeholder
    )

    def _tree_expand_and_repeat(tree: Any, repeats: int, axis: int) -> datatypes.PyTree:
        def _expand_and_repeat(x: jax.Array) -> jax.Array:
            return jnp.repeat(jnp.expand_dims(x, axis=axis), repeats, axis=axis)

        return jax.tree_util.tree_map(_expand_and_repeat, tree)

    num_obj = 1

    obj_dim_idx = len(state.shape)
    global_obs_expanded = _tree_expand_and_repeat(
        last_obs, num_obj, obj_dim_idx
    )
    global_obs_expanded.validate()
    chex.assert_shape(global_obs_expanded, state.shape + (num_obj,))

    return global_obs_expanded


def last_sdc_observation_for_current_sdc_from_state(
    state: datatypes.simulator_state.SimulatorState,
    obs_num_steps: int = 1,
    roadgraph_top_k: int = 1000,
    ) -> datatypes.Observation:
    """Constructs the last log Observation from SimulatorState for SDC only (jit-able).
    (Adapted from sdc_observation_from_state in waymax.observation.py)

    Args:
        state: a SimulatorState, with shape (...)
        obs_num_steps: number of steps history included in observation. Last
        timestep is state.timestep.
        roadgraph_top_k: number of topk roadgraph observed by each object.
        coordinate_frame: which coordinate frame the returned observation is using.

    Returns:
        SDC Observation at current timestep from given simulator state, with shape
        (..., 1), where the last object dimension is 1 as there is only one SDC. It
        is not sequeezed to be consistent with multi-agent cases and compatible for
        other utils fnctions.
    """
    obj_xy = state.current_sim_trajectory.xy[..., 0, :]
    obj_yaw = state.current_sim_trajectory.yaw[..., 0]
    obj_valid = state.current_sim_trajectory.valid[..., 0]

    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
    sdc_xy = jnp.take_along_axis(obj_xy, sdc_idx[..., jnp.newaxis], axis=-2)
    sdc_yaw = jnp.take_along_axis(obj_yaw, sdc_idx, axis=-1)
    sdc_valid = jnp.take_along_axis(obj_valid, sdc_idx, axis=-1)

    last_obs = last_log_obs_from_state(state, obs_num_steps)

    is_ego = state.object_metadata.is_sdc[..., jnp.newaxis, :]
    last_obs_filter = last_obs.replace(
        is_ego=is_ego,
        roadgraph_static_points=datatypes.roadgraph.filter_topk_roadgraph_points(
            last_obs.roadgraph_static_points, sdc_xy, roadgraph_top_k
        ),
    )

    pose2d = datatypes.ObjectPose2D.from_center_and_yaw(
        xy=sdc_xy, yaw=sdc_yaw, valid=sdc_valid
    )
    return datatypes.transform_observation(last_obs_filter, pose2d)