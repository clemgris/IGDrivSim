from typing import Any, Optional, Callable

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

from waymax import config as waymax_config
from waymax import datatypes
from waymax.visualization import utils
from waymax.utils import geometry
from waymax.visualization import color
from waymax.visualization import plot_roadgraph_points, plot_traffic_light_signals_as_points

def plot_observation_with_mask(
    obs: datatypes.Observation,
    obj_idx: int,
    viz_config: Optional[dict[str, Any]] = None,
    batch_idx: int = -1,
    highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC,
    mask_function: Optional[Callable[[matplotlib.axes.Axes], None]] = None
) -> np.ndarray:
  """Plots np array image for an Observation.

  Args:
    obs: An Observation instance, with shape (..., obs_A), where obs_A
      represents the number of objects that have observation view over things
      including other objects, roadgraph, and traffic lights.
    obj_idx: The object index in obs_A.
    viz_config: Dict for optional config.
    batch_idx: Optional batch index.
    highlight_obj: Represents the type of objects that will be highlighted with
      `color.COLOR_DICT['controlled']` color.

  Returns:
    np image.
  """
  if batch_idx > -1:
    if len(obs.shape) != 2:
      raise ValueError(f'Expecting ndim 2 for obs, got {len(obs.shape)}')
    obs = jax.tree_util.tree_map(lambda x: x[batch_idx], obs)

  # Shape: (obs_A,) -> ()
  obs = jax.tree_map(lambda x: x[obj_idx], obs)
  if obs.shape:
    raise ValueError(f'Expecting shape () for obs, got {obs.shape}')

  viz_config = (
      utils.VizConfig() if viz_config is None else utils.VizConfig(**viz_config)
  )
  fig, ax = utils.init_fig_ax(viz_config)

  # 1. Plots trajectory.
  # Shape: (num_objects, num_timesteps).
  traj = obs.trajectory
  # The current timestep index of observation is the last one in time dimension
  # as observation is toward the past.
  timestep = traj.num_timesteps - 1
  indices = np.arange(traj.num_objects) if viz_config.show_agent_id else None

  is_controlled = datatypes.get_control_mask(obs.metadata, highlight_obj)
  plot_trajectory(ax, traj, is_controlled, time_idx=timestep, indices=indices)  # pytype: disable=wrong-arg-types  # jax-ndarray

  # 2. Plots road graph elements.
  # Shape: (num_points,)
  plot_roadgraph_points(ax, obs.roadgraph_static_points, verbose=False)

  # Shape: (num_traffic_lights, num_timesteps).
  plot_traffic_light_signals_as_points(
      ax, obs.traffic_lights, timestep, verbose=False
  )

  # Plot the mask
  mask_function(ax)

  # 3. Gets np img, centered on selected agent's current location.
  # Shape (num_objects, 2).
  current_xy = traj.xy[:, timestep, :]
  if viz_config.center_agent_idx == -1:
    xy = current_xy[obs.metadata.is_sdc]
  else:
    xy = current_xy[viz_config.center_agent_idx]
  origin_x, origin_y = xy[0, :2]
  ax.axis((
      origin_x - viz_config.back_x,
      origin_x + viz_config.front_x,
      origin_y - viz_config.back_y,
      origin_y + viz_config.front_y,
  ))

  return utils.img_from_fig(fig)


def plot_observation_with_goal(
    obs: datatypes.Observation,
    obj_idx: int,
    goal: tuple,
    viz_config: Optional[dict[str, Any]] = None,
    batch_idx: int = -1,
    highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC
) -> np.ndarray:
  """Plots np array image for an Observation.

  Args:
    obs: An Observation instance, with shape (..., obs_A), where obs_A
      represents the number of objects that have observation view over things
      including other objects, roadgraph, and traffic lights.
    obj_idx: The object index in obs_A.
    viz_config: Dict for optional config.
    batch_idx: Optional batch index.
    highlight_obj: Represents the type of objects that will be highlighted with
      `color.COLOR_DICT['controlled']` color.

  Returns:
    np image.
  """
  if batch_idx > -1:
    if len(obs.shape) != 2:
      raise ValueError(f'Expecting ndim 2 for obs, got {len(obs.shape)}')
    obs = jax.tree_util.tree_map(lambda x: x[batch_idx], obs)

  # Shape: (obs_A,) -> ()
  obs = jax.tree_map(lambda x: x[obj_idx], obs)
  if obs.shape:
    raise ValueError(f'Expecting shape () for obs, got {obs.shape}')

  viz_config = (
      utils.VizConfig() if viz_config is None else utils.VizConfig(**viz_config)
  )
  fig, ax = utils.init_fig_ax(viz_config)

  # 1. Plots trajectory.
  # Shape: (num_objects, num_timesteps).
  traj = obs.trajectory
  # The current timestep index of observation is the last one in time dimension
  # as observation is toward the past.
  timestep = traj.num_timesteps - 1
  indices = np.arange(traj.num_objects) if viz_config.show_agent_id else None

  is_controlled = datatypes.get_control_mask(obs.metadata, highlight_obj)
  plot_trajectory(ax, traj, is_controlled, time_idx=timestep, indices=indices)  # pytype: disable=wrong-arg-types  # jax-ndarray

  # 2. Plots road graph elements.
  # Shape: (num_points,)
  plot_roadgraph_points(ax, obs.roadgraph_static_points, verbose=False)

  # Shape: (num_traffic_lights, num_timesteps).
  plot_traffic_light_signals_as_points(
      ax, obs.traffic_lights, timestep, verbose=False
  )

  ax.scatter(goal[0], goal[1], marker='X', c='blue')

  # 3. Gets np img, centered on selected agent's current location.
  # Shape (num_objects, 2).
  current_xy = traj.xy[:, timestep, :]
  if viz_config.center_agent_idx == -1:
    xy = current_xy[obs.metadata.is_sdc]
  else:
    xy = current_xy[viz_config.center_agent_idx]
  origin_x, origin_y = xy[0, :2]
  ax.axis((
      origin_x - viz_config.back_x,
      origin_x + viz_config.front_x,
      origin_y - viz_config.back_y,
      origin_y + viz_config.front_y,
  ))

  return utils.img_from_fig(fig)


def plot_observation_with_heading(
    obs: datatypes.Observation,
    obj_idx: int,
    heading: jnp.ndarray,
    viz_config: Optional[dict[str, Any]] = None,
    batch_idx: int = -1,
    highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC
) -> np.ndarray:
  """Plots np array image for an Observation.

  Args:
    obs: An Observation instance, with shape (..., obs_A), where obs_A
      represents the number of objects that have observation view over things
      including other objects, roadgraph, and traffic lights.
    obj_idx: The object index in obs_A.
    viz_config: Dict for optional config.
    batch_idx: Optional batch index.
    highlight_obj: Represents the type of objects that will be highlighted with
      `color.COLOR_DICT['controlled']` color.

  Returns:
    np image.
  """
  if batch_idx > -1:
    if len(obs.shape) != 2:
      raise ValueError(f'Expecting ndim 2 for obs, got {len(obs.shape)}')
    obs = jax.tree_util.tree_map(lambda x: x[batch_idx], obs)

  # Shape: (obs_A,) -> ()
  obs = jax.tree_map(lambda x: x[obj_idx], obs)
  if obs.shape:
    raise ValueError(f'Expecting shape () for obs, got {obs.shape}')

  viz_config = (
      utils.VizConfig() if viz_config is None else utils.VizConfig(**viz_config)
  )
  fig, ax = utils.init_fig_ax(viz_config)

  # 1. Plots trajectory.
  # Shape: (num_objects, num_timesteps).
  traj = obs.trajectory
  # The current timestep index of observation is the last one in time dimension
  # as observation is toward the past.
  timestep = traj.num_timesteps - 1
  indices = np.arange(traj.num_objects) if viz_config.show_agent_id else None

  is_controlled = datatypes.get_control_mask(obs.metadata, highlight_obj)
  plot_trajectory(ax, traj, is_controlled, time_idx=timestep, indices=indices)  # pytype: disable=wrong-arg-types  # jax-ndarray

  # 2. Plots road graph elements.
  # Shape: (num_points,)
  plot_roadgraph_points(ax, obs.roadgraph_static_points, verbose=False)

  # Shape: (num_traffic_lights, num_timesteps).
  plot_traffic_light_signals_as_points(
      ax, obs.traffic_lights, timestep, verbose=False
  )

  ax.quiver(0, 0, heading[..., 0], heading[..., 1], color='cyan')

  # 3. Gets np img, centered on selected agent's current location.
  # Shape (num_objects, 2).
  current_xy = traj.xy[:, timestep, :]
  if viz_config.center_agent_idx == -1:
    xy = current_xy[obs.metadata.is_sdc]
  else:
    xy = current_xy[viz_config.center_agent_idx]
  origin_x, origin_y = xy[0, :2]
  ax.axis((
      origin_x - viz_config.back_x,
      origin_x + viz_config.front_x,
      origin_y - viz_config.back_y,
      origin_y + viz_config.front_y,
  ))

  return utils.img_from_fig(fig)

def _plot_bounding_boxes(
    ax: matplotlib.axes.Axes,
    traj_5dof: np.ndarray,
    time_idx: int,
    is_controlled: np.ndarray,
    valid: np.ndarray,
) -> None:
  """Helper function to plot multiple bounding boxes across time."""
  # Plots bounding boxes (traj_5dof) with shape: (A, T)
  # is_controlled: (A,)
  # valid: (A, T)
  valid_controlled = is_controlled[:, np.newaxis] & valid
  valid_context = ~is_controlled[:, np.newaxis] & valid

  num_obj = traj_5dof.shape[0]
  time_indices = np.tile(
      np.arange(traj_5dof.shape[1])[np.newaxis, :], (num_obj, 1)
  )
  # Shrinks bounding_boxes for non-current steps.
  traj_5dof[time_indices != time_idx, 2:4] /= 10
  utils.plot_numpy_bounding_boxes(
      ax=ax,
      bboxes=traj_5dof[(time_indices >= time_idx) & valid_controlled],
      color=color.COLOR_DICT['controlled'],
  )

  utils.plot_numpy_bounding_boxes(
      ax=ax,
      bboxes=traj_5dof[(time_indices < time_idx) & valid_controlled],
      color=color.COLOR_DICT['controlled'],
      as_center_pts=True,
  )

  utils.plot_numpy_bounding_boxes(
      ax=ax,
      bboxes=traj_5dof[(time_indices >= time_idx) & valid_context],
      color=color.COLOR_DICT['context'],
  )

  # Shows current overlap
  # (A, A)
  overlap_fn = jax.jit(geometry.compute_pairwise_overlaps)
  overlap_mask_matrix = overlap_fn(traj_5dof[:, time_idx])
  # Remove overlap against invalid objects.
  overlap_mask_matrix = np.where(
      valid[None, :, time_idx], overlap_mask_matrix, False
  )
  # (A,)
  overlap_mask = np.any(overlap_mask_matrix, axis=1)

  utils.plot_numpy_bounding_boxes(
      ax=ax,
      bboxes=traj_5dof[:, time_idx][overlap_mask & valid[:, time_idx]],
      color=color.COLOR_DICT['overlap'],
  )

def _index_pytree(inputs: Any, idx: int) -> Any:
  """Helper function to get idx-th example in a batch."""

  def local_index(x):
    if x.ndim > 0:
      return x[idx]
    else:
      return x

  return jax.tree_util.tree_map(local_index, inputs)

def plot_trajectory(
    ax: matplotlib.axes.Axes,
    traj: datatypes.Trajectory,
    is_controlled: np.ndarray,
    time_idx: Optional[int] = None,
    indices: Optional[np.ndarray] = None,
) -> None:
  """Plots a Trajectory with different color for controlled and context.

  Plots the full bounding_boxes only for time_idx step, overlap is
  highlighted.

  Notation: A: number of agents; T: numbe of time steps; 5 degree of freedom:
  center x, center y, length, width, yaw.

  Args:
    ax: matplotlib axes.
    traj: a Trajectory with shape (A, T).
    is_controlled: binary mask for controlled object, shape (A,).
    time_idx: step index to highlight bbox, -1 for last step. Default(None) for
      not showing bbox.
    indices: ids to show for each agents if not None, shape (A,).
  """
  if len(traj.shape) != 2:
    raise ValueError('traj should have shape (A, T)')

  traj_5dof = np.array(
      traj.stack_fields(['x', 'y', 'length', 'width', 'yaw'])
  )  # Forces to np from jnp

  num_obj, num_steps, _ = traj_5dof.shape
  if time_idx is not None:
    if time_idx == -1:
      time_idx = num_steps - 1
    if time_idx >= num_steps:
      raise ValueError('time_idx is out of range.')

  # Adds id if needed.
  if indices is not None and time_idx is not None:
    for i in range(num_obj):
      if not traj.valid[i, time_idx]:
        continue
      ax.text(
          traj_5dof[i, time_idx, 0] - 2,
          traj_5dof[i, time_idx, 1] + 2,
          f'{indices[i]}',
          zorder=10,
      )
  _plot_bounding_boxes(ax, traj_5dof, time_idx, is_controlled, traj.valid)  # pytype: disable=wrong-arg-types  # jax-ndarray


def plot_simulator_state(
    state: datatypes.SimulatorState,
    viz_config: Optional[dict[str, Any]] = None,
    batch_idx: int = -1,
    highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC,
) -> np.ndarray:
  """Plots np array image for SimulatorState.

  Args:
    state: A SimulatorState instance.
    use_log_traj: Set True to use logged trajectory, o/w uses simulated
      trajectory.
    viz_config: dict for optional config.
    batch_idx: optional batch index.
    highlight_obj: Represents the type of objects that will be highlighted with
      `color.COLOR_DICT['controlled']` color.

  Returns:
    np image.
  """
  if batch_idx > -1:
    if len(state.shape) != 1:
      raise ValueError(
          'Expecting one batch dimension, got %s' % len(state.shape)
      )
    state = _index_pytree(state, batch_idx)
  if state.shape:
    raise ValueError('Expecting 0 batch dimension, got %s' % len(state.shape))

  viz_config = (
      utils.VizConfig() if viz_config is None else utils.VizConfig(**viz_config)
  )
  fig, ax = utils.init_fig_ax(viz_config)

  # 1. Plots road graph elements.
  plot_roadgraph_points(ax, state.roadgraph_points, verbose=False)
  plot_traffic_light_signals_as_points(
      ax, state.log_traffic_light, state.timestep, verbose=False
  )

  # 2. Plots trajectory.

  is_controlled = datatypes.get_control_mask(
      state.object_metadata, highlight_obj
  )

  # Plot log trajectory
  log_traj_5dof = np.array(
      state.log_trajectory.stack_fields(['x', 'y', 'length', 'width', 'yaw'])
  )  # Forces to np from jnp
  valid_controlled = is_controlled[:, np.newaxis] & state.log_trajectory.valid

  utils.plot_numpy_bounding_boxes(
      ax=ax,
      bboxes=log_traj_5dof[valid_controlled],
      color=np.array([0,1,1]), # log trajectory in cyan
      as_center_pts=True,
  )

  # Plot sim trajectory
  traj = state.sim_trajectory
  indices = np.arange(traj.num_objects) if viz_config.show_agent_id else None
  plot_trajectory(
      ax, traj, is_controlled, time_idx=state.timestep, indices=indices
  )  # pytype: disable=wrong-arg-types  # jax-ndarray

  # 3. Gets np img, centered on selected agent's current location.
  # [A, 2]
  current_xy = traj.xy[:, state.timestep, :]
  if viz_config.center_agent_idx == -1:
    xy = current_xy[state.object_metadata.is_sdc]
  else:
    xy = current_xy[viz_config.center_agent_idx]
  origin_x, origin_y = xy[0, :2]
  ax.axis((
      origin_x - viz_config.back_x,
      origin_x + viz_config.front_x,
      origin_y - viz_config.back_y,
      origin_y + viz_config.front_y,
  ))

  return utils.img_from_fig(fig)