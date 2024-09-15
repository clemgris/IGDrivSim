import jax
import jax.numpy as jnp

from waymax import datatypes

def combine_two_object_pose_2d(
    src_pose: datatypes.ObjectPose2D, dst_pose: datatypes.ObjectPose2D
) -> datatypes.ObjectPose2D:
  """Combines two ObjectPose2D as inverse(src_pose) plus dst_pose.
  (Taken from global_observation_from_state in waymax.observation.py)

  Applying transformation using the returned pose is equivalent to applying
  transformation first with inverse(src_pose) and then dst_pose. Note as data
  transformation is much more expensive than computing the combined pose, it's
  more efficient to apply one data transformation with the combined pose instead
  of applying multiple transformations with multiple poses.

  Args:
    src_pose: The source pose.
    dst_pose: The destination/target pose.

  Returns:
    The combined pose.
  """
  return datatypes.ObjectPose2D.from_transformation(
      matrix=jnp.matmul(
          dst_pose.matrix, jnp.linalg.inv(src_pose.matrix), precision='float32'
      ),
      delta_yaw=dst_pose.delta_yaw - src_pose.delta_yaw,
      valid=dst_pose.valid & src_pose.valid,
  )

def radius_point_extra(line_point, line_dir, circle_center, circle_radius):
  """Return an intersection between the semi line defined by the a point
  line_point and a direction line_dir and a circle of center circle_center
  and radius circle_radius. If no such intersection exists, it returns the default
  value (0,0).

  Args:
      line_point: The semi line initial point.
      line_dir: The semi line direction vector.
      circle_center: The circle center.
      circle_radius: The radius of the circle.
  Returns:
      An intersection if it exists and (0,0) otherwise.
  """
  def circle_semi_line_intersection(line_point, line_dir, circle_center, circle_radius):
      vec_to_center = circle_center - line_point
      projection_length = jnp.dot(vec_to_center, line_dir.T) / jnp.linalg.norm(line_dir)
      closest_point = line_point + (projection_length * line_dir / jnp.linalg.norm(line_dir))
      distance_to_center = jnp.linalg.norm(closest_point - circle_center)

      d = jnp.sqrt(jnp.maximum(circle_radius**2 - distance_to_center**2, 0))
      direction = line_dir / jnp.linalg.norm(line_dir)
      intersection1 = closest_point + d * direction
      intersection2 = closest_point - d * direction

      intersection1_valid = (jnp.dot(intersection1 - line_point, line_dir.T) >= 0) & (distance_to_center <= circle_radius)
      intersection2_valid = (jnp.dot(intersection2 - line_point, line_dir.T) >= 0) & (distance_to_center <= circle_radius)

      intersection = jax.lax.cond(intersection1_valid,
                                  lambda _: intersection1,
                                  lambda _: jax.lax.cond(intersection2_valid,
                                                         lambda _: intersection2,
                                                         lambda _: jnp.zeros_like(intersection2),
                                                         None),
                                  None)
      return intersection
  return jax.vmap(circle_semi_line_intersection, (0, 0, 0, None))(line_point, line_dir, circle_center, circle_radius)