# Copyright 2023 The Waymax Authors.
#
# Licensed under the Waymax License Agreement for Non-commercial Use
# Use (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     https://github.com/waymo-research/waymax/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code snippet is adapted from the Waymax library with slight modifications
# tailored for the specific requirements of this project.


"""Util functions for general dataloading."""

import functools
import math
import os
from typing import Callable, Optional, Sequence, TypeVar

import jax
import tensorflow as tf

from waymax import config as _config
from waymax import dataloader


T = TypeVar('T')
AUTOTUNE = tf.data.AUTOTUNE


def generate_sharded_filenames(path: str) -> Sequence[str]:
  """Returns the filenames of individual sharded files.

  A sharded file is a set of files of the format filename-XXXXX-of-YYYYY,
  where XXXXX is a placeholder for the index of the shard, and YYYYY is the
  total number of shards. These files are collectively referred to by a
  sharded path filename@YYYYY.

  For example, the sharded path `myfile@100` refers to the set of files
    - myfile-00000-of-00100
    - myfile-00001-of-00100
    - ...
    - myfile-00098-of-00100
    - myfile-00099-of-00100

  Args:
    path: A path to a sharded file, with format `filename@shards`, where shards
      is an integer denoting the number of total shards.

  Returns:
    An iterator through the complete set of filenames that the path refers to,
    with each filename having the format `filename-XXXXX-of-YYYYY`
  """
  base_name, num_shards = path.split('@')
  num_shards = int(num_shards)
  shard_width = max(5, int(math.log10(num_shards) + 1))
  format_str = base_name + '-%0' + str(shard_width) + 'd-of-%05d'
  return [format_str % (i, num_shards) for i in range(num_shards)]


def tf_examples_dataset(
    path: str,
    data_format: _config.DataFormat,
    preprocess_fn: Callable[[bytes], dict[str, tf.Tensor]],
    shuffle_seed: Optional[int] = None,
    shuffle_buffer_size: int = 100,
    repeat: Optional[int] = None,
    batch_dims: Sequence[int] = (),
    num_shards: int = 1,
    deterministic: bool = True,
    drop_remainder: bool = False,
    tf_data_service_address: Optional[str] = None,
    batch_by_scenario: bool = True,
    filter_function: Callable[[dict[str, tf.Tensor]], bool] = lambda x: x,
    num_files: int=None,
    should_cache: bool = True,
) -> tf.data.Dataset:
  """Returns a dataset of Open Motion dataset TFExamples.

  Each TFExample contains data for the trajectory of all objects, the roadgraph,
  and traffic light states. See https://waymo.com/open/data/motion/tfexample
  for the data format definition.

  Args:
    path: The path to the dataset.
    data_format: Data format of the dataset.
    preprocess_fn: Function for parsing and preprocessing individual examples.
    shuffle_seed: Seed for shuffling. If left default (None), will not shuffle
      the dataset.
    shuffle_buffer_size: The size of the shuffle buffer.
    repeat: Number of times to repeat the dataset. Default (None) will repeat
      infinitely.
    batch_dims: List of size of batch dimensions. Multiple batch dimension can
      be used to provide inputs for multiple devices. E.g.
      [jax.local_device_count(), batch_size_per_device].
    num_shards: Number of shards for parallel loading, no effect on data
      returned.
    deterministic: Whether to use deterministic parallel processing.
    drop_remainder: Arg for tf.data.Dataset.batch. Set True to drop remainder if
      the last batch does not contains enough examples.
    tf_data_service_address: Set to use tf data service.
    batch_by_scenario: If True, one example in a returned batch is the entire
      scenario containing all objects; if False, the dataset will treat
      individual object trajectories as a training example rather than an entire
      scenario.

  Returns:
    A tf.data.Dataset of Open Motion Dataset tf.Example elements.
  """

  if data_format == _config.DataFormat.TFRECORD:
    dataset_fn = tf.data.TFRecordDataset
  else:
    raise ValueError('Data format %s is not supported.' % data_format)

  # Get the list of files containing the training data
  files_to_load = [path]
  if '@' in os.path.basename(path):
    files_to_load = generate_sharded_filenames(path)
  # Truncate the dataset
  if num_files:
    files_to_load = files_to_load[:num_files]
  files = tf.data.Dataset.from_tensor_slices(files_to_load)
  # Split files across multiple processes for distributed training/eval.
  files = files.shard(jax.process_count(), jax.process_index())

  def _make_dataset(
      shard_index: int, num_shards: int, local_files: tf.data.Dataset
  ):
    local_files = local_files.shard(num_shards, shard_index)
    ds = dataset_fn(local_files)
    ds = ds.map(
        preprocess_fn, num_parallel_calls=AUTOTUNE, deterministic=deterministic
    )
    if filter_function:
      ds = ds.filter(filter_function)
    return ds
  make_dataset_fn = functools.partial(
      _make_dataset, num_shards=num_shards, local_files=files
  )
  indices = tf.data.Dataset.range(num_shards)
  dataset = indices.interleave(
      make_dataset_fn, num_parallel_calls=AUTOTUNE, deterministic=deterministic
  )
  # Cache
  if should_cache:
    dataset = dataset.cache()
  # Shuffle
  if shuffle_seed is not None:
    local_seed = jax.random.PRNGKey(shuffle_seed)[0]
    dataset = dataset.shuffle(shuffle_buffer_size, seed=local_seed)
  # Repeat
  dataset = dataset.repeat(repeat)
  # Batch
  if not batch_by_scenario:
    dataset = dataset.unbatch()
  if batch_dims:
    for batch_size in reversed(batch_dims):
      dataset = dataset.batch(
          batch_size,
          drop_remainder=drop_remainder,
          num_parallel_calls=AUTOTUNE,
          deterministic=deterministic,
      )

  if tf_data_service_address is not None:
    dataset = dataset.apply(
        tf.data.experimental.service.distribute(
            processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
            service=tf_data_service_address,
        )
    )
  return dataset.prefetch(AUTOTUNE)

def inter_filter_funct(data: Callable[[dict[str, tf.Tensor]], bool]):
    is_sdc = data['state/is_sdc']

    has_sdc = tf.reduce_any(is_sdc > 0)

    if has_sdc:
        interact_condition = tf.math.reduce_any(data['state/objects_of_interest'] == 1)

        return interact_condition
    else:
        return False

def speed_filter_funct(data: Callable[[dict[str, tf.Tensor]], bool],
                       min_mean_speed=2):
    is_sdc = data['state/is_sdc']

    has_sdc = tf.reduce_any(is_sdc > 0)

    if has_sdc:

        obj_speed = data['state/all/speed']
        _, sdc_idx = tf.math.top_k(data['state/is_sdc'], k=1)
        sdc_speed = tf.experimental.numpy.take_along_axis(obj_speed, sdc_idx[..., None], axis=1)

        mean_speed_condition = tf.reduce_mean(sdc_speed) > min_mean_speed

        return mean_speed_condition
    else:
        return False

def inter_speed_filter_funct(data: Callable[[dict[str, tf.Tensor]], bool],
                             min_mean_speed=0.5):
  is_sdc = data['state/is_sdc']

  has_sdc = tf.reduce_any(is_sdc > 0)

  if has_sdc:
    interact_condition = tf.math.reduce_any(data['state/objects_of_interest'] == 1)
    obj_speed = data['state/all/speed']
    _, sdc_idx = tf.math.top_k(data['state/is_sdc'], k=1)
    sdc_speed = tf.experimental.numpy.take_along_axis(obj_speed, sdc_idx[..., None], axis=1)

    mean_speed_condition = tf.reduce_mean(sdc_speed) > min_mean_speed

    return mean_speed_condition & interact_condition
  else:
      return False

def sub_val_filter_funct(data: Callable[[dict[str, tf.Tensor]], bool],
                      sub_val_set):
  is_sdc = data['state/is_sdc']
  has_sdc = tf.reduce_any(is_sdc > 0)

  if has_sdc:
    id = data['scenario/id']
    in_sub_val_set = tf.reduce_any(tf.reduce_all(id == sub_val_set, axis=-1))

    return in_sub_val_set
  else:
      return False

def pad_id(tensor):
  desired_shape = [1, 16]
  tensor_shape = tf.shape(tensor)

  padded_tensor = tf.pad(tensor, [[0, 0], [0, desired_shape[-1] - tensor_shape[-1]]])

  return padded_tensor

def preprocess_serialized_womd_data(
    serialized: bytes, config: _config.DatasetConfig
) -> dict[str, tf.Tensor]:
  womd_features = dataloader.womd_utils.get_features_description(
      include_sdc_paths=config.include_sdc_paths,
      max_num_rg_points=config.max_num_rg_points,
      num_paths=config.num_paths,
      num_points_per_path=config.num_points_per_path,
  )
  womd_features['scenario/id'] = tf.io.FixedLenFeature([1], tf.string)

  deserialized = tf.io.parse_example(serialized, womd_features)

  parsed_id = deserialized.pop('scenario/id')
  deserialized['scenario/id'] = pad_id(tf.io.decode_raw(parsed_id, tf.int8))

  return dataloader.preprocess_womd_example(
      deserialized,
      aggregate_timesteps=config.aggregate_timesteps,
      max_num_objects=config.max_num_objects,
  )