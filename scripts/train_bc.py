from datetime import datetime
import functools
import jax
import jax.numpy as jnp
import json
import os

# from jax import config
# config.update("jax_debug_nans", True)

from waymax import config as _config
from waymax import dataloader
from rnnbc import make_train

from dataset.config import N_TRAINING_INTER, N_TRAINING
from utils.dataloader import tf_examples_dataset, inter_filter_funct, speed_filter_funct

# Desable preallocation for jax and tensorflow
import os

print(jax.devices())
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

##
# CONFIG
##

# Training config
config = {
    "algo": "bc",
    "bins": 256,
    "discrete": True,
    "dynamics": "delta",
    "extractor": "ExtractObs",
    "feature_extractor": "KeyExtractor",
    "feature_extractor_kwargs": {
        "final_hidden_layers": 512,
        "hidden_layers": {
            #  'xy': 128,
            #  'xyyaw': 128,
            #  'xyyawv': 512,
            #  'sdc_speed': 32,
            #  'heading': 32,
            #  '[xyyawv, sdc_speed, heading]': 512,
            "[xyyawv, sdc_speed]": 512,
            "roadgraph_map": 512,
        },
        "keys": [
            # 'xy',
            # 'xyyaw',
            # 'xyyawv',
            # 'sdc_speed',
            # 'proxy_goal',
            # 'noisy_proxy_goal',
            # 'heading',
            # '[xyyawv, sdc_speed, heading]',
            "[xyyawv, sdc_speed]",
            "roadgraph_map",
        ],
        "kwargs": {
            #  'heading': {'radius': 20},
            #  'noisy_proxy_goal': {'sigma': 100}
        },
    },
    "freq_eval": 10,
    "freq_save": 10,
    "filter_fun_name": "inter_filter_fun",
    "filter_fun_args": {},
    "include_sdc_paths": False,
    "key": 42,
    "loss": "logprob",
    "lr": 3e-4,
    "lr_max": 3e-4,
    "lr_transition_epoch": 200,
    "lr_scheduler": "linear",  #'one_cycle_cosine', #'cosine'
    "max_grad_norm": 0.5,
    "max_num_obj": 32,
    "max_num_rg_points": 20000,
    "num_envs": 256,
    "num_envs_eval": 128,
    "num_epochs": 200,
    "num_steps": 80,
    "obs_mask": None,  #'SpeedConicObsMask', #'ZeroMask', #'SpeedGaussianNoise', #'SpeedUniformNoise', #'SpeedConicObsMask',
    "obs_mask_kwargs": None,
    # {},
    # { # Uniform noise
    # 'v_max': 15,
    # 'bound_max': 5
    # },
    # { # Gaussian noise
    # 'v_max': 1,
    # 'sigma_max':0
    # },
    # {
    # 'radius': 0, #None, # Sanity check (as full obs)
    # 'angle_max': 0, # 2 / 3 * jnp.pi,
    # 'angle_min': 0, # Sanity check (as full obs)
    # 'v_max': 15, # 15,
    # },
    "optimiser": "adamw",
    "roadgraph_top_k": 2000,
    "shuffle_seed": 123,
    "shuffle_buffer_size": 1000,
    "total_timesteps": 100,
    "num_files": 1000,
    "training_path": "gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/tf_example/training/training_tfexample.tfrecord@1000",
    "validation_path": "gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150",
    "should_cache": True,
}

# for radius in [15]: #15, 25, 50, 100]:
#     for angle_min in [jnp.pi / 12]: #12, jnp.pi / 8, jnp.pi / 4, jnp.pi / 2]:

# config['obs_mask_kwargs']['radius'] = radius
# config['obs_mask_kwargs']['angle_min'] = angle_min

# for sigma_max in [0,1,3,5]:
#

# config['obs_mask_kwargs']['sigma_max'] = sigma_max

# for radius in [1, 5, 10, 20, 50]:

#     config['feature_extractor_kwargs']['kwargs']['heading']['radius'] = radius

trials = []

# No obs
trials.append(
    {
        "radius": 0,
        "angle_max": 0,  # 2 / 3 * jnp.pi,
        "angle_min": 0,  # Sanity check (as full obs)
        "v_max": 15,  # 15,
    }
)

# Partial obs
trials.append(
    {
        "radius": 2,
        "angle_max": 2 / 3 * jnp.pi,
        "angle_min": 1 / 4 * jnp.pi,
        "v_max": 15,  # 15,
    }
)

trials.append(
    {
        "radius": 5,
        "angle_max": 2 / 3 * jnp.pi,
        "angle_min": 1 / 4 * jnp.pi,
        "v_max": 15,  # 15,
    }
)

trials.append(
    {
        "radius": 10,
        "angle_max": 2 / 3 * jnp.pi,
        "angle_min": 1 / 4 * jnp.pi,
        "v_max": 15,  # 15,
    }
)

trials.append(
    {
        "radius": 15,
        "angle_max": 2 / 3 * jnp.pi,
        "angle_min": 1 / 4 * jnp.pi,
        "v_max": 15,  # 15,
    }
)

# for trial in trials:
#     config['obs_mask_kwargs'] = trial

# Ckeckpoint path
current_time = datetime.now()
date_string = current_time.strftime("%Y%m%d_%H%M%S")

log_folder = f"logs/{date_string}"
os.makedirs(log_folder, exist_ok="True")

config["log_folder"] = log_folder

# Save training config
training_args = config

# Data iter config
WOD_1_1_0_TRAINING = _config.DatasetConfig(
    path=config["training_path"],
    max_num_rg_points=config["max_num_rg_points"],
    shuffle_seed=config["shuffle_seed"],
    shuffle_buffer_size=config["shuffle_buffer_size"],
    data_format=_config.DataFormat.TFRECORD,
    batch_dims=(config["num_envs"],),
    max_num_objects=config["max_num_obj"],
    include_sdc_paths=config["include_sdc_paths"],
    repeat=None,
)

# Data iter config
WOD_1_1_0_VALIDATION = _config.DatasetConfig(
    path=config["validation_path"],
    max_num_rg_points=config["max_num_rg_points"],
    shuffle_seed=None,
    data_format=_config.DataFormat.TFRECORD,
    batch_dims=(config["num_envs_eval"],),
    max_num_objects=config["max_num_obj"],
    include_sdc_paths=config["include_sdc_paths"],
    repeat=1,
)

filter_functions = {
    "inter_filter_fun": inter_filter_funct,
    "speed_filter_fun": speed_filter_funct,
}

# Training dataset
train_dataset = tf_examples_dataset(
    path=WOD_1_1_0_TRAINING.path,
    data_format=WOD_1_1_0_TRAINING.data_format,
    preprocess_fn=functools.partial(
        dataloader.preprocess_serialized_womd_data, config=WOD_1_1_0_TRAINING
    ),
    shuffle_seed=WOD_1_1_0_TRAINING.shuffle_seed,
    shuffle_buffer_size=WOD_1_1_0_TRAINING.shuffle_buffer_size,
    repeat=WOD_1_1_0_TRAINING.repeat,
    batch_dims=WOD_1_1_0_TRAINING.batch_dims,
    num_shards=WOD_1_1_0_TRAINING.num_shards,
    deterministic=WOD_1_1_0_TRAINING.deterministic,
    drop_remainder=WOD_1_1_0_TRAINING.drop_remainder,
    tf_data_service_address=WOD_1_1_0_TRAINING.tf_data_service_address,
    batch_by_scenario=WOD_1_1_0_TRAINING.batch_by_scenario,
    filter_function=functools.partial(
        filter_functions[config["filter_fun_name"]], **config["filter_fun_args"]
    ),
    num_files=config["num_files"],
    should_cache=config["should_cache"],
)

if config["filter_fun_name"]:
    if config["filter_fun_name"] == "inter_filter_fun":
        config["num_training_data"] = N_TRAINING_INTER
    else:
        raise ValueError(
            "Number of training data satisfying the filtering condition is unknown"
        )
else:
    config["num_training_data"] = N_TRAINING

data = train_dataset.as_numpy_iterator().next()  # DEBUG

# Validation dataset
val_dataset = dataloader.tf_examples_dataset(
    path=WOD_1_1_0_VALIDATION.path,
    data_format=WOD_1_1_0_VALIDATION.data_format,
    preprocess_fn=functools.partial(
        dataloader.preprocess_serialized_womd_data, config=WOD_1_1_0_VALIDATION
    ),
    shuffle_seed=WOD_1_1_0_VALIDATION.shuffle_seed,
    shuffle_buffer_size=WOD_1_1_0_VALIDATION.shuffle_buffer_size,
    repeat=WOD_1_1_0_VALIDATION.repeat,
    batch_dims=WOD_1_1_0_VALIDATION.batch_dims,
    num_shards=WOD_1_1_0_VALIDATION.num_shards,
    deterministic=WOD_1_1_0_VALIDATION.deterministic,
    drop_remainder=WOD_1_1_0_VALIDATION.drop_remainder,
    tf_data_service_address=WOD_1_1_0_VALIDATION.tf_data_service_address,
    batch_by_scenario=WOD_1_1_0_VALIDATION.batch_by_scenario,
)

# Env config
env_config = _config.EnvironmentConfig(
    controlled_object=_config.ObjectType.SDC, max_num_objects=config["max_num_obj"]
)

with open(os.path.join(log_folder, "args.json"), "w") as json_file:
    json.dump(training_args, json_file, indent=4)

##
# TRAINING
##
print(jax.devices())
training = make_train(
    config,
    env_config,
    train_dataset,
    val_dataset,
    data,  # DEBUG
)

# with jax.disable_jit(): # DEBUG
training_dict = training.train()
