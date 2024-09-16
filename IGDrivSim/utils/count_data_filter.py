import functools
from tqdm import tqdm
from typing import Dict

from waymax import config as _config
from waymax import dataloader

from IGDrivSim.dataset import N_TRAINING
from .dataloader import (
    tf_examples_dataset,
    inter_filter_funct,
    speed_filter_funct,
    inter_speed_filter_funct,
)


def count_unfiltered_data(filter_fun_name: str, filter_fun_args: Dict):
    # Training config
    config = {
        "include_sdc_paths": False,
        "max_num_obj": 32,
        "max_num_rg_points": 20000,
        "num_envs": 1,
        "shuffle_seed": 123,
        "shuffle_buffer_size": 1000,
        "num_files": 1000,
        "training_path": "/data/tucana/shared/WOD_1_1_0/tf_example/training/training_tfexample.tfrecord@1000",
        "validation_path": "/data/tucana/shared/WOD_1_1_0/tf_example/validation/validation_tfexample.tfrecord@150",
        "should_cache": False,
    }

    config["filter_fun_name"] = filter_fun_name
    config["filter_fun_args"] = filter_fun_args

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
        repeat=1,
    )

    filter_functions = {
        "inter_filter_fun": inter_filter_funct,
        "speed_filter_fun": speed_filter_funct,
        "inter_speed_filter_fun": inter_speed_filter_funct,
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

    count = 0
    for _ in tqdm(train_dataset.as_numpy_iterator(), desc="Training", total=N_TRAINING):
        count += 1

    return count
