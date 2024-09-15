import argparse
from datetime import datetime
import functools
import jax
import json
import os

from waymax import config as _config
from waymax import dataloader

import sys
sys.path.append('.')

from IGWaymax.model import make_train
from IGWaymax.dataset import N_TRAINING_INTER, N_TRAINING, N_TRAINING_INTER_SPEED_0_5, N_TRAINING_INTER_SPEED_0_7
from IGWaymax.utils import tf_examples_dataset, inter_filter_funct, speed_filter_funct, inter_speed_filter_funct, count_unfiltered_data

# Desable preallocation for jax and tensorflow
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
print(jax.devices())

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser(description="Agent training using BC & RL")
parser.add_argument('--config', '-conf', type=str, help='Name of the config')
parser.add_argument('--loss_weight_bc', '-w_bc', type=float, help='Weight of the BC loss')
parser.add_argument('--loss_weight_rl', '-w_rl', type=float, help='Weight of the RL loss')
parser.add_argument('--sigma', type=float, default=None)
parser.add_argument('--radius', type=float, default=None)
parser.add_argument('--prob',  type=float, default=None)

if __name__ == "__main__":
    args = parser.parse_args()

    ##
    # CONFIG
    ##

    with open(os.path.join("./scripts/configs", f'{args.config}.json'), 'r') as file:
        config = json.load(file)

    config['loss_weight_bc'] = args.loss_weight_bc
    config['loss_weight_rl'] = args.loss_weight_rl

    if config['loss_weight_rl'] == 0:
        config['algo'] = 'bc'
    else:
        config['algo'] = 'bc_rl'

    if config['obs_mask_kwargs']:
        for key, _ in config['obs_mask_kwargs'].items():
            config['obs_mask_kwargs'][key] = getattr(args, key)

    # Ckeckpoint path
    current_time = datetime.now()
    date_string = current_time.strftime("%Y%m%d_%H%M%S")

    log_folder = f"logs_bc_rl/{date_string}"
    os.makedirs(log_folder, exist_ok='True')

    config['log_folder'] = log_folder

    # Save training config
    training_args = config

    # Data iter config
    WOD_1_1_0_TRAINING = _config.DatasetConfig(
        path=config['training_path'],
        max_num_rg_points=config['max_num_rg_points'],
        shuffle_seed=config['shuffle_seed'],
        shuffle_buffer_size=config['shuffle_buffer_size'],
        data_format=_config.DataFormat.TFRECORD,
        batch_dims = (config['num_envs'],),
        max_num_objects=config['max_num_obj'],
        include_sdc_paths=config['include_sdc_paths'],
        repeat=None
    )

    # Data iter config
    WOD_1_1_0_VALIDATION = _config.DatasetConfig(
        path=config['validation_path'],
        max_num_rg_points=config['max_num_rg_points'],
        shuffle_seed=None,
        data_format=_config.DataFormat.TFRECORD,
        batch_dims = (config['num_envs_eval'],),
        max_num_objects=config['max_num_obj'],
        include_sdc_paths=config['include_sdc_paths'],
        repeat=1
    )

    filter_functions = {'inter_filter_fun': inter_filter_funct,
                        'speed_filter_fun': speed_filter_funct,
                        'inter_speed_filter_fun': inter_speed_filter_funct}

    # Training dataset
    train_dataset = tf_examples_dataset(
        path=WOD_1_1_0_TRAINING.path,
        data_format=WOD_1_1_0_TRAINING.data_format,
        preprocess_fn=functools.partial(dataloader.preprocess_serialized_womd_data, config=WOD_1_1_0_TRAINING),
        shuffle_seed=WOD_1_1_0_TRAINING.shuffle_seed,
        shuffle_buffer_size=WOD_1_1_0_TRAINING.shuffle_buffer_size,
        repeat=WOD_1_1_0_TRAINING.repeat,
        batch_dims=WOD_1_1_0_TRAINING.batch_dims,
        num_shards=WOD_1_1_0_TRAINING.num_shards,
        deterministic=WOD_1_1_0_TRAINING.deterministic,
        drop_remainder=WOD_1_1_0_TRAINING.drop_remainder,
        tf_data_service_address=WOD_1_1_0_TRAINING.tf_data_service_address,
        batch_by_scenario=WOD_1_1_0_TRAINING.batch_by_scenario,
        filter_function=functools.partial(filter_functions[config['filter_fun_name']], **config['filter_fun_args']),
        num_files = config['num_files'],
        should_cache = config['should_cache']
    )

    if config['filter_fun_name']:
        if config['filter_fun_name'] == 'inter_filter_fun':
            config['num_training_data'] = N_TRAINING_INTER
        elif (config['filter_fun_name'] == 'inter_speed_filter_fun') and (config['filter_fun_args']['min_mean_speed'] == 0.5):
            config['num_training_data'] = N_TRAINING_INTER_SPEED_0_5
        elif (config['filter_fun_name'] == 'inter_speed_filter_fun') and (config['filter_fun_args']['min_mean_speed'] == 0.7):
            config['num_training_data'] = N_TRAINING_INTER_SPEED_0_7
        else:
            print('Counting training data that satisfy the filter (this may take some time)')
            config['num_training_data'] = count_unfiltered_data(config['filter_fun_name'], config['filter_fun_args'])
    else:
        config['num_training_data'] = N_TRAINING

    print('Number of training data:', config['num_training_data'])

    data = train_dataset.as_numpy_iterator().next() # DEBUG

    # Validation dataset
    val_dataset = dataloader.tf_examples_dataset(
        path=WOD_1_1_0_VALIDATION.path,
        data_format=WOD_1_1_0_VALIDATION.data_format,
        preprocess_fn=functools.partial(dataloader.preprocess_serialized_womd_data, config=WOD_1_1_0_VALIDATION),
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
        controlled_object=_config.ObjectType.SDC,
        max_num_objects=config['max_num_obj']
    )

    with open(os.path.join(log_folder, 'args.json'), 'w') as json_file:
        json.dump(training_args, json_file, indent=4)

    ##
    # TRAINING
    ##
    print(jax.devices())
    training = make_train(config,
                        env_config,
                        train_dataset,
                        val_dataset,
                        data # DEBUG
                        )

    # with jax.disable_jit(): # DEBUG
    training_dict = training.train()
