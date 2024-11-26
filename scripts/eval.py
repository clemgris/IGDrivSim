import argparse
import functools
import jax
import json
import os
import pickle
import tensorflow as tf
import sys

sys.path.append(".")

from waymax import config as _config

from IGDrivSim.utils import (
    tf_examples_dataset,
    preprocess_serialized_womd_data,
    inter_filter_funct,
    speed_filter_funct,
    sub_val_filter_funct,
)
from IGDrivSim.model import make_eval_heuristic_policy, make_eval

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print(jax.devices())

gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

##
# CONFIG
##

parser = argparse.ArgumentParser(description="Agent evaluation")
parser.add_argument("--expe_id", "-expe", type=str, help="Id of the experiment")
parser.add_argument("--epochs", "-e", type=int, help="Number of training epochs")
parser.add_argument(
    "--folder", "-f", type=str, help="Folder containing the model", default="logs"
)
parser.add_argument(
    "--n_envs_eval", "-n", type=int, help="Evaluation batch size", default=1
)
parser.add_argument(
    "--sub_valid_model",
    "-sub_valid_model",
    type=str,
    help="Name of the validation subset to be evaluated on",
    default=None,
)
parser.add_argument(
    "--IDM", "-IDM", type=int, help="Use IDM for simulated agents", default=0
)
parser.add_argument("--GIF", "-GIF", type=int, help="Generate GIFs", default=0)

if __name__ == "__main__":
    args = parser.parse_args()

    # Training config
    load_folder = f"./{args.folder}"
    expe_num = args.expe_id

    os.makedirs(f"./animation/{expe_num}", exist_ok=True)

    if expe_num == "heuristic_policy":
        config = {
            "bins": 256,
            "discrete": False,
            "dynamics": "delta",
            "extractor": "ExtractObs",
            "feature_extractor": "KeyExtractor",
            "feature_extractor_kwargs": {
                "final_hidden_layers": None,
                "hidden_layers": {},
                "keys": ["proxy_goal"],
                "kwargs": {},
            },
            "include_sdc_paths": False,
            "key": 42,
            "log_folder": "_____heuristic_policy",
            "max_num_obj": 32,
            "max_num_rg_points": 20000,
            "num_steps": 80,
            "obs_mask": None,
            "obs_mask_kwargs": None,
            "roadgraph_top_k": 2000,
            "training_path": "gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/tf_example/training/training_tfexample.tfrecord@1000",
            "validation_path": "gs://waymo_open_dataset_motion_v_1_1_0/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150",
            "should_cache": True,
        }
    else:
        with open(os.path.join(load_folder, expe_num, "args.json"), "r") as file:
            config = json.load(file)

    config["num_epochs"] = 1
    config["num_envs_eval"] = args.n_envs_eval
    config["folder_name"] = args.folder

    if args.GIF and args.n_envs_eval != 1:
        raise ValueError("To generate gifs, the batch size (n_envs_eval) has to be 1.")

    filter_functions = {
        "inter_filter_fun": inter_filter_funct,
        "speed_filter_fun": speed_filter_funct,
        "sub_val_filter_fun": sub_val_filter_funct,
    }

    # config['filter_fun_name'] = 'speed_filter_fun'
    # config['filter_fun_args'] = {'min_mean_speed': 1}

    config["filter_fun_name"] = "sub_val_filter_fun"
    if args.sub_valid_model:
        with open(
            os.path.join(
                f"../dataset/sub_validation_id/sub_validation_id_{args.sub_valid_model}.pkl"
            ),
            "rb",
        ) as file:
            sub_val_id = pickle.load(file)
            config["filter_fun_args"] = args.sub_valid_model
        filter_fun_args = {"sub_val_set": sub_val_id}
        filter_fun = functools.partial(
            filter_functions[config["filter_fun_name"]], **filter_fun_args
        )
    else:
        filter_fun = None

    n_epochs = args.epochs

    print("Create datasets")

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

    # Validation dataset
    val_dataset = tf_examples_dataset(
        path=WOD_1_1_0_VALIDATION.path,
        data_format=WOD_1_1_0_VALIDATION.data_format,
        preprocess_fn=functools.partial(
            preprocess_serialized_womd_data, config=WOD_1_1_0_VALIDATION
        ),
        shuffle_seed=WOD_1_1_0_VALIDATION.shuffle_seed,
        shuffle_buffer_size=WOD_1_1_0_VALIDATION.shuffle_buffer_size,
        repeat=WOD_1_1_0_VALIDATION.repeat,
        batch_dims=WOD_1_1_0_VALIDATION.batch_dims,
        num_shards=WOD_1_1_0_VALIDATION.num_shards,
        deterministic=WOD_1_1_0_VALIDATION.deterministic,
        drop_remainder=False,
        filter_function=filter_fun,
        tf_data_service_address=WOD_1_1_0_VALIDATION.tf_data_service_address,
        batch_by_scenario=WOD_1_1_0_VALIDATION.batch_by_scenario,
    )

    # Env config
    config["IDM"] = bool(args.IDM)
    config["GIF"] = bool(args.GIF)

    env_config = _config.EnvironmentConfig(
        controlled_object=_config.ObjectType.SDC,
        max_num_objects=config["max_num_obj"],
    )

    ##
    # EVALUATION
    ##

    if expe_num == "heuristic_policy":
        evaluation = make_eval_heuristic_policy(config, env_config, val_dataset, None)
    else:
        print("Load network parameters")

        with open(
            os.path.join(load_folder, expe_num, f"params_{n_epochs}.pkl"), "rb"
        ) as file:
            params = pickle.load(file)

        evaluation = make_eval(config, env_config, val_dataset, params)

    # with jax.disable_jit(): # DEBUG
    evaluation_dict = evaluation.train()

    if args.sub_valid_model is None:
        with open(
            os.path.join(
                load_folder,
                expe_num,
                f"eval_metrics_{n_epochs}_IDM_{config['IDM']}.pkl",
            ),
            "wb",
        ) as pkl_file:
            pickle.dump(evaluation_dict["metrics"], pkl_file)
        with open(
            os.path.join(
                load_folder,
                expe_num,
                f"config_eval_{n_epochs}_IDM_{config['IDM']}.json",
            ),
            "w",
        ) as json_file:
            json.dump(config, json_file, indent=4)

        print(
            f"{evaluation_dict['hard_scenario_id'].shape[0]} hard scenario (overlap > 10%)"
        )
        print(
            f"{evaluation_dict['easy_scenario_id'].shape[0]} easy scenario (overlap = 0%)"
        )

        with open(
            os.path.join(
                "../dataset/sub_validation_id",
                f"sub_validation_id_{expe_num}_hard.pkl",
            ),
            "wb",
        ) as pkl_file:
            pickle.dump(evaluation_dict["hard_scenario_id"], pkl_file)

        with open(
            os.path.join(
                "../dataset/sub_validation_id",
                f"sub_validation_id_{expe_num}_easy.pkl",
            ),
            "wb",
        ) as pkl_file:
            pickle.dump(evaluation_dict["easy_scenario_id"], pkl_file)
    else:
        with open(
            os.path.join(
                load_folder,
                expe_num,
                f"eval_metrics_{n_epochs}_IDM_{config['IDM']}_sub_validation_{args.sub_valid_model}.pkl",
            ),
            "wb",
        ) as pkl_file:
            pickle.dump(evaluation_dict["metrics"], pkl_file)
        with open(
            os.path.join(
                load_folder,
                expe_num,
                f"config_eval_{n_epochs}_IDM_{config['IDM']}_sub_validation_{args.sub_valid_model}.json",
            ),
            "w",
        ) as json_file:
            json.dump(config, json_file, indent=4)
