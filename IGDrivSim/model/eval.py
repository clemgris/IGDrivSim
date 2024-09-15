import dataclasses
from flax.training.train_state import TrainState
import functools
import jax
import jax.numpy as jnp
from jax import random
from PIL import Image

import numpy as np
import json
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import optax
from tqdm import tqdm
from typing import NamedTuple

from waymax import agents
from waymax import dynamics
from waymax import dataloader
from waymax import datatypes
from waymax import env as _env
from waymax import metrics as _metrics
from waymax import visualization

from .feature_extractor import KeyExtractor
from .state_preprocessing import ExtractObs
from .rnn_policy import ActorCriticRNN, ScannedRNN

from IGWaymax.dataset import N_TRAINING, N_VALIDATION, TRAJ_LENGTH, N_FILES
from IGWaymax.obs_mask import (
    SpeedConicObsMask,
    SpeedGaussianNoise,
    SpeedUniformNoise,
    ZeroMask,
    RandomMasking,
    GaussianNoise,
    DistanceObsMask,
)
from IGWaymax.utils import (
    plot_observation_with_goal,
    plot_observation_with_heading,
    plot_simulator_state,
    DiscreteActionSpaceWrapper,
)


class Transition(NamedTuple):
    done: jnp.ndarray
    expert_action: jnp.array
    obs: jnp.ndarray


extractors = {"ExtractObs": ExtractObs}
feature_extractors = {"KeyExtractor": KeyExtractor}
obs_masks = {
    "ZeroMask": ZeroMask,
    "SpeedGaussianNoise": SpeedGaussianNoise,
    "SpeedUniformNoise": SpeedUniformNoise,
    "SpeedConicObsMask": SpeedConicObsMask,
    "RandomMasking": RandomMasking,
    "GaussianNoise": GaussianNoise,
    "DistanceObsMask": DistanceObsMask,
}


class make_eval:
    def __init__(self, config, env_config, val_dataset, params):
        self.config = config
        self.env_config = env_config

        # Device
        self.devices = jax.devices()
        print(f"Available devices: {self.devices}")

        # Params
        self.params = params

        # Postprocessing function
        self._post_process = functools.partial(
            dataloader.womd_factories.simulator_state_from_womd_dict,
            include_sdc_paths=config["include_sdc_paths"],
        )

        # VALIDATION DATASET
        self.val_dataset = val_dataset

        # Random key
        self.key = random.PRNGKey(self.config["key"])

        # DEFINE ENV
        if self.config["dynamics"] == "bicycle":
            self.wrapped_dynamics_model = dynamics.InvertibleBicycleModel()
        elif self.config["dynamics"] == "delta":
            self.wrapped_dynamics_model = dynamics.DeltaLocal()
        else:
            raise ValueError("Unknown dynamics")

        if config["discrete"]:
            action_space_dim = self.wrapped_dynamics_model.action_spec().shape[0]
            self.wrapped_dynamics_model = DiscreteActionSpaceWrapper(
                dynamics_model=self.wrapped_dynamics_model,
                bins=config["bins"] * np.ones(action_space_dim, dtype=int),
            )
            self.config["num_components"] = None
            self.config["num_action"] = config["bins"] + 1
        else:
            self.wrapped_dynamics_model = self.wrapped_dynamics_model
            self.config["num_components"] = 6
            self.config["num_action"] = None

        self.dynamics_model = _env.PlanningAgentDynamics(self.wrapped_dynamics_model)

        if config["IDM"]:
            sim_actors = [
                agents.IDMRoutePolicy(
                    is_controlled_func=lambda state: 1 - state.object_metadata.is_sdc
                )
            ]
        else:
            sim_actors = ()

        self.env = _env.PlanningAgentEnvironment(
            dynamics_model=self.wrapped_dynamics_model,
            config=env_config,
            sim_agent_actors=sim_actors,
        )

        # DEFINE EXPERT AGENT
        self.expert_agent = agents.create_expert_actor(self.dynamics_model)

        # DEFINE EXTRACTOR AND FEATURE_EXTRACTOR
        self.extractor = extractors[self.config["extractor"]](self.config)
        self.feature_extractor = feature_extractors[self.config["feature_extractor"]]
        self.feature_extractor_kwargs = self.config["feature_extractor_kwargs"]

        # DEFINE OBSERVABILITY MASK
        if "obs_mask" not in self.config.keys():
            self.config["obs_mask"] = None

        if self.config["obs_mask"]:
            self.obs_mask = obs_masks[self.config["obs_mask"]](
                **self.config["obs_mask_kwargs"]
            )
        else:
            self.obs_mask = None

        # SAVE HARD VALIDATION SCENARIO
        self.hard_scenario_id = []
        self.easy_scenario_id = []

    # SCHEDULER
    def linear_schedule(self, count):
        n_update_per_epoch = (
            N_TRAINING * self.config["num_files"] / N_FILES
        ) // self.config["num_envs"]
        n_epoch = jnp.array([count // n_update_per_epoch])
        frac = jnp.where(n_epoch <= 20, 1, 1 / (2 ** (n_epoch - 20)))
        return self.config["lr"] * frac

    def train(
        self,
    ):
        # INIT NETWORK
        self.action_space_dim = self.wrapped_dynamics_model.action_spec().shape[0]

        network = ActorCriticRNN(
            self.action_space_dim,
            self.dynamics_model.action_spec().minimum,
            self.dynamics_model.action_spec().maximum,
            feature_extractor_class=self.feature_extractor,
            feature_extractor_kwargs=self.feature_extractor_kwargs,
            discrete=self.config["discrete"],
            num_components=self.config["num_components"],
            num_action=self.config["num_action"],
        )

        feature_extractor_shape = self.feature_extractor_kwargs["final_hidden_layers"]

        if self.config["lr_scheduler"]:
            tx = optax.chain(
                optax.clip_by_global_norm(self.config["max_grad_norm"]),
                optax.adam(learning_rate=self.linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(self.config["max_grad_norm"]),
                optax.adam(self.config["lr"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=self.params,
            tx=tx,
        )

        jit_postprocess_fn = jax.jit(self._post_process)

        def gaussian_entropy(std):
            entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * std**2)
            entropy = entropy.sum(axis=(-2), keepdims=True)
            return entropy

        def categorical_entropy(probabilities):
            entropy = -jnp.sum(probabilities * jnp.log(probabilities), axis=-1)
            return entropy

        # UPDATE THE SIMULATOR FROM THE LOG
        def _log_step(cary, rng_extract):
            current_state, rng = cary

            done = current_state.is_done
            # Extract obs in SDC referential
            obs = datatypes.sdc_observation_from_state(
                current_state, roadgraph_top_k=self.config["roadgraph_top_k"]
            )

            # Mask
            if self.obs_mask is not None:
                if self.obs_mask.mask_per_step:
                    rng, rng_obs = jax.random.split(rng)
                else:
                    rng_obs = rng_extract
                obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)

            # Extract the features from the observation

            # rng, rng_extract = jax.random.split(rng)
            obsv = self.extractor(current_state, obs, rng_extract)

            transition = Transition(done, None, obsv)

            # Update the simulator state with the expert action (current action space)
            expert_action = self.expert_agent.select_action(
                state=current_state, params=None, rng=None, actor_state=None
            )

            # Patch bug in waymax (squeeze timestep dimension when using reset --> need squeezed timestep for update)
            current_timestep = current_state["timestep"]
            # Squeeze timestep dim
            current_state = dataclasses.replace(
                current_state, timestep=current_timestep[0]
            )

            current_state = self.env.step(current_state, expert_action.action)

            # Unsqueeze timestep dim
            current_state = dataclasses.replace(
                current_state, timestep=current_timestep + 1
            )

            # # Update the simulator with the log trajectory (continuous action space)
            # current_state = datatypes.update_state_by_log(current_state, num_steps=1)

            return (current_state, rng), transition

        def _loss_fn(params, init_rnn_state, log_traj_batch, traj_batch, rng):
            # Compute the rnn_state from the log on the first steps
            rnn_state, _, _, _, _, _ = network.apply(
                params, init_rnn_state, (log_traj_batch.obs, log_traj_batch.done)
            )
            # Compute the action for the rest of the trajectory
            _, action_dist, _, weights, _, actor_std = network.apply(
                params, rnn_state, (traj_batch.obs, traj_batch.done)
            )

            if self.config["discrete"]:
                action_dist_entropy = action_dist.entropy()
                cat_entropy = None
            else:
                probabilities = jnp.exp(jax.nn.log_softmax(weights))
                action_dist_entropy = (
                    (probabilities * gaussian_entropy(actor_std)).sum(axis=-1).mean()
                )
                cat_entropy = categorical_entropy(probabilities).mean()

            if self.config["loss"] == "logprob":
                log_prob = action_dist.log_prob(
                    traj_batch.expert_action.action.data[..., : self.action_space_dim]
                )
                total_loss = -log_prob.mean(axis=-1)

            elif self.config["loss"] == "mse":
                action = action_dist.sample(seed=rng)
                expert_action = traj_batch.expert_action.action.data[
                    ..., : self.action_space_dim
                ]
                total_loss = ((action - expert_action) ** 2).mean(axis=-1)
            else:
                raise ValueError("Unknown loss function")

            return total_loss, (action_dist_entropy, cat_entropy)

        def _mse_fn(params, init_rnn_state, log_traj_batch, traj_batch, rng):
            # Compute the rnn_state from the log on the first steps
            rnn_state, _, _, _, _, _ = network.apply(
                params, init_rnn_state, (log_traj_batch.obs, log_traj_batch.done)
            )
            # Compute the action for the rest of the trajectory
            _, action_dist, _, _, _, _ = network.apply(
                params, rnn_state, (traj_batch.obs, traj_batch.done)
            )

            action = action_dist.sample(seed=rng)
            expert_action = traj_batch.expert_action.action.data
            if self.config["discrete"]:
                action = self.wrapped_dynamics_model._discretizer.make_continuous(
                    action
                )
                expert_action = (
                    self.wrapped_dynamics_model._discretizer.make_continuous(
                        expert_action
                    )
                )
            mse = ((action - expert_action) ** 2).mean(axis=-1)

            return mse

        # Evaluate
        def _eval_scenario(train_state, scenario, rng):
            init_rnn_state_eval = ScannedRNN.initialize_carry(
                (scenario.shape[0], feature_extractor_shape)
            )

            rng, rng_extract = jax.random.split(rng)
            # Compute the rnn_state on first self.env.config.init_steps from the log trajectory
            (current_state, rng), log_traj_batch = jax.lax.scan(
                _log_step,
                (scenario, rng),
                rng_extract[None].repeat(self.env.config.init_steps - 1, axis=0),
                self.env.config.init_steps - 1,
            )

            rnn_state, _, _, _, _, _ = network.apply(
                train_state.params,
                init_rnn_state_eval,
                (log_traj_batch.obs, log_traj_batch.done),
            )

            def extand(x):
                if isinstance(x, jnp.ndarray):
                    return x[jnp.newaxis, ...]
                else:
                    return x

            def _eval_step(cary, rng_extract):
                current_state, rnn_state, rng = cary

                done = current_state.is_done

                # Extract obs in SDC referential
                obs = datatypes.sdc_observation_from_state(
                    current_state, roadgraph_top_k=self.config["roadgraph_top_k"]
                )

                # Mask
                if self.obs_mask is not None:
                    if self.obs_mask.mask_per_step:
                        rng, rng_obs = jax.random.split(rng)
                    else:
                        rng_obs = rng_extract
                    obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)

                # Extract the features from the observation
                # rng, rng_extract = jax.random.split(rng)
                obsv = self.extractor(current_state, obs, rng_extract)

                # Sample action and update scenario
                rnn_state, action_dist, _, _, _, _ = network.apply(
                    train_state.params,
                    rnn_state,
                    (jax.tree_map(extand, obsv), done[jnp.newaxis, ...]),
                )
                rng, rng_sample = jax.random.split(rng)
                action_data = action_dist.sample(seed=rng_sample).squeeze(0)
                action = datatypes.Action(
                    data=action_data,
                    valid=jnp.ones((action_data.shape[0], 1), dtype="bool"),
                )

                # Patch bug in waymax (squeeze timestep dimension when using reset --> need squeezed timestep for update)
                current_timestep = current_state["timestep"]
                # Squeeze timestep dim
                current_state = dataclasses.replace(
                    current_state, timestep=current_timestep[0]
                )

                current_state = self.env.step(current_state, action)

                # Unsqueeze timestep dim
                current_state = dataclasses.replace(
                    current_state, timestep=current_timestep + 1
                )

                metric = self.env.metrics(current_state)

                return (current_state, rnn_state, rng), metric

            # Collect trajectory and update scenario with log
            def _env_step_expert(cary, rng_extract):
                current_state, rng = cary

                done = current_state.is_done

                # Extract obs in SDC referential
                obs = datatypes.sdc_observation_from_state(
                    current_state, roadgraph_top_k=self.config["roadgraph_top_k"]
                )

                expert_action = self.expert_agent.select_action(
                    state=current_state, params=None, rng=None, actor_state=None
                )

                # Mask
                if self.obs_mask is not None:
                    if self.obs_mask.mask_per_step:
                        rng, rng_obs = jax.random.split(rng)
                    else:
                        rng_obs = rng_extract
                    obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)

                # Extract the features from the observation
                # rng, rng_extract = jax.random.split(rng)
                obsv = self.extractor(current_state, obs, rng_extract)

                transition = Transition(done, expert_action, obsv)

                # Update the simulator state with the expert action (current action space)
                expert_action = self.expert_agent.select_action(
                    state=current_state, params=None, rng=None, actor_state=None
                )

                # Patch bug in waymax (squeeze timestep dimension when using reset --> need squeezed timestep for update)
                current_timestep = current_state["timestep"]
                # Squeeze timestep dim
                current_state = dataclasses.replace(
                    current_state, timestep=current_timestep[0]
                )

                current_state = self.env.step(current_state, expert_action.action)

                # Unsqueeze timestep dim
                current_state = dataclasses.replace(
                    current_state, timestep=current_timestep + 1
                )

                # # Update the simulator state with the log trajectory (continuous action space)
                # current_state = datatypes.update_state_by_log(current_state, num_steps=1)

                return (current_state, rng), transition

            def _eval_step_gif(cary, rng_extract):
                current_state, rnn_state, rng = cary

                done = current_state.is_done

                # Extract obs in SDC referential
                obs = datatypes.sdc_observation_from_state(
                    current_state, roadgraph_top_k=self.config["roadgraph_top_k"]
                )

                # Mask
                if self.obs_mask is not None:
                    if self.obs_mask.mask_per_step:
                        rng, rng_obs = jax.random.split(rng)
                    else:
                        rng_obs = rng_extract
                    obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)

                # Extract the features from the observation
                # rng, rng_extract = jax.random.split(rng)
                obsv = self.extractor(current_state, obs, rng_extract)

                # Generate image

                reduced_sdc_obs = jax.tree_map(lambda x: x[0, ...], obs)  # Unbatch
                list_features = self.config["feature_extractor_kwargs"]["keys"]
                list_features = np.concatenate(
                    [
                        [item.strip() for item in feature.strip("[]").split(",")]
                        for feature in list_features
                    ]
                ).tolist()
                if ("noisy_proxy_goal" in list_features) or (
                    "proxy_goal" in list_features
                ):
                    if "proxy_goal" in list_features:
                        goal = obsv["proxy_goal"][0]
                    elif "noisy_proxy_goal" in list_features:
                        goal = obsv["noisy_proxy_goal"][0]

                    ego_img = plot_observation_with_goal(
                        reduced_sdc_obs, obj_idx=0, goal=goal
                    )

                elif "heading" in list_features:
                    ego_img = plot_observation_with_heading(
                        reduced_sdc_obs, obj_idx=0, heading=obsv["heading"].squeeze()
                    )
                else:
                    ego_img = visualization.plot_observation(
                        jax.tree_map(lambda x: x[0], obs), obj_idx=0
                    )

                global_img = plot_simulator_state(
                    jax.tree_map(lambda x: x[0], current_state)
                )

                # Sample action and update scenario
                rnn_state, action_dist, _, _, _, _ = network.apply(
                    train_state.params,
                    rnn_state,
                    (jax.tree_map(extand, obsv), done[jnp.newaxis, ...]),
                )
                rng, rng_sample = jax.random.split(rng)
                action_data = action_dist.sample(seed=rng_sample).squeeze(0)
                action = datatypes.Action(
                    data=action_data,
                    valid=jnp.ones((action_data.shape[0], 1), dtype="bool"),
                )

                # Patch bug in waymax (squeeze timestep dimension when using reset --> need squeezed timestep for update)
                current_timestep = current_state["timestep"]
                # Squeeze timestep dim
                current_state = dataclasses.replace(
                    current_state, timestep=current_timestep[0]
                )

                current_state = self.env.step(current_state, action)

                # Unsqueeze timestep dim
                current_state = dataclasses.replace(
                    current_state, timestep=current_timestep + 1
                )

                metric = self.env.metrics(current_state)

                return (current_state, rnn_state, rng), (ego_img, global_img, metric)

            ego_imgs = []
            global_imgs = []
            if not self.config["GIF"]:
                rng, rng_step = jax.random.split(rng)
                _, scenario_metrics = jax.lax.scan(
                    _eval_step,
                    (current_state, rnn_state, rng_step),
                    rng_extract[None].repeat(
                        TRAJ_LENGTH - self.env.config.init_steps, axis=0
                    ),
                    TRAJ_LENGTH - self.env.config.init_steps,
                )

                _, traj_batch = jax.lax.scan(
                    _env_step_expert,
                    (current_state, rng_step),
                    rng_extract[None].repeat(
                        TRAJ_LENGTH - self.env.config.init_steps, axis=0
                    ),
                    TRAJ_LENGTH - self.env.config.init_steps,
                )

                rng, rng_loss = jax.random.split(rng)
                val_loss, _ = _loss_fn(
                    train_state.params,
                    init_rnn_state_eval,
                    log_traj_batch,
                    traj_batch,
                    rng_loss,
                )
                val_loss_metric = _metrics.MetricResult(
                    value=jnp.array([val_loss]),
                    valid=jnp.ones_like(jnp.array([val_loss]), dtype=bool),
                )
                scenario_metrics["loss"] = val_loss_metric

                rng, rng_mse = jax.random.split(rng)
                val_mse = _mse_fn(
                    train_state.params,
                    init_rnn_state_eval,
                    log_traj_batch,
                    traj_batch,
                    rng_mse,
                )
                val_mse_metric = _metrics.MetricResult(
                    value=jnp.array([val_mse]),
                    valid=jnp.ones_like(jnp.array([val_mse]), dtype=bool),
                )
                scenario_metrics["mse"] = val_mse_metric

            else:
                scenario_metrics = []

                # Loop over timesteps
                for _ in range(TRAJ_LENGTH - self.env.config.init_steps):
                    # Call _eval_step for each timestep
                    (current_state, rnn_state, rng), (ego_img, global_img, metric) = (
                        _eval_step_gif((current_state, rnn_state, rng), rng_extract)
                    )
                    scenario_metrics.append(metric)
                    ego_imgs.append(ego_img)
                    global_imgs.append(global_img)

                # Combine metrics for all timesteps
                scenario_metrics = jax.tree_map(
                    lambda *args: jnp.stack(args), *scenario_metrics
                )

            return ego_imgs, global_imgs, scenario_metrics

        jit_eval_scenario = jax.jit(_eval_scenario)

        # EVALUATION LOOP
        def _evaluate_epoch(train_state, rng):
            all_metrics = {
                "log_divergence": [],
                "max_log_divergence": [],
                "overlap_rate": [],
                "overlap": [],
                "offroad_rate": [],
                "offroad": [],
                "loss": [],
                "mse": [],
            }

            all_metrics_inter = {
                "log_divergence": [],
                "max_log_divergence": [],
                "overlap_rate": [],
                "overlap": [],
                "offroad_rate": [],
                "offroad": [],
                "loss": [],
                "mse": [],
            }

            all_metrics_hard = {
                "log_divergence": [],
                "max_log_divergence": [],
                "overlap_rate": [],
                "overlap": [],
                "offroad_rate": [],
                "offroad": [],
                "loss": [],
                "mse": [],
            }

            all_metrics_easy = {
                "log_divergence": [],
                "max_log_divergence": [],
                "overlap_rate": [],
                "overlap": [],
                "offroad_rate": [],
                "offroad": [],
                "loss": [],
                "mse": [],
            }

            num_data = {"all": 0, "inter": 0, "hard": 0, "easy": 0}
            t = 0
            for data in tqdm(
                self.val_dataset.as_numpy_iterator(),
                desc="Validation",
                total=N_VALIDATION // self.config["num_envs_eval"] + 1,
            ):
                t += 1
                all_scenario_metrics = {}

                scenario = jit_postprocess_fn(data)

                # Scenario does not contain the SDC
                pass_condition = not jnp.any(scenario.object_metadata.is_sdc)

                if pass_condition:
                    pass
                else:
                    rng, rng_eval = jax.random.split(rng)
                    if self.config["GIF"]:
                        ego_imgs, global_imgs, scenario_metrics = _eval_scenario(
                            train_state, scenario, rng_eval
                        )

                        ego_frames = [Image.fromarray(img) for img in ego_imgs]
                        tag = ""
                        ego_frames[0].save(
                            os.path.join(
                                "../animation/",
                                self.config["log_folder"][
                                    len(self.config["folder_name"]) + 1 :
                                ],
                                f"ex_{t}{tag}_ego.gif",
                            ),
                            save_all=True,
                            append_images=ego_frames[1:],
                            duration=100,
                            loop=0,
                        )

                        global_frames = [Image.fromarray(img) for img in global_imgs]
                        tag = ""
                        global_frames[0].save(
                            os.path.join(
                                "../animation/",
                                self.config["log_folder"][
                                    len(self.config["folder_name"]) + 1 :
                                ],
                                f"ex_{t}{tag}_global.gif",
                            ),
                            save_all=True,
                            append_images=global_frames[1:],
                            duration=100,
                            loop=0,
                        )
                    else:
                        _, _, scenario_metrics = jit_eval_scenario(
                            train_state, scenario, rng_eval
                        )

                    overlap = (
                        scenario_metrics["overlap"].value
                        * scenario_metrics["overlap"].valid
                    ).sum(axis=0) / scenario_metrics["overlap"].valid.sum(axis=0)

                    num_data["all"] += scenario.shape[0]
                    num_data["inter"] += jnp.any(
                        scenario.object_metadata.objects_of_interest, axis=1
                    ).sum()
                    num_data["hard"] += (overlap > 0.0).sum()
                    num_data["easy"] += (overlap == 0).sum()
                    if jnp.any(overlap > 0.0):
                        self.hard_scenario_id.append(data["scenario/id"][overlap > 0.0])
                    if jnp.any(overlap == 0):
                        self.easy_scenario_id.append(data["scenario/id"][overlap == 0])

                    for key, value in scenario_metrics.items():
                        if jnp.any(value.valid):
                            all_metrics[key].append(value.value[value.valid].mean())
                            all_scenario_metrics[key] = value.value[value.valid].mean()
                            ndim = value.value.ndim
                            if jnp.any(overlap > 0.0):
                                all_metrics_hard[key].append(
                                    (
                                        (value.value * value.valid).sum(
                                            axis=tuple(range(ndim - 1))
                                        )
                                        / value.valid.sum(axis=tuple(range(ndim - 1)))
                                    )[overlap > 0.0].mean()
                                )
                            if jnp.any(overlap == 0):
                                all_metrics_easy[key].append(
                                    (
                                        (value.value * value.valid).sum(
                                            axis=tuple(range(ndim - 1))
                                        )
                                        / value.valid.sum(axis=tuple(range(ndim - 1)))
                                    )[overlap == 0].mean()
                                )
                        if jnp.any(scenario.object_metadata.objects_of_interest):
                            has_inter = jnp.any(
                                scenario.object_metadata.objects_of_interest, axis=1
                            ) * jnp.ones_like(value.valid, dtype=bool)
                            all_metrics_inter[key].append(
                                value.value[value.valid & has_inter].mean()
                            )

                    key = "max_log_divergence"
                    value = scenario_metrics["log_divergence"]
                    if jnp.any(value.valid):
                        all_metrics[key].append(value.value.max(axis=0).mean())
                        all_scenario_metrics[key] = value.value.max(axis=0).mean()
                        if jnp.any(overlap > 0.0):
                            all_metrics_hard[key].append(
                                (value.value * value.valid)
                                .max(axis=0)[overlap > 0.0]
                                .mean()
                            )
                        if jnp.any(overlap == 0):
                            all_metrics_easy[key].append(
                                (value.value * value.valid)
                                .max(axis=0)[overlap == 0]
                                .mean()
                            )
                    if jnp.any(scenario.object_metadata.objects_of_interest):
                        has_inter = jnp.any(
                            scenario.object_metadata.objects_of_interest, axis=1
                        )
                        all_metrics_inter[key].append(
                            (value.value.max(axis=0) * has_inter).sum()
                            / has_inter.sum()
                        )

                    key = "overlap_rate"
                    value = scenario_metrics["overlap"]
                    if jnp.any(value.valid):
                        all_metrics[key].append(jnp.any(value.value, axis=0).mean())
                        all_scenario_metrics[key] = jnp.any(value.value, axis=0).mean()
                        if jnp.any(overlap > 0.0):
                            all_metrics_hard[key].append(
                                jnp.any(value.value, axis=0)[overlap > 0.0].mean()
                            )
                        if jnp.any(overlap == 0):
                            all_metrics_easy[key].append(
                                jnp.any(value.value, axis=0)[overlap == 0].mean()
                            )
                    if jnp.any(scenario.object_metadata.objects_of_interest):
                        has_inter = jnp.any(
                            scenario.object_metadata.objects_of_interest, axis=1
                        )
                        all_metrics_inter[key].append(
                            (jnp.any(value.value, axis=0) * has_inter).sum()
                            / has_inter.sum()
                        )

                    key = "offroad_rate"
                    value = scenario_metrics["offroad"]
                    if jnp.any(value.valid):
                        all_metrics[key].append(jnp.any(value.value, axis=0).mean())
                        all_scenario_metrics[key] = jnp.any(value.value, axis=0).mean()
                        if jnp.any(overlap > 0.0):
                            all_metrics_hard[key].append(
                                jnp.any(value.value, axis=0)[overlap > 0.0].mean()
                            )
                        if jnp.any(overlap == 0):
                            all_metrics_easy[key].append(
                                jnp.any(value.value, axis=0)[overlap == 0].mean()
                            )
                    if jnp.any(scenario.object_metadata.objects_of_interest):
                        has_inter = jnp.any(
                            scenario.object_metadata.objects_of_interest, axis=1
                        )
                        all_metrics_inter[key].append(
                            (jnp.any(value.value, axis=0) * has_inter).sum()
                            / has_inter.sum()
                        )

                    if self.config["GIF"]:
                        folder = os.path.join(
                            os.path.join(
                                "../animation/",
                                self.config["log_folder"][
                                    len(self.config["folder_name"]) + 1 :
                                ],
                                f"ex_{t}.json",
                            )
                        )
                        with open(folder, "w") as json_file:
                            json.dump(
                                jax.tree_map(lambda x: x.item(), all_scenario_metrics),
                                json_file,
                                indent=4,
                            )

            return train_state, (
                all_metrics,
                all_metrics_inter,
                all_metrics_hard,
                all_metrics_easy,
                num_data,
            )

        metrics = {}
        rng = self.key
        for epoch in range(self.config["num_epochs"]):
            metrics[epoch] = {}

            # Validation
            rng, rng_eval = jax.random.split(rng)
            (
                _,
                (
                    val_metric,
                    val_metric_inter,
                    val_metric_hard,
                    val_metric_easy,
                    num_data,
                ),
            ) = _evaluate_epoch(train_state, rng_eval)
            metrics[epoch]["validation"] = val_metric

            val_message = f"Epoch | {epoch} | Val (full)| "
            for key, value in val_metric.items():
                val_message += f" {key} | {jnp.array(value).mean():.4f} | "

            print(val_message, f"{num_data['all']} data")

            val_message = f"Epoch | {epoch} | Val (inter)| "
            for key, value in val_metric_inter.items():
                val_message += f" {key} | {jnp.array(value).mean():.4f} | "

            print(val_message, f"{num_data['inter']} data")

            val_message = f"Epoch | {epoch} | Val (hard)| "
            for key, value in val_metric_hard.items():
                val_message += f" {key} | {jnp.array(value).mean():.4f} | "

            print(val_message, f"{num_data['hard']} data")

            val_message = f"Epoch | {epoch} | Val (easy)| "
            for key, value in val_metric_easy.items():
                val_message += f" {key} | {jnp.array(value).mean():.4f} | "

            print(val_message, f"{num_data['easy']} data")

        return {
            "train_state": train_state,
            "metrics": metrics,
            "hard_scenario_id": jnp.concatenate(self.hard_scenario_id),
            "easy_scenario_id": jnp.concatenate(self.easy_scenario_id),
        }
