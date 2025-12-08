# IGDrivSim: A Benchmark for the Imitation Gap in Autonomous Driving 🚗 (IROS, 2025)

This repository provides the code and experiments for **IGDrivSim**, a benchmark built on top of the [Waymax](https://github.com/waymo-research/waymax.git) simulator, designed to investigate the **imitation gap** in learning autonomous driving policies from human expert demonstrations.

📄 For more details, please refer to our paper: [IGDrivSim: A Benchmark for the Imitation Gap in Autonomous Driving](https://web3.arxiv.org/abs/2411.04653).

## Key Features ✨
- **Imitation Gap Study**: Examines how discrepancies between human perception and self-driving sensors affect learning.
- **Benchmark Tasks**: Simulated driving tasks with full observability, noisy data, and restricted fields of view.
- **Baselines**: Includes evaluation of behavioral cloning and a combination of behavioral cloning with reinforcement learning (PPO) trained with Jax on Waymax.

## Installation 🛠️

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/IGDrivSim.git
   cd IGDrivSim

2. Installation

   a. Configure access to Waymo Open Motion Dataset

   A simple way to configure access via command line is the following:

   - Apply for Waymo Open Dataset access.
   - Install gcloud CLI.
   - Run the following commands:
      ```bash
      gcloud auth login <your_email> # with the same email used for step 1.
      gcloud auth application-default login

   b. Install Required Packages:

   - With **Docker**: Create a Docker image and build a container using the provided Dockerfile.

   - With pip:
      ```bash
      cd docker
      pip install -r requirements.txt
      python -m pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

   Please refer to [JAX](https://github.com/google/jax#installation) for specific instructions on how to setup JAX with GPU/CUDA support if needed.

## Run experiments 🔄

The experiment configurations are located in scripts/config.

To reproduce the results run:
   ```bash
   python scripts/train_bc_rl.py -w_bc 1. -w_rl 0. -conf CircularMasking --radius 4
   ```

## Visualize Partial Observability 📊

To visualize the different partial observability modes in IGDrivSim, open the notebook:
   ```bash
   notebook limited_obs.ipynb
   ```

## Acknowledgements 📚

* The utility functions are adapted from the [Waymax](https://github.com/waymo-research/waymax.git) repo.
* The PPO update implementation is based on [PureJaxRL](https://github.com/luchris429/purejaxrl).

## Citation 🔖

```bibtex
@misc{grislain2024igdrivsimbenchmarkimitationgap,
      title={IGDrivSim: A Benchmark for the Imitation Gap in Autonomous Driving},
      author={Clémence Grislain and Risto Vuorio and Cong Lu and Shimon Whiteson},
      year={2024},
      eprint={2411.04653},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2411.04653},
}
```
