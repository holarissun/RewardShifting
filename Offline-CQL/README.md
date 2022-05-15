# CQL


## Vanilla CQL Repo:

Code for Conservative Q-Learning for Offline Reinforcement Learning (https://arxiv.org/abs/2006.04779)

In this repository we provide code for CQL algorithm described in the paper linked above. We provide code in two sub-directories: `atari` containing code for Atari experiments and `d4rl` containing code for D4RL experiments. Due to changes in the datasets in D4RL, we expect some changes in CQL performance on the new D4RL datasets and we will soon provide a table with new performance numbers for CQL here in this README. We will continually keep updating the numbers here.

If you find this repository useful for your research, please cite:

```
@article{kumar2020conservative,
  author       = {Aviral Kumar and Aurick Zhou and George Tucker and Sergey Levine},
  title        = {Conservative Q-Learning for Offline Reinforcement Learning},
  conference   = {arXiv Pre-print},
  url          = {https://arxiv.org/abs/2006.04779},
}
```

## D4RL Experiments
Our code is built off of [rlkit](https://github.com/vitchyr/rlkit). Please install the conda environment for rlkit while making sure to install `torch>=1.1.0`. Please install [d4rl](https://github.com/rail-berkeley/d4rl). Code for the CQL algorithm is present in `rlkit/torch/sac/cql.py`. After this, for running CQL on the MuJoCo environments, run:

```
python examples/cql_mujoco_new.py --env=<d4rl-mujoco-env-with-version e.g. hopper-medium-v0>
        --policy_lr=1e-4 --seed=10 --lagrange_thresh=-1.0
        --min_q_weight=(5.0 or 10.0) --gpu=<gpu-id> --min_q_version=3
```

In terms of parameters, we have found `min_q_weight=5.0` or `min_q_weight=10.0` along with `policy_lr=1e-4` or `policy_lr=3e-4` to work reasonably fine for the Gym MuJoCo tasks. These parameters are slightly different from the paper (which will be updated soon) due to differences in the D4RL datasets. For sample performance numbers (final numbers to be updated soon), hopper-medium acheives ~3000 return, and hopper-medium-exprt obtains ~1300 return at the end of 500k gradient steps. To run `CQL(\rho)` [i.e. without the importance sampling], set `min_q_version=2`.


## Experiments Based on CQL:
After setting up CQL and D4RL, our experiments can be reproduced by running:

```
python examples/cql_mujoco_new.py --conservative_factor=50.0 --env=walker2d-medium-v2 --policy_lr=1e-4 --seed=10 --lagrange_thresh=5.0 --min_q_weight=5.0 --gpu=0 --min_q_version=3
```



