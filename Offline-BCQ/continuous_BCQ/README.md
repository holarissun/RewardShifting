# Positive Reward Shifting for Conservative Exploitation


## Our Implementation is Based on BCQ:
Batch-Constrained deep Q-learning (BCQ) is the first batch deep reinforcement learning, an algorithm which aims to learn offline without interactions with the environment.

BCQ was first introduced in our [ICML 2019 paper](https://arxiv.org/abs/1812.02900) which focused on continuous action domains. A discrete-action version of BCQ was introduced in a followup [Deep RL workshop NeurIPS 2019 paper](https://arxiv.org/abs/1910.01708). Code for each of these algorithms can be found under their corresponding folder.


Code for Batch-Constrained deep Q-Learning (BCQ). If you use our code please cite the [paper](https://arxiv.org/abs/1812.02900).

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym).
Networks are trained using [PyTorch 1.4](https://github.com/pytorch/pytorch) and Python 3.6.

### Overview

If you are interested in reproducing some of the results from the paper, a behavioral policy (DDPG) needs to be trained by running:
```
python main.py --train_behavioral --gaussian_std 0.1
```
This will save the PyTorch model. A new buffer, corresponding to the "imperfect demonstrations" task, can then be collected by running:
```
python main.py --generate_buffer --max_timesteps 100000
```
Or for the "imitation" task by running:
```
python main.py --generate_buffer --gaussian_std 0.0 --rand_action_p 0.0
```
Finally train BCQ by running:
```
python main.py
```

Settings can be adjusted with different arguments to main.py.

DDPG was updated to learn more consistently. Additionally, with version updates to Python, PyTorch and environments, results may not correspond exactly to the paper. Some people have reported instability using the v2 environments, so sticking with v3 is preferred.

### Bibtex

```
@inproceedings{fujimoto2019off,
  title={Off-Policy Deep Reinforcement Learning without Exploration},
  author={Fujimoto, Scott and Meger, David and Precup, Doina},
  booktitle={International Conference on Machine Learning},
  pages={2052--2062},
  year={2019}
}
```

```
@article{fujimoto2019benchmarking,
  title={Benchmarking Batch Deep Reinforcement Learning Algorithms},
  author={Fujimoto, Scott and Conti, Edoardo and Ghavamzadeh, Mohammad and Pineau, Joelle},
  journal={arXiv preprint arXiv:1910.01708},
  year={2019}
}
```


## How to Run:

After generating the replay buffer as described above (and also d4rl, if using d4rl dataset), run either

```
python main.py --gpu 0 --conservative_factor 1.0/8.0/20.0/50.0 --buffer_name d4rl/BCQ
```
or

```
python main_MultiQ.py --gpu 0 --conservative_factor 1.0/8.0/20.0/50.0 --buffer_name d4rl/BCQ --d4rl_dataset hopper-expert-v2
```

the latter script (main_MultiQ.py) extends the primal BCQ and enables early-stopping for offline-RL problems, yet this is a little bit deviated from our key insight. All results reported in our paper can be reproduced by just running main.py ,calling the primal BCQ.py, rather than our improved Dual Q function structure.





