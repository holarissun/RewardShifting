# 🙋 How To Design A Reward Function For Your Reinforcement Learning Task (In Value-Based RL)? 

- To boost exploration, you should use negative rewards, such that the agent will visit more unvisited state-action pairs.
- To boost exploitation, you should use positive rewards, such that the agent will repeatedly visit previously visited state-action pairs.

### Our paper provides a detailed analysis of how reward design affects the learning process.
#### This repo is related to the topic of
  - Reward Design in Deep RL
  - Reward Design for Better Exploration
  - Offline-RL (conservation via reward shifting)
  - Value-Based Deep-RL
    

# 🚀 Let us Exploit Reward Shifting in Value-Based Deep-RL!

### 🌐 [Project Page](https://sites.google.com/view/rewardshaping)

### Key Insight: A positive reward shifting leads to conservative exploitation, and a negative reward shifting leads to curiosity-driven exploration.


## 🏋️ Reproduction & Basic Usage: 

To reproduce our results, please follow instructions in each folder. 
Actually, the easiest way of reproduction is to play with reward shifting!

🧑🏻‍💻 In your tasks with value-based DRL, please just try to add a line right after the line of interaction with your environment, e.g., 

```next_s, r, done, info = env.step(a)```

```r = r + args.shifting_constant```

❕Don't forget to remove such a shift in evaluating your policy :) 

## 💡 Potential Ideas 


Here are several potential extensions of our work:
- Theoretically, the guidance of choosing shifting constant values.
- Methodologically, the choice of ensemble bias values
- Empirically, combining upper and lower bound (as non-linear combination) with Thompson Sampling for better exploration.
- Other linear reward shaping, e.g., with non-trivial scaling factor **k**.

## 📝 BibTex

```
@article{sun2022exploit,
  title={Exploit Reward Shifting in Value-Based Deep-RL: Optimistic Curiosity-Based Exploration and Conservative Exploitation via Linear Reward Shaping},
  author={Sun, Hao and Han, Lei and Yang, Rui and Ma, Xiaoteng and Guo, Jian and Zhou, Bolei},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={37719--37734},
  year={2022}
}
