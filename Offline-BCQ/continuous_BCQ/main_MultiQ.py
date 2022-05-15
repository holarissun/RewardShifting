import argparse
import gym
import numpy as np
import os
import torch

import BCQ_Dual
import DDPG
import utils
import torch.nn.functional as F

# Handles interactions with the environment, i.e. train behavioral or generate buffer
def interact_with_environment(env, state_dim, action_dim, max_action, device, args):
    # For saving files
    # setting = f"{args.env}_{args.seed}"
    setting = f"{args.env}_0"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Initialize and load policy
    policy = DDPG.DDPG(state_dim, action_dim, max_action, device)#, args.discount, args.tau)
    if args.generate_buffer: policy.load(f"./models/behavioral_{setting}")

    # Initialize buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device, max_size = args.max_buffer_size)

    evaluations = []

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # Interact with the environment for max_timesteps
    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action with noise
        if (
            (args.generate_buffer and np.random.uniform(0, 1) < args.rand_action_p) or 
            (args.train_behavioral and t < args.start_timesteps)
        ):
            action = env.action_space.sample()
        else: 
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.gaussian_std, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if args.train_behavioral and t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if args.train_behavioral and (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/behavioral_{setting}", evaluations)
            policy.save(f"./models/behavioral_{setting}")

    # Save final policy
    if args.train_behavioral:
        policy.save(f"./models/behavioral_{setting}")

    # Save final buffer and performance
    else:
        evaluations.append(eval_policy(policy, args.env, args.seed))
        np.save(f"./results/buffer_performance_{setting}", evaluations)
        replay_buffer.save(f"./buffers/{buffer_name}")


# Trains BCQ offline
def train_BCQ(state_dim, action_dim, max_action, device, args):
    
    if args.d4rl_dataset is not None:
        # For saving files
        #setting = f"{args.env}_{args.seed}"
        setting = f"{args.env}_0"
        buffer_name = f"{args.buffer_name}_{setting}_{args.d4rl_dataset}"

        # Initialize policy
        policy = BCQ_Dual.BCQ_Dual(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi, args.conservative_factor, args.hidden_num1, args.hidden_num2)
        
        # Load buffer
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device, max_size = args.max_buffer_size)
        replay_buffer.load_from_d4rl(args.d4rl_dataset)
        
    else:
        # For saving files
        #setting = f"{args.env}_{args.seed}"
        setting = f"{args.env}_0"
        buffer_name = f"{args.buffer_name}_{setting}"

        # Initialize policy
        policy = BCQ_Dual.BCQ_Dual(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi, args.conservative_factor, args.hidden_num1, args.hidden_num2)

        # Load buffer
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device, max_size = args.max_buffer_size)
        replay_buffer.load(f"./buffers/{buffer_name}")

    evaluations = []
    episode_num = 0
    done = True 
    training_iters = 0

    while training_iters < args.max_timesteps: 
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        
        evaluations.append(eval_policy(policy, args.env, args.seed))
        np.save(f"./results/BCQ_Dual_{setting}", evaluations)
        
        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}")
        
        '''there might exsit several different settings TOBEDONE later: Action Randomization/ State + Action Randomization'''
        
        
        # Random States + Random Action
        
        random_states = torch.as_tensor((np.random.randn(100, state_dim)*3).clip(-10,10)).float().to(device)
        random_actions = torch.as_tensor((np.random.uniform(-max_action, max_action, (100, action_dim)))).float().to(device)
        
        # Sampled States
        state, action, next_state, reward, not_done = replay_buffer.sample(100)

        # Critic Training
        with torch.no_grad():
            current_Q1, current_Q2 = policy.critic_target(state, action)
            current_Q1_2, current_Q2_2 = policy.critic_target2(state, action)
            
            rand_s_a_current_Q1, rand_s_a_current_Q2 = policy.critic_target(random_states, random_actions)
            rand_s_a_current_Q1_2, rand_s_a_current_Q2_2 = policy.critic_target2(random_states, random_actions)

            rand_a_current_Q1, rand_a_current_Q2 = policy.critic_target(state, random_actions)
            rand_a_current_Q1_2, rand_a_current_Q2_2 = policy.critic_target2(state, random_actions)
            

            print('difference between Q1 Q1_2:', F.mse_loss(current_Q1, current_Q1_2).cpu().detach().numpy())
            print('difference between rand_s_a Q1 Q1_2:', F.mse_loss(rand_s_a_current_Q1, rand_s_a_current_Q1_2).cpu().detach().numpy())
            print('difference between rand_a Q1 Q1_2:', F.mse_loss(rand_a_current_Q1, rand_a_current_Q1_2).cpu().detach().numpy())
            
            print('difference between Q2 Q2_2:', F.mse_loss(current_Q2, current_Q2_2).cpu().detach().numpy())
            print('difference between rand_s_a Q2 Q2_2:', F.mse_loss(rand_s_a_current_Q2, rand_s_a_current_Q2_2).cpu().detach().numpy())
            print('difference between rand_a Q2 Q2_2:', F.mse_loss(rand_a_current_Q2, rand_a_current_Q2_2).cpu().detach().numpy())

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v3")               # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Robust")          # Prepends name to filename
    parser.add_argument("--eval_freq", default=1e3, type=float)     # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used before training behavioral
    parser.add_argument("--rand_action_p", default=0.3, type=float) # Probability of selecting random action during batch generation
    parser.add_argument("--gaussian_std", default=0.3, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    parser.add_argument("--batch_size", default=100, type=int)      # Mini batch size for networks
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--lmbda", default=0.75)                    # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi", default=0.05)                      # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral (DDPG)
    parser.add_argument("--generate_buffer", action="store_true")   # If true, generate buffer
    parser.add_argument("--conservative_factor",default=0.0, type=float) # update reward, for conservative update
    parser.add_argument("--gpu", default = None, type = int)
    parser.add_argument("--hidden_num1", default = 400, type = int)
    parser.add_argument("--hidden_num2", default = 300, type = int)
    parser.add_argument("--d4rl_dataset", default = None)           # If use d4rl dataset, replace this with the dataset name
    parser.add_argument("--max_buffer_size", default=100000, type=int)
    args = parser.parse_args()
    
    if args.gpu>=0:
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'
    print("---------------------------------------")
    if args.train_behavioral:
        print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
    elif args.generate_buffer:
        print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
    else:
        print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if args.train_behavioral and args.generate_buffer:
        print("Train_behavioral and generate_buffer cannot both be true.")
        exit()

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    if not os.path.exists("./buffers"):
        os.makedirs("./buffers")

    env = gym.make(args.env)

    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    

    if args.train_behavioral or args.generate_buffer:
        interact_with_environment(env, state_dim, action_dim, max_action, device, args)
    else:
        train_BCQ(state_dim, action_dim, max_action, device, args)
