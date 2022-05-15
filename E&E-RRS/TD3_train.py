from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import os
import torch.nn.functional as F
import utils
import TD3
from dm_control import suite
def parse_args():
    parser = ArgumentParser(description='train args')
    #parser.add_argument('config', help='config file path')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-en','--env_name', type=str, default=None)
    parser.add_argument('-mt','--max_timesteps',type=int, default= 1e6)
    parser.add_argument('-a','--alias', type=str, default=None)
    parser.add_argument('-sr','--sample_reuse', type=int, default=None)
    parser.add_argument('-b','--batch_size', type=int, default=None)
    parser.add_argument('-r','--repeat',type=int,default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    ENV_NAME = args.env_name
    alias = f'ENV_{args.env_name}_SampleReuse_{args.sample_reuse}_BS_{args.batch_size}_EXTAlias_{args.alias}'

    

    print('all benchmarking environments:')
    print('      ',suite.BENCHMARKING)
    print('\n now evaluating: \n       ', ENV_NAME)


    


    def eval_policy(policy, eval_episodes=10):
        eval_env = suite.load(ENV_NAME.split('-')[0], ENV_NAME.split('-')[1])

        avg_reward = 0.
        for _ in range(eval_episodes):
            env_stat = eval_env.reset()
            state, done = np.hstack(list(env_stat.observation.values())), False
            while not done:
                action = policy.select_action(np.array(state))

                eval_env_stat = eval_env.step(action)

                state, reward = np.hstack(list(eval_env_stat.observation.values())), eval_env_stat.reward, 
                if eval_env_stat.discount is not None:
                    done = False
                else:
                    done = True
                    break
                avg_reward += reward

        avg_reward /= eval_episodes
        return avg_reward

    env = suite.load(ENV_NAME.split('-')[0], ENV_NAME.split('-')[1])
    torch.manual_seed(0)
    np.random.seed(0)

    spec = env.action_spec()
    state_dim = len(np.hstack(list(env.reset().observation.values())))
    action_dim = spec.shape[0]
    max_action = spec.maximum[0]
    os.makedirs('results',exist_ok=True)
    args_policy_noise = 0.2
    args_noise_clip = 0.5
    args_policy_freq = 2
    args_max_timesteps = args.max_timesteps
    args_expl_noise = 0.1
    args_batch_size = args.batch_size
    args_sample_reuse = args.sample_reuse
    args_eval_freq = 1000
    args_start_timesteps = 25000

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": 0.99,
        "tau": 0.005
    }

    for repeat in range(args.repeat, args.repeat+1):
        if True:
            args_policy = 'TD3'

            if args_policy == "TD3":
                # Target policy smoothing is scaled wrt the action scale
                kwargs["policy_noise"] = args_policy_noise * max_action
                kwargs["noise_clip"] = args_noise_clip * max_action
                kwargs["policy_freq"] = args_policy_freq
                policy = TD3.TD3(**kwargs)
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

        # Evaluate untrained policy
        evaluations = [eval_policy(policy)]

        env_stat = env.reset()
        state, done = np.hstack(list(env_stat.observation.values())), False



        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        counter = 0
        msk_list = []        
        temp_curve = [eval_policy(policy)]
        temp_val = []
        for t in range(int(args_max_timesteps)):
            episode_timesteps += 1
            counter += 1
            # Select action randomly or according to policy
            if t < args_start_timesteps:
                action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
            else:
                if np.random.uniform(0,1) < 0.0:
                    action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
                else:
                    action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * args_expl_noise, size=action_dim)
                    ).clip(-max_action, max_action)

            # Perform action
            env_stat = env.step(action)
            next_state, reward = np.hstack(list(env_stat.observation.values())), env_stat.reward
            if env_stat.discount is not None:
                done = False

                done_bool = float(done) if episode_timesteps < env._step_limit else 0

                replay_buffer.add(state, action, next_state, reward, done_bool)

                state = next_state
                episode_reward += reward

            else:
                done = True




            if t >= args_start_timesteps:
                '''TD3'''
                for sample_reuse_i in range(args.sample_reuse):
                    policy.train(replay_buffer, args_batch_size)

            # Train agent after collecting sufficient data
            if done or episode_timesteps > env._step_limit-1:
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                msk_list = []
                env_stat = env.reset()
                state, done = np.hstack(list(env_stat.observation.values())), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            # Evaluate episode
            if (t + 1) % args_eval_freq == 0:
                evaluations.append(eval_policy(policy))
                print('recent Evaluation:',evaluations[-1])
                np.save('results/evaluations_alias{}_ENV{}_Repeat{}'.format(alias,ENV_NAME,repeat),evaluations)