from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import os
import torch.nn.functional as F
import utils
import TD3_MultiQ
import gym
from collections import deque
def parse_args():
    parser = ArgumentParser(description='train args')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-en','--env_name', type=str, default='Humanoid-v2')
    parser.add_argument('-mt','--max_timesteps',type=int, default= 2e6)
    parser.add_argument('-a','--alias', type=str, default=None)
    parser.add_argument('-sr','--sample_reuse', type=int, default=1)
    parser.add_argument('-b','--batch_size', type=int, default=256)
    parser.add_argument('-r','--repeat',type=int,default=None)
    parser.add_argument('-rl','--reward_list',nargs='+',default=None)
    parser.add_argument('-eps','--epsilon',type=float,default=None)
    parser.add_argument('-mxlen','--coach_hist_maxlen',type=int,default=1)
    parser.add_argument('-cm','--coach_mode',type=str,default=None)
    parser.add_argument('-bta','--beta',type=float,default = 1.0)
    parser.add_argument('-sf','--switch_frequency',type=int, default=1000)
    parser.add_argument('-expn','--expl_noise',type=bool,default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    ENV_NAME = args.env_name
    alias = f'MultiQ_UPD_{args.env_name}_SampleReuse_{args.sample_reuse}_BS_{args.batch_size}_EXTAlias_{args.alias}_sf_{args.switch_frequency}_eps_{args.epsilon}_beta_{args.beta}_CoachLen{args.coach_hist_maxlen}'
    NUM_REWARD = len(args.reward_list)
    
    
    def eps_argmax(values, epsilon):
        out_tmp_argmax = np.zeros((NUM_REWARD,))
        out_tmp_argmax += epsilon/(NUM_REWARD-1)
        out_tmp_argmax[np.argmax(values)] = 1.0 - epsilon
        assert np.sum(out_tmp_argmax) == 1.0
        
        return out_tmp_argmax
        
        
        
    def eps_softmax(values, epsilon):
        value_with_eps = np.asarray(values).clip(-10,30)
        exp_value = np.round(np.exp(args.beta * np.asarray(value_with_eps)),5) + 0.001
        sum_exp_value = sum(exp_value)
        soft_max_value =(exp_value/sum_exp_value) * (1.0-epsilon) + epsilon/NUM_REWARD
        soft_max_value[-1] = 1 - sum(soft_max_value[:-1])
        try:
            assert sum(soft_max_value) == 1
        except:
            print('values',values)
        return soft_max_value

    def eval_policy(policy, eval_episodes=10):
        eval_env = gym.make(ENV_NAME)

        avg_reward = 0.
        for _ in range(eval_episodes):
            state = eval_env.reset()
            done = False
            while not done:
                action = policy.select_action(np.array(state))

                state, reward, done, _ = eval_env.step(action)

                #state, reward = np.hstack(list(eval_env_stat.observation.values())), eval_env_stat.reward, 
                if done:
                    break
                avg_reward += reward

        avg_reward /= eval_episodes
        return avg_reward
    Coach_queues = [deque(maxlen = args.coach_hist_maxlen) for _ in range(NUM_REWARD)]
    Coach_Values = np.zeros((NUM_REWARD,))
    for i in range(len(Coach_queues)):
        Coach_queues[i].extend([0]*args.coach_hist_maxlen)
    

    env = gym.make(ENV_NAME)
    torch.manual_seed(0)
    np.random.seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    os.makedirs('results',exist_ok=True)
    args_policy_noise = 0.2
    args_noise_clip = 0.5
    args_policy_freq = 2
    args_max_timesteps = args.max_timesteps
    args_expl_noise = 0.1 if args.expl_noise else 0.0
    args_batch_size = args.batch_size
    args_sample_reuse = args.sample_reuse
    args_eval_freq = 1000
    args_start_timesteps = 25000

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": 0.99,
        "tau": 0.005,
        "rew_lst":np.asarray(args.reward_list).astype(float)
    }

    for repeat in range(args.repeat, args.repeat+1):
        args_policy = 'TD3_MultiQ'
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args_policy_noise * max_action
        kwargs["noise_clip"] = args_noise_clip * max_action
        kwargs["policy_freq"] = args_policy_freq
        policy = TD3_MultiQ.TD3_MultiQ(**kwargs)
        replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

        # Evaluate untrained policy
        evaluations = [eval_policy(policy)]

        if args.coach_mode == 'softmax':
            value_idx_now = np.random.choice(NUM_REWARD, 1, p=eps_softmax(Coach_Values, args.epsilon))[0]
        elif args.coach_mode == 'argmax':
            value_idx_now = np.random.choice(NUM_REWARD, 1, p=eps_argmax(Coach_Values, args.epsilon))[0]
        elif args.coach_mode == 'random':
            value_idx_now = np.random.randint(0,NUM_REWARD)
        else:
            raise NotImplementedError
        print(f'using env {value_idx_now}')


        state = env.reset()
        value_timestep = 0
        done = False
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
                action = np.random.uniform(-max_action, max_action, action_dim)
            else:
                if np.random.uniform(0,1) < 0.0:
                    action = np.random.uniform(-max_action, max_action, action_dim)
                else:
                    action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * args_expl_noise, size=action_dim)
                    ).clip(-max_action, max_action)

            # Perform action
            next_state, reward, done, _ = env.step(action)
            #next_state, reward = np.hstack(list(env_stat.observation.values())), env_stat.reward
           

            done_bool = float(done) if episode_timesteps < 1000 else 0

            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward





            if t >= args_start_timesteps:
                '''TD3'''
                for sample_reuse_i in range(args.sample_reuse):
                    policy.train(replay_buffer, args_batch_size, policy_upd_idx=value_idx_now)

            # Train agent after collecting sufficient data
            if done:
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                msk_list = []
                value_timestep += episode_timesteps
                
                if value_timestep >= args.switch_frequency:
                    n_eval_before_update = int(args.switch_frequency/args_eval_freq)
                    '''vdiff:'''
                    print(f'update Coach!, with mean value in last {n_eval_before_update} evals.')
                    Coach_queues[value_idx_now].append(np.mean(evaluations[-int(n_eval_before_update/2):]) - np.mean(evaluations[-n_eval_before_update:-int(n_eval_before_update/2)]))
                    Coach_Values[value_idx_now] = np.mean(Coach_queues[value_idx_now])
                    if args.coach_mode == 'softmax':
                        value_idx_now = np.random.choice(NUM_REWARD, 1, p=eps_softmax(Coach_Values, args.epsilon))[0]
                    elif args.coach_mode == 'argmax':
                        value_idx_now = np.random.choice(NUM_REWARD, 1, p=eps_argmax(Coach_Values, args.epsilon))[0]
                    elif args.coach_mode == 'random':
                        value_idx_now = np.random.randint(0,NUM_REWARD)
                    else:
                        raise NotImplementedError
                    print(f'using value {value_idx_now}')
                    value_timestep = 0


                state = env.reset()
                
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            # Evaluate episode
            if (t + 1) % args_eval_freq == 0:
                evaluations.append(eval_policy(policy))
                print('recent Evaluation:',evaluations[-1], 'max_eval',max(evaluations))
                print('current coach values:',Coach_Values.round(3))
                
                if args.coach_mode == 'softmax':
                    print('current coach probs:',eps_softmax(Coach_Values,args.epsilon).round(3))
                elif args.coach_mode == 'argmax':
                    print('current coach probs:',eps_argmax(Coach_Values,args.epsilon).round(3))
                elif args.coach_mode == 'random':
                    print('use random coach')
                else:
                    raise NotImplementedError

                np.save('results/evaluations_alias{}_ENV{}_Repeat{}'.format(alias,ENV_NAME,repeat),evaluations)