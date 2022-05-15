import torch 
import numpy as np
import copy
import torch.nn.functional as F
from collections import deque
from RND import RND
import random
from log_utils import logger, mean_val


class QNet(torch.nn.Module):
    def __init__(self,in_dim,out_dim,n_hid):
        super(QNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hid = n_hid
        
        self.fc1 = torch.nn.Linear(in_dim,n_hid,'relu')
        self.fc2 = torch.nn.Linear(n_hid,out_dim,'linear')
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        return y

class DQN_RND:
    def __init__(self, env, gamma, args):
        self.env = env
        self.mn = args.mn
        acts = env.action_space
        obs_shape = env.observation_space['image'].shape[0]*env.observation_space['image'].shape[1]*env.observation_space['image'].shape[2] if 'MiniGrid' in args.env_name else env.observation_space.shape[0]
        self.model = QNet(obs_shape,acts.n,64).cuda()
        self.target_model = copy.deepcopy(self.model).cuda()
        self.rnd = RND(obs_shape,512,512)
        self.gamma = args.gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001)
        self.batch_size = 64
        self.epsilon = 0.1
        self.buffer_size = args.buffer_size
        self.step_counter = 0
        self.epsi_high = 0.9
        self.epsi_low = 0.05
        self.steps = 0
        self.count = 0
        self.decay = 200
        self.eps = self.epsi_high
        self.update_target_step = 500
        self.mode = args.mode
        self.log = logger()
        self.log.add_log('real_return')
        self.log.add_log('combined_return')
        self.log.add_log('avg_loss')
        self.args = args
        self.replay_buffer = deque(maxlen=args.buffer_size)
        self.intrinsic_reward_history = []
    def run_episode(self):
        obs = self.env.reset()
        obs = obs['image'].flatten() if 'MiniGrid' in self.args.env_name else obs
        sum_r = 0
        sum_tot_r = 0
        mean_loss = mean_val()

        for t in range(self.args.max_game_len):
            self.steps += 1
            self.eps = self.epsi_low + (self.epsi_high-self.epsi_low) * (np.exp(-1.0 * self.steps/self.decay))
            state = torch.Tensor(obs).unsqueeze(0)
            Q = self.model(state.cuda()).cpu().detach()
            num = np.random.rand()
            if (num < self.eps):
                action = torch.randint(0,Q.shape[1],(1,)).type(torch.LongTensor)
            else:
                action = torch.argmax(Q,dim=1)
            new_state, reward, done, info = self.env.step((action.item()))
            new_state = new_state['image'].flatten() if 'MiniGrid' in self.args.env_name else new_state
            sum_r = sum_r + reward
            reward_i = self.rnd.get_reward(state.cuda()).cpu().detach().clamp(-1.0,1.0).item()
            if self.mode == 'RND':
                combined_reward = reward + reward_i
                self.intrinsic_reward_history.append(reward_i)
            elif self.mode == 'DQN':
                combined_reward = reward
            elif self.mode == 'DQNm':
                combined_reward = reward - self.mn
            elif self.mode == 'RNDm':
                combined_reward = reward + reward_i - self.mn
            sum_tot_r += combined_reward
            
            self.replay_buffer.append([obs,action,combined_reward,new_state,done])
            loss = self.update_model()
            mean_loss.append(loss)
            obs = new_state
            
#             self.step_counter = self.step_counter + 1
#             if (self.step_counter > self.update_target_step):
#                 self.target_model.load_state_dict(self.model.state_dict())
#                 self.step_counter = 0
#                 print('updated target model')
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_((1 - 0.95) * param.data + 0.95 * target_param.data)
            if done:
                break
        self.log.add_item('real_return',sum_r)
        self.log.add_item('combined_return',sum_tot_r)
        self.log.add_item('avg_loss',mean_loss.get())
        
    def update_model(self):
        self.optimizer.zero_grad()
        num = len(self.replay_buffer)
        K = np.min([num,self.batch_size])
        samples = random.sample(self.replay_buffer, K)
        
        S0, A0, R1, S1, D1 = zip(*samples)
        S0 = torch.tensor( S0, dtype=torch.float)
        A0 = torch.tensor( A0, dtype=torch.long).view(K, -1)
        R1 = torch.tensor( R1, dtype=torch.float).view(K, -1)
        S1 = torch.tensor( S1, dtype=torch.float)
        D1 = torch.tensor( D1, dtype=torch.float)
       
        Ri = self.rnd.get_reward(S0.cuda())
        self.rnd.update(Ri)
        target_q = R1.squeeze().cuda() + self.gamma*self.target_model(S1.cuda()).max(dim=1)[0].detach()*(1 - D1.cuda())
        policy_q = self.model(S0.cuda()).gather(1,A0.cuda())
        L = F.smooth_l1_loss(policy_q.squeeze(),target_q.squeeze())
        L.backward()
        self.optimizer.step()
        return L.cpu().detach().item()
    
    def run_epoch(self):
        self.run_episode()
        return self.log

