import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_rew=1):
        super(Critic, self).__init__()

        # Q1 architecture
        # self.l1 = nn.Linear(state_dim + action_dim, 256)
        # self.l2 = nn.Linear(256, 256)
        # self.l3 = nn.Linear(256, 1)
        self.n_rew = n_rew
        self.l1 = nn.ModuleList()
        self.l2 = nn.ModuleList()
        self.l3 = nn.ModuleList()

        for _ in range(n_rew):
            self.l1.append(nn.Linear(state_dim + action_dim, 256))
            self.l2.append(nn.Linear(256, 256))
            self.l3.append(nn.Linear(256, 1))


        # Q2 architecture
        self.l4 = nn.ModuleList()
        self.l5 = nn.ModuleList()
        self.l6 = nn.ModuleList()

        for _ in range(n_rew):
            self.l4.append(nn.Linear(state_dim + action_dim, 256))
            self.l5.append(nn.Linear(256, 256))
            self.l6.append(nn.Linear(256, 1))


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)


        for i in range(self.n_rew):
            tmp = F.relu(self.l1[i](sa))
            tmp = F.relu(self.l2[i](tmp))
            if i == 0:
                q1 = self.l3[i](tmp)
            else:
                q1 = torch.cat([q1, self.l3[i](tmp)],1)
            tmp2 = F.relu(self.l4[i](sa))
            tmp2 = F.relu(self.l5[i](tmp2))
            if i == 0:
                q2 = self.l6[i](tmp2)
            else:
                q2 = torch.cat([q2, self.l6[i](tmp2)],1)

        # q1 = F.relu(self.l1(sa))
        # q1 = F.relu(self.l2(q1))
        # q1 = self.l3(q1)

        # q2 = F.relu(self.l4(sa))
        # q2 = F.relu(self.l5(q2))
        # q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        # q1 = F.relu(self.l1(sa))
        # q1 = F.relu(self.l2(q1))
        # q1 = self.l3(q1)
        for i in range(self.n_rew):
            tmp = F.relu(self.l1[i](sa))
            tmp = F.relu(self.l2[i](tmp))
            if i == 0:
                q1 = self.l3[i](tmp)
            else:
                q1 = torch.cat([q1, self.l3[i](tmp)],1)
        return q1


class TD3_MultiQ(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        rew_lst = None,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        self.rew_lst = torch.as_tensor(rew_lst).float().to(device)
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, len(rew_lst)).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer, batch_size=100, update_policy=True, policy_upd_idx = None):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1s, target_Q2s = self.critic_target(next_state, next_action)
            target_Qs = torch.min(target_Q1s, target_Q2s)
            target_Qs = reward + not_done * self.discount * target_Qs + self.rew_lst

        # Get current Q estimates
        current_Q1s, current_Q2s = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1s, target_Qs) + F.mse_loss(current_Q2s, target_Qs)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            if update_policy:
                # Compute actor losse
                actor_loss = -self.critic.Q1(state, self.actor(state))[:,policy_upd_idx].mean()

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            if update_policy:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
