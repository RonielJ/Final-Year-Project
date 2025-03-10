import torch
import torch.optim as optim
from torch.distributions import Normal
from models.actor import Actor
from models.critic import Critic
from config.settings import LEARNING_RATE, TAU, GAMMA

class SAC:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=LEARNING_RATE)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.actor(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        action = torch.tanh(normal.rsample())
        return action.detach().numpy()[0]
