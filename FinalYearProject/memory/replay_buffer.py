import random
import numpy as np
from collections import deque
from config import MEMORY_SIZE

class ReplayBuffer:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
