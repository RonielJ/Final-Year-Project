from environment.traffic_env import TrafficEnvironment
from models.sac_agent import SAC

def train():
    env = TrafficEnvironment()
    agent = SAC(env.observation_space.shape[0], env.action_space.shape[0])

    for episode in range(1000):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state

if __name__ == "__main__":
    train()
