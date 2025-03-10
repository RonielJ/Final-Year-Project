from environment.traffic_env import TrafficEnvironment
from models.sac_agent import SAC
import torch

def evaluate():
    env = TrafficEnvironment()
    agent = SAC(env.observation_space.shape[0], env.action_space.shape[0])

    # Load trained model
    agent.actor.load_state_dict(torch.load("checkpoints/sac_actor.pth"))
    agent.actor.eval()

    total_reward = 0
    state = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Evaluation Reward: {total_reward}")

if __name__ == "__main__":
    evaluate()
