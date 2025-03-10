import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def save_checkpoint(model, filename):
    os.makedirs("../checkpoints", exist_ok=True)
    filepath = os.path.join("../checkpoints", filename)
    torch.save(model.state_dict(), filepath)
    print(f"Checkpoint saved: {filepath}")

def plot_training_progress(rewards, filename="training_progress.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward per Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.savefig(f"plots/{filename}")
    plt.show()
