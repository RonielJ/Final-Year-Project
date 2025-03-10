import os

# General Configurations
NUM_INTERSECTIONS = 4
MAX_VEHICLES = 30
EMERGENCY_VEHICLE_PROBABILITY = 0.1
MIN_GREEN_TIME = 9
MIN_RED_TIME = 4
MAX_EPISODES = 1000
MAX_STEPS = 1000

# Model Configurations
LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
BUFFER_SIZE = 1000000
BATCH_SIZE = 256

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "../checkpoints/")
LOG_DIR = os.path.join(BASE_DIR, "../logs/")
