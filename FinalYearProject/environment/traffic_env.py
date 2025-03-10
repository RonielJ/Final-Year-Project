import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from config.settings import NUM_INTERSECTIONS , MAX_VEHICLES , EMERGENCY_VEHICLE_PROBABILITY , MIN_GREEN_TIME, MIN_RED_TIME

class TrafficEnvironment(gym.Env):
    def __init__(self):
        super(TrafficEnvironment, self).__init__()

        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(NUM_INTERSECTIONS * (4 + 4 + 1),)
        )

        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(NUM_INTERSECTIONS,)
        )

        self.reset()

    def reset(self):
        self.current_step = 0
        self.max_steps = 1000
        self.max_wait_threshold = 5000

        self.vehicles = np.random.randint(0, MAX_VEHICLES // 2, (NUM_INTERSECTIONS, 4))
        self.emergency_vehicles = np.zeros((NUM_INTERSECTIONS, 4), dtype=bool)
        self.phase_duration = np.zeros(NUM_INTERSECTIONS)
        self.current_phase = np.zeros(NUM_INTERSECTIONS, dtype=int)
        self.waiting_time = np.zeros((NUM_INTERSECTIONS, 4))

        return self._get_state()

    def _get_state(self):
        state = []
        for i in range(NUM_INTERSECTIONS):
            state.extend(self.vehicles[i])
            state.extend(self.emergency_vehicles[i].astype(float))
            state.append(self.phase_duration[i])
        return np.array(state)

    def step(self, actions):
        self.current_step += 1
        actions = np.clip(actions, 0, 1)

        for i in range(NUM_INTERSECTIONS):
            green_time = MIN_GREEN_TIME + actions[i] * 6
            red_time = MIN_RED_TIME + (1 - actions[i]) * 3

            self.phase_duration[i] += 1

            if (self.current_phase[i] == 0 and self.phase_duration[i] >= green_time) or \
               (self.current_phase[i] == 1 and self.phase_duration[i] >= red_time):
                self.current_phase[i] = 1 - self.current_phase[i]
                self.phase_duration[i] = 0

        self._update_traffic()
        reward = self._calculate_reward()

        done = (
            self.current_step >= self.max_steps or
            np.sum(self.waiting_time) > self.max_wait_threshold
        )

        return self._get_state(), reward, done, {}

    def _update_traffic(self):
        for i in range(NUM_INTERSECTIONS):
            for lane in range(4):
                if random.random() < 0.3:
                    self.vehicles[i][lane] = min(self.vehicles[i][lane] + 1, MAX_VEHICLES)
                if random.random() < EMERGENCY_VEHICLE_PROBABILITY:
                    self.emergency_vehicles[i][lane] = True

            if self.current_phase[i] == 0:
                self.vehicles[i][0] = max(0, self.vehicles[i][0] - 3)
                self.vehicles[i][2] = max(0, self.vehicles[i][2] - 3)
                self.emergency_vehicles[i][0] = False
                self.emergency_vehicles[i][2] = False
                self.waiting_time[i][1] += 1
                self.waiting_time[i][3] += 1
            else:
                self.vehicles[i][1] = max(0, self.vehicles[i][1] - 3)
                self.vehicles[i][3] = max(0, self.vehicles[i][3] - 3)
                self.emergency_vehicles[i][1] = False
                self.emergency_vehicles[i][3] = False
                self.waiting_time[i][0] += 1
                self.waiting_time[i][2] += 1

    def _calculate_reward(self):
        reward = 0
        for i in range(NUM_INTERSECTIONS):
            reward -= np.sum(self.waiting_time[i]) * 0.1
            emergency_waiting = np.sum(self.emergency_vehicles[i] * self.waiting_time[i])
            reward -= emergency_waiting * 1.0
            reward += np.sum(self.vehicles[i] < MAX_VEHICLES * 0.5) * 0.5
        return reward
