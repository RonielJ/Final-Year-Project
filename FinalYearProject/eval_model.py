import torch
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from sumo_env.make_tsc_env import make_env
from sumo_datasets.TEST_CONFIG import TEST_SUMO_CONFIG as SUMO_CONFIG

path_convert = get_abs_path(__file__)
logger.remove()

def initialize_environment(config_name, env_config, model_root):
    params = {
        'root_folder': path_convert("./sumo_datasets/"),
        'init_config': env_config,
        'env_dict': {config_name: env_config},
        'num_seconds': 3600,
        'use_gui': False,
        'log_file': path_convert('./log/')
    }
    env = SubprocVecEnv([make_env(env_index='0', **params)])
    return env

def load_model(env, model_root):
    model_path = f"{model_root}/last_rl_model.zip"
    model = PPO.load(model_path, env=env, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model

def test_model(env, model):
    obs = env.reset()
    dones = False
    total_reward = 0
    total_waiting_time = 0
    actions = {'0':0, '1':0}
    while not dones:
        action, _states = model.predict(obs, deterministic=True)
        actions[str(action[0])] += 1
        obs, rewards, dones, infos = env.step(action)
        total_reward += rewards[0]
        total_waiting_time += infos[0]['avg_waiting_time']
    env.close()
    return total_reward, total_waiting_time, actions

def Evaluate() -> None:
    model_root = path_convert('./save_models/')
    for config_name, config_data in SUMO_CONFIG.items():
        for net in config_data['nets']:
            for route in config_data['routes']:
                env_config = {
                    'tls_id': config_data['tls_id'],
                    'sumocfg': config_data['sumocfg'],
                    'nets': [net],
                    'routes': [route]
                }
                env = initialize_environment(config_name, env_config, model_root)
                model = load_model(env, model_root)
                total_reward, total_waiting_time, actions = test_model(env, model)
                print(
                    f'{config_name}||{net}||{route}||Waiting time: {total_waiting_time:.2f}||Reward: {total_reward:.2f}||Actions: {actions}'
                )

if __name__== "__main__":
    Evaluate()