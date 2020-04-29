# Code adapted from https://github.com/araffin/rl-baselines-zoo
# Author: Antonin Raffin

import argparse
import os
import time
import warnings

# Remove warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import numpy as np
from stable_baselines.common import set_global_seeds

from config import ENV_ID
from utils.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
import matplotlib.pyplot as plt
import random
from donkeycar.parts.datastore import TubHandler
from config import N_COMMAND_HISTORY, MIN_THROTTLE, MAX_THROTTLE, MAX_STEERING, MIN_STEERING, MAX_STEERING_DIFF

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', help='Log folder', type=str, default='logs')
parser.add_argument('-m', '--model', help='Model Path', type=str, default='')
parser.add_argument('--algo', help='RL Algorithm', default='sac',
                    type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument('-n', '--n-timesteps', help='number of timesteps', default=1000,
                    type=int)
parser.add_argument('--exp-id', help='Experiment ID (-1: no exp folder, 0: latest)', default=0,
                    type=int)
parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                    type=int)
parser.add_argument('--no-render', action='store_true', default=False,
                    help='Do not render the environment (useful for tests)')
parser.add_argument('--deterministic', action='store_true', default=False,
                    help='Use deterministic actions')
parser.add_argument('--norm-reward', action='store_true', default=False,
                    help='Normalize reward if applicable (trained with VecNormalize)')
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
parser.add_argument('--reward-log', help='Where to log reward', default='', type=str)
parser.add_argument('-vae', '--vae-path', help='Path to saved VAE', type=str, default='')
parser.add_argument('-best', '--best-model', action='store_true', default=False,
                    help='Use best saved model of that experiment (if it exists)')

args = parser.parse_args()

algo = args.algo
folder = args.folder

if args.exp_id == 0:
    args.exp_id = get_latest_run_id(os.path.join(folder, algo), ENV_ID)
    print('Loading latest experiment, id={}'.format(args.exp_id))

# Sanity checks
if args.exp_id > 0:
    log_path = os.path.join(folder, algo, '{}_{}'.format(ENV_ID, args.exp_id))
else:
    log_path = os.path.join(folder, algo)

best_path = ''
if args.best_model:
    best_path = '_best'

if args.model != '':
    model_path = args.model
else: 
    model_path = os.path.join(log_path, "{}{}.pkl".format(ENV_ID, best_path))

assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)
assert os.path.isfile(model_path), "No model found for {} on {}, path: {}".format(algo, ENV_ID, model_path)

set_global_seeds(args.seed)

stats_path = os.path.join(log_path, ENV_ID)
hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward)
hyperparams['vae_path'] = args.vae_path

log_dir = args.reward_log if args.reward_log != '' else None

env = create_test_env(stats_path=stats_path, seed=args.seed, log_dir=log_dir,
                      hyperparams=hyperparams)

model = ALGOS[algo].load(model_path)

obs = env.reset()
command_history = np.zeros((1, 2 * N_COMMAND_HISTORY ))
# Force deterministic for SAC and DDPG
deterministic = args.deterministic or algo in ['ddpg', 'sac']
if args.verbose >= 1:
    print("Deterministic actions: {}".format(deterministic))
reward_per_timestep= []
running_reward = 0.0
ep_len = 0
inputs=['cam/image_array',
        'user/angle', 'user/throttle', 
        'user/mode']

types=['image_array',
       'float', 'float',
       'str']

th = TubHandler(path='/home/sriharish/RL/lat_tub')
tub = th.new_tub_writer(inputs=inputs, types=types, user_meta=[])


for _ in range(args.n_timesteps):
    action, _ = model.predict(obs, deterministic=deterministic)

    img_arr = env.get_images()
    img_arr = np.asarray(img_arr[0])


    # Clip Action to avoid out of bound errors
    if isinstance(env.action_space, gym.spaces.Box):
        action = np.clip(action, env.action_space.low, env.action_space.high)


    if N_COMMAND_HISTORY> 0:
            prev_steering = command_history[0, -2]
            max_diff = (MAX_STEERING_DIFF - 1e-5) * (MAX_STEERING - MIN_STEERING)
            diff = np.clip(action[0][0] - prev_steering, -max_diff, max_diff)
            steering = prev_steering + diff

    # Convert from [-1, 1] to [0, 1]
    t = (action[0][1] + 1) / 2
    # Convert from [0, 1] to [min, max]
    throttle = (1 - t) * MIN_THROTTLE + MAX_THROTTLE * t
   
    tub.run(img_arr, steering, throttle , "user")

    obs, reward, done, infos = env.step(action)

    command_history = np.roll(command_history, shift=-2, axis=-1)
    command_history[..., -2:] = action[0]

    if not args.no_render:
        env.render('human')
    running_reward += reward[0]
    reward_per_timestep +=[reward[0]]

    ep_len += 1

    if done and args.verbose >= 1:
        # NOTE: for env using VecNormalize, the mean reward
        # is a normalized reward when `--norm_reward` flag is passed
        print("Episode Reward: {:.2f}".format(running_reward))
        print("Episode Length", ep_len)
        running_reward = 0.0
        ep_len = 0

env.reset()

env.envs[0].env.exit_scene()

#Plot the rewards against time steps
plt.figure(1)
plt.plot(reward_per_timestep)
plt.title('Episode Rewards')
plt.ylabel("Reward")
plt.xlabel("Timestep")
filename = "Trained1" + str(random.random()) + ".png"
plt.savefig(filename)
plt.show()
# Close connection does work properly for now
# env.envs[0].env.close_connection()
time.sleep(0.5)
