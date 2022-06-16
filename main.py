import sys

import gym
import numpy as np
import torch
import argparse
import os
import utils
import algos
from logger import logger, setup_logger
import d4rl
import torch.nn as nn
import time
from data_utils import d4rl_trajectories
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_policy(policy, mean, std, eval_episodes=10):
    all_returns = []
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        episodic_len = 0
        episodic_reward = 0
        while not done:
            obs = (np.array(obs).reshape(1, -1) - mean) / std
            action = policy.select_action(obs)
            obs, rew, done, info = env.step(action)
            episodic_reward += rew
            episodic_len += 1
            if episodic_len+1 == env._max_episode_steps:
                done = True
        all_returns.append(episodic_reward)
    all_returns = np.array(all_returns)
    avg_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    median_return = np.median(all_returns)
    min_return = np.min(all_returns)

    d4rl_score = env.get_normalized_score(avg_return)

    print ("---------------------------------------")
    print ("Evaluation over %d episodes: %f | normalized score :%f " % (eval_episodes, avg_return, d4rl_score))
    print ("---------------------------------------")
    return avg_return, std_return, median_return, min_return, d4rl_score




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="hopper-expert-v2")    # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=float)      # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--version", default='0', type=str)
    parser.add_argument('--algo_name', default="TD3_BC", type=str)    # Which algo to run (see the options below in the main function)
    parser.add_argument('--log_dir', default='./data_tmp/', type=str) # Logging directory
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--buffer_size', default=1000000, type=int)
    args = parser.parse_args()



    init_time = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(args.env_name)
    env.seed(args.seed)


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(f'State dim :{state_dim}, Action dim: {action_dim}')
    print('Max action: ', max_action)

    # Load buffer
    replay_buffer = utils.ReplayBuffer()
    dataset = env.unwrapped.get_dataset()
    num_trajectories = int(args.buffer_size / env._max_episode_steps)
    d4rl_trajectories(dataset, env, replay_buffer, buffer_size=args.buffer_size)
    mean, std = replay_buffer.normalize_states()

    hparam_str_dict = dict(algo=args.algo_name, seed=args.seed, env=args.env_name,
                           batch_size=args.batch_size, buffer_size=args.buffer_size)
    variant = hparam_str_dict
    file_name = ','.join(['%s=%s' % (k, str(hparam_str_dict[k])) for k in sorted(hparam_str_dict.keys())])

    print ("---------------------------------------")
    print ("Settings: " + file_name)
    print ("---------------------------------------")

    setup_logger(file_name, variant=variant, log_dir=os.path.join(args.log_dir, file_name))


    if args.algo_name == 'BCQ':
        policy = algos.BCQ(state_dim=state_dim,
                           action_dim=action_dim,
                           max_action=max_action,
                           discount=args.gamma)
    elif args.algo_name == 'IQL':
        policy = algos.IQL(state_dim=state_dim,
                           action_dim=action_dim,
                           max_action=max_action,
                           hidden_dim=args.hidden_dim,
                           discount=args.gamma)
    elif args.algo_name == 'BCQ-v2':
        policy = algos.BCQ(state_dim=state_dim,
                           action_dim=action_dim,
                           max_action=max_action,
                           cloning=True,
                           discount=args.gamma)
    elif args.algo_name == 'TD3_BC':
        policy = algos.TD3_BC(state_dim=state_dim,
                              action_dim=action_dim,
                              max_action=max_action,
                              hidden_dim=args.hidden_dim,
                              discount=args.gamma)
    else:
        sys.exit(f'Choose the right algo name, {args.algo_name} not found')



    training_iters = 0
    while training_iters < args.max_timesteps: 
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)
        avg_return, std_return, median_return, min_return, d4rl_score = evaluate_policy(policy, mean, std)
        training_iters += args.eval_freq
        print("Training iterations: " + str(training_iters))
        logger.record_tabular('Training Epochs', int(training_iters // int(args.eval_freq)))
        logger.record_tabular('Eval/AverageReturn', avg_return)
        logger.record_tabular('Eval/StdReturn', std_return)
        logger.record_tabular('Eval/MedianReturn', median_return)
        logger.record_tabular('Eval/MinReturn', min_return)
        logger.record_tabular('Eval/D4RL_score', d4rl_score)
        logger.record_tabular('training time', (time.time() - init_time) / (60 * 60))

        logger.dump_tabular()
