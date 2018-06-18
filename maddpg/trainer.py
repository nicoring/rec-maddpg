import argparse
import time
import os
import signal

import numpy as np
from tensorboardX import SummaryWriter
import torch
import gym
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

from agents import MaddpgAgent, RandomAgent
from models import Actor, Critic


def parse_args():
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument('--scenario', type=str, default='simple', help='name of the scenario script')
    parser.add_argument('--max-train-steps', type=int, default=500_000, help='maximum episode length')
    parser.add_argument('--max-episode-len', type=int, default=25, help='maximum episode length')
    parser.add_argument('--num-adversaries', type=int, default=0, help='number of adversaries')
    parser.add_argument('--render', default=False, action='store_true', help='display agent policies')
    parser.add_argument('--render-only', default=False, action='store_true', help='display agent policies')
    parser.add_argument('--train-every', type=int, default=100, help='simulation steps in between network updates')
    parser.add_argument('--benchmark', default=False, action='store_true', help='')
    parser.add_argument('--hidden', type=int, default=64, help='number of hidden units in actor and critic')
    parser.add_argument('--batch-size', type=int, default=1024, help='size of minibatch that is sampled from replay buffer')
    parser.add_argument('--memory-size', type=int, default=1_000_000, help='size of replay memory')
    parser.add_argument('--tau', type=float, default=0.01, help='update rate for exponential update of target network params')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor for training of critic')
    parser.add_argument('--exp-name', default='test', help='name of experiment')
    parser.add_argument('--exp-run-num', type=str, default='', help='run number of experiment gets appended to log dir')
    parser.add_argument('--train-steps', type=int, default=1)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=-1)
    parser.add_argument('--lr-actor', type=float, default=1e-3)
    parser.add_argument('--lr-critic', type=float, default=1e-2)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--resume', help='dirname of saved state')
    # parser.add_argument('--good-policy', type=str, default='maddpg', help='policy for good agents')
    # parser.add_argument('--adv-policy', type=str, default='maddpg', help='policy of adversaries')

    return parser.parse_args()


def make_env(scenario_name, benchmark=False):
    scenario = scenarios.load(scenario_name + '.py').Scenario()
    world = scenario.make_world()
    if benchmark and hasattr(scenario, 'benchmark_data'):
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def create_random_agents(env, num_adversaries):
    return [RandomAgent(i, 'agent_%d' % i, env) for i in range(env.n)]


def create_agents(env, params):
    agents = []
    n_agents = env.n

    n_critic_inputs = 0
    action_splits = []
    for action_space in env.action_space:
        if isinstance(action_space, gym.spaces.Discrete):
            action_splits.append([action_space.n])
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            action_splits.append(action_space.nvec)

    n_obs = sum(o.shape[0] for o in env.observation_space)
    n_actions = sum(sum(a) for a in action_splits)
    n_critic_inputs += (n_actions + n_obs)

    for i in range(n_agents):
        n_observations = env.observation_space[i].shape[0]
        actor = Actor(n_observations, action_splits[i], params.hidden)
        critic = Critic(n_critic_inputs, params.hidden)
        agent = MaddpgAgent(i, 'agent_%d' % i, actor, critic, params)
        agents.append(agent)
    return agents


def save_agent_states(filename, agents):
    states = [agent.get_state() for agent in agents]
    torch.save(states, filename)


def load_agent_states(filename, agents):
    states = torch.load(filename)
    for agent, state in zip(agents, states):
        agent.load_state(state)


def train(args):
    def signal_handling(signum, frame):
        nonlocal terminated
        terminated = True
    signal.signal(signal.SIGINT, signal_handling)

    env = make_env(args.scenario, args.benchmark)
    agents = create_agents(env, args)

    # load state of agents if state file is given
    if args.resume:
        filename = os.path.join(args.resume, 'checkpoint.pth.tar')
        if os.path.isfile(filename):
            load_agent_states(filename, agents)
            args.exp_name, args.exp_run_num = os.path.normpath(args.resume).split('/')[1:]
        else:
            print("Couldn't find checkpoint at %s" % args.resume)

    # logging
    run_name = str(np.datetime64('now')) if args.exp_run_num == '' else args.exp_run_num
    writer = SummaryWriter(log_dir=os.path.join('runs', args.exp_name, run_name))
    dirname = os.path.join('results', args.exp_name, run_name)
    os.makedirs(dirname, exist_ok=True)
    rewards_file = os.path.join(dirname, 'rewards.csv')
    if not os.path.isfile(rewards_file):
        with open(rewards_file, 'w') as f:
            line = ','.join(['cum_reward'] + [a.name for a in agents]) + '\n'
            f.write(line)


    episode_returns = []
    agent_returns = []
    train_step = 0
    terminated = False

    while (train_step <= args.max_train_steps) and not terminated:
        obs = env.reset()
        done = False
        terminal = False
        cum_reward = 0
        agents_cum_reward = [0.0] * env.n
        episode_step = 0
        while not (done or terminal):
            # global step count
            train_step += 1
            # epsiode step count
            episode_step += 1

            # act with all agents in environment and receive observation and rewards
            actions = [agent.act(o) for o, agent in zip(obs, agents)]
            if args.debug:
                obs_t = torch.tensor(obs, dtype=torch.float)
                values = {agent.name: agent.critic(obs_t, actions) for agent in agents}
                writer.add_scalars('values', values, train_step)
            numpy_actions = [a.numpy() for a in actions]
            new_obs, rewards, dones, _ = env.step(numpy_actions)
            done = all(dones)
            terminal = episode_step >= args.max_episode_len

            # store rewards
            for i, reward in enumerate(rewards):
                cum_reward += reward
                agents_cum_reward[i] += reward

            # rendering environment
            if args.render and (len(episode_returns) % args.eval_every == 0) or args.render_only:
                time.sleep(0.1)
                env.render()
                if args.render_only:
                    continue

            # store tuples (observation, action, reward, next observation, is done) for each agent
            for i, agent in enumerate(agents):
                agent.experience(obs[i], actions[i], rewards[i], new_obs[i], dones[i])
            obs = new_obs

            # train agents
            if train_step % args.train_every == 0:
                for _ in range(args.train_steps):
                    for agent in agents:
                        actor_loss, critic_loss = agent.update(agents)
                        if args.debug:
                            writer.add_scalar('actor_loss', actor_loss, train_step)
                            writer.add_scalar('critic_loss', critic_loss, train_step)

        if len(episode_returns) % args.eval_every == 0:
            msg = 'step {}: episode {} finished with a return of {:.2f}'
            print(msg.format(train_step, len(episode_returns), cum_reward))

        if args.render_only:
            continue

        if (args.save_every != -1) and (len(episode_returns) % args.save_every == 0):
            filename = os.path.join(dirname, 'checkpoint.pth.tar')
            save_agent_states(filename, agents)

        # store and log rewards
        episode_returns.append(cum_reward)
        agent_returns.append(agents_cum_reward)

        with open(rewards_file, 'a') as f:
            line = ','.join(map(str, [cum_reward] + agents_cum_reward)) + '\n'
            f.write(line)

        agent_rewards_dict = {a.name: r for a, r in zip(agents, agents_cum_reward)}
        writer.add_scalar('reward', cum_reward, train_step)
        writer.add_scalars('agent_rewards', agent_rewards_dict, train_step)

    print('Finished training with %d episodes' % len(episode_returns))
    # TODO: store returns

if __name__ == '__main__':
    train(parse_args())
