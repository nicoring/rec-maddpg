import argparse
import time
import os

import numpy as np
from tensorboardX import SummaryWriter
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

from agents import MaddpgAgent, RandomAgent
from models import Actor, Critic

def parse_args():
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument('--scenario', type=str, default='simple', help='name of the scenario script')
    parser.add_argument('--max-train-steps', type=int, default=1_000_000, help='maximum episode length')
    parser.add_argument('--max-episode-len', type=int, default=25, help='maximum episode length')
    # parser.add_argument('--num-episodes', type=int, default=60000, help='number of episodes')
    parser.add_argument('--num-adversaries', type=int, default=0, help='number of adversaries')
    parser.add_argument('--render', default=False, action='store_true', help='display agent policies')
    parser.add_argument('--train-every', type=int, default=100, help='simulation steps in between network updates')
    parser.add_argument('--benchmark', default=False, action='store_true', help='')
    parser.add_argument('--hidden', type=int, default=64, help='number of hidden units in actor and critic')
    parser.add_argument('--batch-size', type=int, default=1024, help='size of minibatch that is sampled from replay buffer')
    parser.add_argument('--memory-size', type=int, default=1_000_000, help='size of replay memory')
    parser.add_argument('--tau', type=float, default=0.01, help='update rate for exponential update of target network params')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor for training of critic')
    parser.add_argument('--exp-name', default='test', help='name of experiment')
    parser.add_argument('--exp-run-num', type=str, default='', help='run number of experiment gets appended to log dir')
    # parser.add_argument('--good-policy', type=str, default='maddpg', help='policy for good agents')
    # parser.add_argument('--adv-policy', type=str, default='maddpg', help='policy of adversaries')

    return parser.parse_args()

def make_env(scenario_name, benchmark=False):
    scenario = scenarios.load(scenario_name + '.py').Scenario()
    world = scenario.make_world()
    if benchmark and hasattr(scenario, 'benchmark_data'):
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def create_random_agents(env, num_adversaries):
    return [RandomAgent(i, 'agent_%d' % i, env) for i in range(env.n)]

def create_agents(env, params):
    agents = []

    n_agents = env.n
    
    for i in range(n_agents):
        n_observations = env.observation_space[i].shape[0]
        n_actions = env.action_space[i].n
        actor = Actor(n_observations, n_actions, params.hidden)
        n_critic_inputs = sum(o.shape[0] for o in env.observation_space) + sum(a.n for a in env.action_space)
        critic = Critic(n_critic_inputs, params.hidden)
        agent = MaddpgAgent(i, 'agent_%d' % i, actor, critic, params)
        agents.append(agent)

    return agents

def train(args):
    run_name = str(np.datetime64('now')) if args.exp_run_num == '' else args.exp_run_num
    writer = SummaryWriter(log_dir=os.path.join('runs', args.exp_name, run_name))
    env = make_env(args.scenario, args.benchmark)
    agents = create_agents(env, args)

    episode_returns = []
    agent_returns = []
    train_step = 0
    
    while train_step <= args.max_train_steps:
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

            actions = [agent.act(obs) for obs, agent in zip(obs, agents)]
            new_obs, rewards, dones, infos = env.step(actions)
            done = all(dones)
            terminal = episode_step >= args.max_episode_len
            for i, agent in enumerate(agents):
                agent.experience(new_obs[i], actions[i], rewards[i], new_obs[i], dones[i])
            obs = new_obs

            for i, r in enumerate(rewards):
                cum_reward += r
                agents_cum_reward[i] += r

            # rendering environment
            if args.render and (len(episode_returns) % 1000 == 0):
                time.sleep(0.1)
                env.render()

            # train agents
            if train_step % args.train_every == 0:
                for agent in agents:
                    actor_loss, critic_loss = agent.update(agents)
                    writer.add_scalar('actor_loss', actor_loss, train_step)
                    writer.add_scalar('critic_loss', critic_loss, train_step)
        if len(episode_returns) % 1000 == 0:
            print('episode {} finished with a return of {:.2f}'.format(len(episode_returns), cum_reward))

        # store and log rewards
        episode_returns.append(cum_reward)
        agent_returns.append(agents_cum_reward)
        agent_rewards_dict = {a.name: r for a, r in zip(agents, agents_cum_reward)}
        writer.add_scalar('reward', cum_reward, train_step)
        writer.add_scalars('agent_rewards', agent_rewards_dict, train_step)

    print('Finished training with %d episodes' % len(episode_returns))

if __name__ == '__main__':
    train(parse_args())
