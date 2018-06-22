import argparse
import time
import os
import signal
import itertools as it

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
    parser.add_argument('--evaluate', default=False, action='store_true', help='run agent policy without noise and training')
    parser.add_argument('--train-every', type=int, default=100, help='simulation steps in between network updates')
    parser.add_argument('--benchmark', default=False, action='store_true', help='')
    parser.add_argument('--hidden', type=int, default=64, help='number of hidden units in actor and critic')
    parser.add_argument('--batch-size', type=int, default=1024, help='size of minibatch that is sampled from replay buffer')
    parser.add_argument('--memory-size', type=int, default=1_000_000, help='size of replay memory')
    parser.add_argument('--tau', type=float, default=0.01, help='update rate for exponential update of target network params')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor for training of critic')
    parser.add_argument('--sigma', type=float, default=0.1, help='std deviation of Gaussian noise process for exploration')
    parser.add_argument('--save-dir', type=str, default='results')
    parser.add_argument('--exp-name', type=str, default='test', help='name of experiment')
    parser.add_argument('--exp-run-num', type=str, default='', help='run number of experiment gets appended to log dir')
    parser.add_argument('--num-runs', type=int)
    parser.add_argument('--train-steps', type=int, default=1, help='how many train steps of the networks are done')
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=-1)
    parser.add_argument('--lr-actor', type=float, default=1e-3)
    parser.add_argument('--lr-critic', type=float, default=1e-2)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--resume', help='dirname of saved state')
    parser.add_argument('--load-memories', default=False, action='store_true')
    parser.add_argument('--local-obs', default=False, action='store_true')
    parser.add_argument('--local-actions', default=False, action='store_true')
    parser.add_argument('--conf', type=int)
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

    action_splits = []
    for action_space in env.action_space:
        if isinstance(action_space, gym.spaces.Discrete):
            action_splits.append([action_space.n])
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            action_splits.append(action_space.nvec)

    n_obs_each = [o.shape[0] for o in env.observation_space]
    n_actions_each = [sum(a) for a in action_splits]

    for i in range(n_agents):
        if params.local_obs:
            n_obs = n_obs_each[i]
        else:
            n_obs = sum(n_obs_each)
        if params.local_actions:
            n_actions = n_actions_each[i]
        else:
            n_actions = sum(n_actions_each)
        n_critic_inputs = n_obs + n_actions
        n_observations = env.observation_space[i].shape[0]
        actor = Actor(n_observations, action_splits[i], params.hidden)
        critic = Critic(n_critic_inputs, params.hidden)
        agent = MaddpgAgent(i, 'agent_%d' % i, actor, critic, params)
        agents.append(agent)
    return agents


def save_agent_states(dirname, agents, save_memories=False):
    states_and_memories = [agent.get_state() for agent in agents]
    states, memories = zip(*states_and_memories)
    states_filename = os.path.join(dirname, 'states.pth.tar')
    memories_filename = os.path.join(dirname, 'memories.pth.tar')
    torch.save(states, states_filename)
    if save_memories:
        torch.save(memories, memories_filename)


def load_agent_states(dirname, agents, load_memories=False):
    states_filename = os.path.join(dirname, 'states.pth.tar')
    memories_filename = os.path.join(dirname, 'memories.pth.tar')
    states = torch.load(states_filename)
    if load_memories:
        memories = torch.load(memories_filename)
    for i, agent in enumerate(agents):
        state = {}
        state['state_dicts'] = states[i]
        if load_memories:
            state['memory'] = memories[i]
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
        load_agent_states(args.resume, agents)
        args.exp_name, args.exp_run_num = os.path.normpath(args.resume).split('/')[-2:]
        args.save_dir = os.path.join(args.resume[:-2])

    # logging
    run_name = str(np.datetime64('now')) if args.exp_run_num == '' else args.exp_run_num
    writer = SummaryWriter(log_dir=os.path.join('tensorboard-logs', args.exp_name, run_name))
    dirname = os.path.join(args.save_dir, args.exp_name, run_name)
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
            actions = [agent.act(o, explore=not args.evaluate) for o, agent in zip(obs, agents)]
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
            if args.render and (len(episode_returns) % args.eval_every == 0):
                time.sleep(0.1)
                env.render()

            # store tuples (observation, action, reward, next observation, is done) for each agent
            for i, agent in enumerate(agents):
                agent.experience(obs[i], actions[i], rewards[i], new_obs[i], dones[i])

            # store current observation for next step
            obs = new_obs

            if args.evaluate:
                continue

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

        # store and log rewards
        episode_returns.append(cum_reward)
        agent_returns.append(agents_cum_reward)

        if args.evaluate:
            continue

        # save agent states
        if (args.save_every != -1) and (len(episode_returns) % args.save_every == 0):
            save_agent_states(dirname, agents)

        # save cumulatitive reward and agent rewards
        with open(rewards_file, 'a') as f:
            line = ','.join(map(str, [cum_reward] + agents_cum_reward)) + '\n'
            f.write(line)

        # log rewards for tensorboard
        agent_rewards_dict = {a.name: r for a, r in zip(agents, agents_cum_reward)}
        writer.add_scalar('reward', cum_reward, train_step)
        writer.add_scalars('agent_rewards', agent_rewards_dict, train_step)

    print('Finished training with %d episodes' % len(episode_returns))


def train_multiple_times(args, num_runs):
    for i in range(num_runs):
        args.exp_run_num = str(i)
        dirname = os.path.join(args.save_dir, args.exp_name, str(i))
        if os.path.isdir(dirname):
            print('Skipping: ' + dirname)
            continue
        train(args)


def run_config(args, num):
    scenario_names = [
        'simple',
        'simple_adversary',
        'simple_crypto',
        'simple_push',
        'simple_reference',
        'simple_speaker_listener',
        'simple_spread',
        'simple_tag',
        'simple_world_comm'
    ]
    local_obs_choices = [True, False]
    local_actions_choices = [True, False]
    config = list(it.product(scenario_names, local_obs_choices, local_actions_choices))[num]
    print('running conf: (scenario: %s, local_obs: %r, local_actions: %r)' % config)
    scenario_name, local_obs, local_actions = config
    args.scenario = scenario_name
    args.local_obs = local_obs
    args.local_actions = local_actions
    args.exp_name = '%s_%s_%s' % (scenario_name, str(local_obs), str(local_actions))
    train_multiple_times(args, args.num_runs)


def main():
    args = parse_args()
    if args.conf is not None:
        run_config(args, args.conf)
    elif args.num_runs is not None:
        train_multiple_times(args, args.num_runs)
    else:
        train(args)


if __name__ == '__main__':
    main()
