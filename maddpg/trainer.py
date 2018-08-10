import argparse
import time
import os
import signal
import math
import itertools as it
from collections import defaultdict

import numpy as np
from tensorboardX import SummaryWriter
import torch
import gym
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

from agents import MADDPGAgent, MARDPGAgent, RandomAgent, SpreadScriptedAgent
from models import Actor, Critic, LSTMCritic


def parse_args():
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument('--scenario', type=str, default='simple', help='name of the scenario script')
    parser.add_argument('--max-train-steps', type=int, default=1_000_000, help='maximum episode length')
    parser.add_argument('--max-episode-len', type=int, default=25, help='maximum episode length')
    parser.add_argument('--num-adversaries', type=int, default=0, help='number of adversaries')
    parser.add_argument('--render', default=False, action='store_true', help='display agent policies')
    parser.add_argument('--eval', dest='evaluate', default=False, action='store_true', help='run agent policy without noise and training')
    parser.add_argument('--train-every', type=int, default=100, help='simulation steps in between network updates')
    parser.add_argument('--hidden', type=int, default=64, help='number of hidden units in actor and critic')
    parser.add_argument('--batch-size', type=int, default=1024, help='size of minibatch that is sampled from replay buffer')
    parser.add_argument('--memory-size', type=int, default=50_000, help='size of replay memory')
    parser.add_argument('--tau', type=float, default=0.01, help='update rate for exponential update of target network params')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor for training of critic')
    parser.add_argument('--save-dir', type=str, default='results')
    parser.add_argument('--exp-name', type=str, default='test', help='name of experiment')
    parser.add_argument('--exp-run-num', type=str, default='', help='run number of experiment gets appended to log dir')
    parser.add_argument('--num-runs', type=int)
    parser.add_argument('--train-steps', type=int, default=1, help='how many train steps of the networks are done')
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=-1)
    parser.add_argument('--lr-actor', type=float, default=1e-3)
    parser.add_argument('--lr-critic', type=float, default=1e-2)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--resume', help='dirname of saved state')
    parser.add_argument('--load-memories', default=False, action='store_true')
    parser.add_argument('--local-obs', default=False, action='store_true')
    parser.add_argument('--local-actions', default=False, action='store_true')
    parser.add_argument('--conf', type=int)
    parser.add_argument('--conf-name', type=str)
    parser.add_argument('--sparse-reward', default=True, action='store_false', dest='shaped')

    ## Agent modeling
    parser.add_argument('--use-agent-models', default=False, action='store_true')
    parser.add_argument('--max-past', type=int, default=5000)
    parser.add_argument('--modeling-train-steps', type=int, default=20)
    parser.add_argument('--modeling-batch-size', type=int, default=64)
    parser.add_argument('--modeling-lr', type=float, default=1e-4)
    parser.add_argument('--sigma-noise', type=float)
    parser.add_argument('--temp-noise', type=float)

    ## Recurrent agents
    parser.add_argument('--recurrent-critic', default=False, action='store_true')
    parser.add_argument('--recurrent-agent', default=False, action='store_true')
    parser.add_argument('--recurrent-agent-model', default=False, action='store_true')

    return parser.parse_args()


def make_env(scenario_name, shaped=True):
    scenario = scenarios.load(scenario_name + '.py').Scenario()
    world = scenario.make_world()
    if scenario.has_shaped_reward:
        reward_callback = scenario.shaped_reward if shaped else scenario.sparse_reward
    else:
        reward_callback = scenario.reward
    env = MultiAgentEnv(world,
                        reset_callback=scenario.reset_world,
                        reward_callback=reward_callback,
                        observation_callback=scenario.observation,
                        done_callback=scenario.done)
    return env


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
        if params.recurrent_critic:
            critic = LSTMCritic(n_critic_inputs, params.hidden)
            agent = MARDPGAgent(i, 'agent_%d' % i, env, actor, critic, params)
        else:
            critic = Critic(n_critic_inputs, params.hidden)
            agent = MADDPGAgent(i, 'agent_%d' % i, env, actor, critic, params)
        agents.append(agent)
    return agents


def save_agent_states(dirname, agents, save_models=False):
    states, models = zip(*[agent.get_state() for agent in agents])
    states_filename = os.path.join(dirname, 'states.pth.tar')
    torch.save(states, states_filename)
    if save_models:
        models_filename = os.path.join(dirname, 'models.pth.tar')
        torch.save(models, models_filename)


def load_agent_states(dirname, agents, load_models=False):
    states_filename = os.path.join(dirname, 'states.pth.tar')
    states = torch.load(states_filename, map_location='cpu')
    if load_models:
        models_filename = os.path.join(dirname, 'models.pth.tar')
        models = torch.load(models_filename, map_location='cpu')
    for i, agent in enumerate(agents):
        state = {}
        state['state_dicts'] = states[i]
        if load_models:
            state['models'] = models[i]
        agent.load_state(state)


def evaluate(env, agents, num_runs, args):
    dones_sum = 0.0
    rewards_all = []
    for _ in range(num_runs):
        obs = env.reset()
        done = False
        terminal = False
        episode_step = 0
        cum_rewards = np.zeros(len(agents), dtype=np.float)
        while not (done or terminal):
            # epsiode step count
            episode_step += 1
            # act with all agents in environment and receive observation and rewards
            actions = [agent.act(o, explore=False) for o, agent in zip(obs, agents)]
            new_obs, rewards, dones, _ = env.step(actions)
            cum_rewards += rewards
            done = all(dones)
            terminal = episode_step >= args.max_episode_len
            obs = new_obs
        rewards_all.append(cum_rewards)
        if done:
            dones_sum += 1
    return dones_sum / num_runs, np.mean(rewards_all, axis=0)


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def eval_msg(start_time, train_step, episode_count, agents, sr_mean, rewards):
    msg = 'time: {}, step {}, episode {}: success rate: {:.3f}, return: {:.2f}, '
    msg += ', '.join([a.name + ': {:.2f}' for a in agents])
    cum_reward = rewards.sum()
    print(msg.format(time_since(start_time), train_step, episode_count, sr_mean, cum_reward, *rewards))


def add_entropies(writer, obs, agents, train_step):
    entropies = {}
    for o, agent in zip(obs, agents):
        device = next(agent.actor.parameters()).device
        o = torch.tensor(o, dtype=torch.float, requires_grad=False, device=device)
        ps = agent.actor.prob_dists(o)
        agent_entropies = []
        for p in ps:
            agent_entropies.append(p.base_dist._categorical.entropy().detach().cpu().numpy())
        entropies[agent.name] = np.mean(agent_entropies)
    writer.add_scalars('policy_entropy', entropies, train_step)


def train(args):
    def signal_handling(signum, frame):
        nonlocal terminated
        terminated = True
    signal.signal(signal.SIGINT, signal_handling)

    env = make_env(args.scenario, args.shaped)
    agents = create_agents(env, args)
    if args.use_agent_models:
        for agent in agents:
            agent.init_agent_models(agents)
    

    # load state of agents if state file is given
    if args.resume:
        load_agent_states(args.resume, agents, load_models=args.use_agent_models)
        args.exp_name, args.exp_run_num = os.path.normpath(args.resume).split('/')[-2:]
        args.save_dir = os.path.join(args.resume[:-2])

    # logging
    run_name = str(np.datetime64('now')) if args.exp_run_num == '' else args.exp_run_num
    writer = SummaryWriter(log_dir=os.path.join('tensorboard-logs', args.exp_name, run_name))
    dirname = os.path.join(args.save_dir, args.exp_name, run_name)
    os.makedirs(dirname, exist_ok=True)
    train_rewards_file = os.path.join(dirname, 'train_rewards.csv')
    eval_rewards_file = os.path.join(dirname, 'eval_rewards.csv')
    success_rate_file = os.path.join(dirname, 'success_rate.csv')
    kl_divergence_file = os.path.join(dirname, 'kl_divergence.csv')
    times_file = os.path.join(dirname, 'times.csv')
    for rewards_file in [train_rewards_file, eval_rewards_file]:
        if not os.path.isfile(rewards_file):
            with open(rewards_file, 'w') as f:
                line = ','.join(['step', 'cum_reward'] + [a.name for a in agents]) + '\n'
                f.write(line)
    if not os.path.isfile(success_rate_file):
        with open(success_rate_file, 'w') as f:
            f.write('step,success_rate\n')
    if args.use_agent_models and not os.path.isfile(kl_divergence_file):
        with open(kl_divergence_file, 'w') as f:
            headers = ['step']
            for agent in agents:
                for model_idx, model in agent.agent_models.items():
                    for i in range(len(model.action_split)):
                        headers.append('%d_%d_%d' % (agent.index, model_idx, i))
            f.write(','.join(headers) + '\n')

    episode_count = 0
    train_step = 0
    terminated = False

    start_time = time.time()
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
                add_entropies(writer, obs, agents, train_step)
            new_obs, rewards, dones, _ = env.step(actions)
            done = all(dones)
            terminal = episode_step >= args.max_episode_len

            # store rewards
            for i, reward in enumerate(rewards):
                cum_reward += reward
                agents_cum_reward[i] += reward

            # rendering environment
            if args.render and ((episode_count % 500 == 0) or args.evaluate):
                time.sleep(0.1)
                env.render()

            # store tuples (observation, action, reward, next observation, is done) for each agent
            for i, agent in enumerate(agents):
                agent.experience(episode_count, obs[i], actions[i], rewards[i], new_obs[i], dones[i])

            # store current observation for next step
            obs = new_obs

            if args.evaluate:
                continue

            # train agents
            if train_step % args.train_every == 0:
                for _ in range(args.train_steps):
                    losses = defaultdict(dict)
                    all_kls = []
                    for agent in agents:
                        actor_loss, critic_loss, model_loss, model_kls = agent.update(agents)
                        losses['actor_loss'][agent.name] = actor_loss
                        losses['critic_loss'][agent.name] = critic_loss
                        if args.use_agent_models:
                            losses['model_loss'][agent.name] = model_loss
                            kls_dict = {}
                            for idx, kls in model_kls:
                                for i, kl in enumerate(kls):
                                    kls_dict['%s_%i' % (agents[idx].name, i)] = kl
                                    all_kls.append(kl.item())
                            if args.debug:
                                writer.add_scalars('kl_%s' % agent.name, kls_dict, train_step)
                    if args.use_agent_models:
                        with open(kl_divergence_file, 'a') as f:
                            line = ','.join(map(str, [train_step] + all_kls)) + '\n'
                            f.write(line)
                    if args.debug:
                        for name, loss_dict in losses.items():
                            writer.add_scalars(name, loss_dict, train_step)

            if train_step % args.eval_every == 0 or args.evaluate:
                sr_mean, rewards = evaluate(env, agents, 50, args)
                if not args.evaluate:
                    with open(success_rate_file, 'a') as f:
                        line = '{},{}\n'.format(train_step, sr_mean)
                        f.write(line)
                    with open(eval_rewards_file, 'a') as f:
                        line = ','.join(map(str, [train_step, rewards.sum()] + list(rewards))) + '\n'
                        f.write(line)
                    with open(times_file, 'a') as f:
                        f.write(str(time.time()) + '\n')
                if args.debug:
                    writer.add_scalar('success_rate', sr_mean, train_step)
                eval_msg(start_time, train_step, episode_count, agents, sr_mean, rewards)

        episode_count += 1
        if args.evaluate:
            sr_mean, rewards = evaluate(env, agents, 50, args)
            eval_msg(start_time, train_step, episode_count, agents, sr_mean, rewards)
            continue

        # save agent states
        if (args.save_every != -1) and (episode_count % args.save_every == 0):
            save_agent_states(dirname, agents, save_models=args.use_agent_models)

        # save cumulatitive reward and agent rewards
        with open(train_rewards_file, 'a') as f:
            line = ','.join(map(str, [train_step, cum_reward] + agents_cum_reward)) + '\n'
            f.write(line)

        # log rewards for tensorboard
        if args.debug:
            agent_rewards_dict = {a.name: r for a, r in zip(agents, agents_cum_reward)}
            writer.add_scalar('reward', cum_reward, train_step)
            writer.add_scalars('agent_rewards', agent_rewards_dict, train_step)

    print('Finished training with %d episodes' % episode_count)


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
    return args


def run_config2(args, num):
    scenario_names = [
        'simple',
        'simple_adversary',
        'simple_crypto',
        'simple_push',
        'simple_reference',
        'simple_speaker_listener',
        'simple_spread',
        'simple_tag',
        'simple_world_comm',
        'simple_spread_comm'
    ]
    obs_actions_model = [[False, True, False],[True, True, False], [False, False, True], [False, False, False]]
    config = list(it.product(scenario_names, obs_actions_model))[num]
    scenario_name, (local_obs, local_actions, use_models) = config
    print('running conf: (scenario: %s, local_obs: %r, local_actions: %r, use_models: %r)' % (scenario_name, local_obs, local_actions, use_models))
    args.scenario = scenario_name
    args.local_obs = local_obs
    args.local_actions = local_actions
    args.use_agent_models = use_models
    args.exp_name = '%s_%s_%s_%s' % (scenario_name, str(local_obs), str(local_actions), str(use_models))
    return args


def run_config3(args, num):
    scenario_names = [
        # 'simple',
        # 'simple_adversary',
        # 'simple_crypto',
        # 'simple_push',
        # 'simple_reference',
        'simple_speaker_listener',
        'simple_spread',
        # 'simple_tag',
        # 'simple_world_comm',
        'simple_spread_comm'
    ]
    obs_actions_model = [[False, False, True], [False, False, False]]
    noises = [0.0, 0.05, 0.1, 0.2, 0.3]
    config = list(it.product(scenario_names, obs_actions_model, noises))[num]
    scenario_name, (local_obs, local_actions, use_models), noise = config
    print('running conf: (scenario: %s, local_obs: %r, local_actions: %r, use_models: %r, noise: %f)' % (scenario_name, local_obs, local_actions, use_models, noise))
    args.scenario = scenario_name
    args.local_obs = local_obs
    args.local_actions = local_actions
    args.use_agent_models = use_models
    if noise > 0.0:
        args.obfuscation_noise = noise
    args.exp_name = '%s_%s_%s_%s_%f' % (scenario_name, str(local_obs), str(local_actions), str(use_models), noise)
    return args


def run_config4(args, num):
    scenario_names = [
        # 'simple_spread',
        # 'simple_spread_comm',
        # 'simple_reference',
        'simple_speaker_listener'
    ]
    use_models = [True]
    recurrent = [True, False]
    no_noise = [(None, None)]
    obs_noise_sigmas = [0.2, 0.4, 0.6, 0.8, 1.0]
    action_noise_temps = [2.0, 5.0, 8.0, 11.0, 13.0]
    noises_both = list(zip(obs_noise_sigmas, action_noise_temps))
    noises_only_actions = [(None, an) for an in action_noise_temps]
    noises_only_obs = [(on, None) for on in obs_noise_sigmas]

    noises = noises_only_actions + noises_only_obs

    config = list(it.product(scenario_names, use_models, recurrent, noises))[num]
    print('running conf: (scenario: %s, models: %r, recurrent: %r, noise: %r' % config)
    scenario, use_agent_models, recurrent, (noise_sigma, noise_temp) = config
    args.scenario = scenario
    args.use_agent_models = use_agent_models
    args.recurrent_critic = recurrent
    args.sigma_noise = noise_sigma
    args.temp_noise = noise_temp
    if recurrent:
        args.batch_size = 256
    else:
        args.batch_size = 1024
    args.max_train_steps = 1_500_000
    noise_sigma = '{:.2f}'.format(noise_sigma) if noise_sigma is not None else str(noise_sigma)
    noise_temp = '{:.2f}'.format(noise_temp) if noise_temp is not None else str(noise_temp)
    args.exp_name = '{}_{}_{}_{}_{}'.format(scenario, str(use_agent_models), str(recurrent), noise_sigma, noise_temp)
    return args


def run_reference(args, num):
    scenario_names = [
        'simple_reference',
    ]
    use_models = [True]
    recurrent = [True, False]
    no_noise = [(None, None)]
    obs_noise_sigmas = [0.03, 0.06, 0.1, 0.13, 0.16]
    noises_only_obs = [(on, None) for on in obs_noise_sigmas]

    noises = noises_only_obs

    config = list(it.product(scenario_names, use_models, recurrent, noises))[num]
    print('running conf: (scenario: %s, models: %r, recurrent: %r, noise: %r' % config)
    scenario, use_agent_models, recurrent, (noise_sigma, noise_temp) = config
    args.scenario = scenario
    args.use_agent_models = use_agent_models
    args.recurrent_critic = recurrent
    args.sigma_noise = noise_sigma
    args.temp_noise = noise_temp
    if recurrent:
        args.batch_size = 256
    else:
        args.batch_size = 1024
    args.max_train_steps = 1_500_000
    noise_sigma = '{:.2f}'.format(noise_sigma) if noise_sigma is not None else str(noise_sigma)
    noise_temp = '{:.2f}'.format(noise_temp) if noise_temp is not None else str(noise_temp)
    args.exp_name = '{}_{}_{}_{}_{}'.format(scenario, str(use_agent_models), str(recurrent), noise_sigma, noise_temp)
    return args


def run_local_action_temp(args, num):
    scenario_names = [
        'simple_speaker_listener',
        'simple_reference',
        'simple_spread_comm'
    ]
    use_models = [False]
    recurrent = [True, False]
    noises = [[None, None]]

    config = list(it.product(scenario_names, use_models, recurrent, noises))[num]
    print('running conf: (scenario: %s, models: %r, recurrent: %r, noise: %r' % config)
    scenario, use_agent_models, recurrent, (noise_sigma, noise_temp) = config
    args.scenario = scenario
    args.use_agent_models = use_agent_models
    args.recurrent_critic = recurrent
    args.sigma_noise = noise_sigma
    args.temp_noise = noise_temp
    args.local_actions = True
    if recurrent:
        args.batch_size = 256
    else:
        args.batch_size = 1024
    args.max_train_steps = 1_500_000
    noise_sigma = '{:.2f}'.format(noise_sigma) if noise_sigma is not None else str(noise_sigma)
    noise_temp = '{:.2f}'.format(noise_temp) if noise_temp is not None else str(noise_temp)
    args.exp_name = '{}_{}_{}_{}_{}'.format(scenario, str(use_agent_models), str(recurrent), noise_sigma, noise_temp)
    return args


def main():
    args = parse_args()

    experiments = {
        'reference': run_reference,
        'local_action_temp': run_local_action_temp
    }

    if args.conf is not None:
        if args.conf_name:
            args = experiments[args.conf_name](args, args.conf-1)
        else:
            args = run_config4(args, args.conf-1)

    if args.num_runs is not None:
        train_multiple_times(args, args.num_runs)
    else:
        train(args)


if __name__ == '__main__':
    main()
