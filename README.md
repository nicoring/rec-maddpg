## Recurrent Multi-Agent Deep Deterministic Policy Gradient (Rec-MADDPG)

This is the code for implementing the Rec-MADDPG algorithm presented in my MSc Dissertation "Communication and Cooperation in Decentralized Multi-AgentReinforcement Learning". It is configured to be run in conjunction with environments from the Multi-Agent Particle Environments (MPE).

### Installation
- Install requirements with `pip install -r requirements.txt`
- Install my adaption of the Multi-Agent Particle environment: https://github.com/nicoring/multiagent-particle-envs

### Usage
- `cd` into the `maddpg` directory
- Run the code with `python trainer.py --scenario SCENARIO_NAME`
- `python trainer.py --help` gives a description of all the available command line options.
- The code stores the success rates and returns as well as the policies of the agents.

### Code Structure
This repository contains the code for MADDPG and Rec-MADDPG in the maddpg directory, which contains the following files:
 - `trainer.py` which is the main file to run and contains the training logic.
 - `agent.py` contains the code for MADDPG and Rec-MADDPG agents.
 - `models.py` contains the code for the actor and policy networks.
 - `memory.py` contains the replay buffer code.
 - `distribitions.py` contains the code for the KL-divergence between Gumbel-Softmax distributions
 - Additionally, there are multiple run scripts.
