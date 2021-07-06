import gym
import wandb
import yaml
import argparse

from ..src.agent.ddpg.ddpg import DDPGAgent


def train(configs):
    env = gym.make('CartPole-v0')
    env.reset()

    # --- training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        help='config file(yaml) path',
                        type=str, default='gym_test.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)

    train(configs)


     