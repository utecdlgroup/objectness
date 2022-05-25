from argparse import ArgumentParser
import math
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

from src.env import ImageClassificationEnv

parser = ArgumentParser()
parser.add_argument(
    '--action_type', type=str, default='velocity',
    choices=['position', 'velocity', 'acceleration'])
parser.add_argument(
    '--action_guess', type=str, default='random',
    choices=['constant_zero','constant_equal', 'random'])

if __name__ == '__main__':
    args = parser.parse_args()

    data_path = '/Users/cesar.salcedo/Documents/datasets/mnist'
    env = ImageClassificationEnv(
        data_path, 12, action_type=args.action_type,
        episode_end='dataset')

    # Prepare constants
    if args.action_type == 'position':
        spread_const = 0.05
    elif args.action_type == 'velocity':
        spread_const = 0.03
    elif args.action_type == 'acceleration':
        spread_const = 0.005

    if args.action_guess == 'constant':
        constant_q = True
    elif args.action_guess == 'random':
        constant_q = False

    # Prepare for animation
    fig, ax = plt.subplots()

    duration = 6
    fps = 15

    rewards = []

    def make_frame(t):
        move = torch.randn(3) * spread_const
        
        if args.action_guess == 'constant_zero':
            guess = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif args.action_guess == 'constant_equal':
            guess = torch.ones(10) * 0.1
        elif args.action_guess == 'random':
            guess = torch.randn(10)
            guess /= guess.sum()
        
        next_state, reward, done = env.step([move, guess])
        
        rewards.append(reward)
        
        ax.clear()
        
        ax.axis('off')
        ax.imshow(next_state.squeeze())
        
        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration = duration)
    animation.write_gif('sample2.gif', fps=fps)

    print("Rewards:")
    print(rewards)
