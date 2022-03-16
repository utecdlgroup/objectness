import math
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

from src.env import ImageClassificationEnv

if __name__ == '__main__':
    data_path = '/Users/cesar.salcedo/Documents/datasets/mnist'
    env = ImageClassificationEnv(data_path, 12, action_type='acceleration')

    fig, ax = plt.subplots()

    duration = 6
    fps = 15

    rewards = []

    def make_frame(t):
        accel = torch.randn(3) * 0.05
        # accel = torch.Tensor([0.1 * math.sin(4 * 3.1416 * t / duration) - 0.05, 0.1 * math.cos(4 * 3.1416 * t / duration) - 0.05, math.sin(2 * 3.1416 * t / duration) / 2 + 0.5])

        # guess = torch.Tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        # guess = torch.ones(10) * 0.1
        guess = torch.randn(10)
        guess /= guess.sum()
        
        next_state, reward, done = env.step([accel, guess])
        
        rewards.append(reward)
        
        ax.clear()
        
        ax.axis('off')
        ax.imshow(next_state.squeeze())
        
        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration = duration)
    animation.ipython_display(fps = fps, loop = True, autoplay = True)

    print("Rewards:")
    print(rewards)