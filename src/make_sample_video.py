import math
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

from src.env import ImageClassificationEnv

if __name__ == '__main__':
    data_path = '/Users/cesar.salcedo/Documents/datasets/mnist'
    env = ImageClassificationEnv(data_path, 12)
    
    fig, ax = plt.subplots()
    
    duration = 6
    fps = 15
    
    def make_frame(t):
        x = env.get_view(torch.Tensor([5 * math.sin(4 * 3.1416 * t / duration), 5 * math.cos(4 * 3.1416 * t / duration), 10 * math.sin(2 * 3.1416 * t / duration)]))[0, 0]
        
        ax.clear()
        
        ax.axis('off')
        ax.imshow(x.squeeze())
        
        return mplfig_to_npimage(fig)
    
    animation = VideoClip(make_frame, duration = duration)
    animation.ipython_display(fps=fps)
