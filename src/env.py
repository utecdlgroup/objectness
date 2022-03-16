import numpy as np
import torch
import torchvision
from torchvision import transforms

class ImageClassificationEnv():
    def __init__(self, data_path, m,
                 threshold=0.9,
                 time_diff=1.,
                 episode_end='batch'
        ):
        '''
        Image classification as an RL environment.

        params:
        - m: size of observation given to agent (m x m sized image).
        - threshold: value that must be surpassed by a class probability
            for it to be considered an agent's final decision.
        - time_diff: step of time (needed for acceleration computation).
        - episode_end: Determines when should an episode end. Options:
            * 'batch': An episode ends after each image classification.
            * 'dataset': An episode ends after all images in a dataset
                iteration are classified.
            * 'never': A continuous task, i.e. the episode never ends.
        '''

        super(ImageClassificationEnv, self).__init__()
        
        self.m = m
        self.time_diff = float(time_diff)
        self.threshold = threshold
        self.episode_end = episode_end

        if episode_end not in ['batch', 'dataset', 'never']:
            raise Exception("Invalid episode_end value '{}'. Available options are 'batch', 'dataset', and 'never'.".format(episode_end))
        
        # Dataset parameters
        self.data = torchvision.datasets.MNIST(data_path, transform=transforms.PILToTensor())
        
        sample = self.data[0]
        x, y = sample
        
        _, h, w = x.shape
        
        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(0., 1.),
            transforms.Pad([w, h]),
        ])
        
        self.data.transform = self.transform
        
        self.data_loader = torch.utils.data.DataLoader(self.data,
                                                       batch_size=1,
                                                       shuffle=True,
                                                       num_workers=2)
        
        # Note: Padding taken into account
        self.w = 3 * w
        self.h = 3 * h
        
        x_max = w            # Right end
        x_min = -x_max       # Left end
        y_max = h            # Top end
        y_min = -y_max       # Bottom end
        zoom_max = 10        # Front end
        zoom_min = -10       # Back end
        
        self.box = torch.Tensor([
            [x_min, y_min, zoom_min],
            [x_max, y_max, zoom_max]
        ])
        
        self.action_space_box = torch.Tensor([
            [-1, -1, -1],
            [1, 1, 1]
        ])
        
        self.iter = self.sample = self.position = self.velocities = None
        
        self.reset()
    
    
    def reset(self):
        self.iter = iter(self.data_loader)
        
        # Have next observation ready
        self._next_image()
        
        # Have agent state ready
        self.position = torch.zeros(3)
        self.velocities = torch.zeros(3)
    
    def step(self, action):
        accel, guess = action
        
        accel = torch.minimum(torch.maximum(accel, self.action_space_box[0]), self.action_space_box[1])
        
        self.velocities += accel * self.time_diff
        self.position += self.velocities * self.time_diff
        
        # Bound position to valid area
        self.position = torch.minimum(torch.maximum(self.position, self.box[0]), self.box[1])
        
        
        self.prediction += guess * self.time_diff
        self.prediction /= self.prediction.sum()
        
        decided_q = self.prediction.max() >= self.threshold
        if decided_q:
            if self.prediction.argmax() == self.label[0]:
                reward = 10
            else:
                reward = -10
                
            self._next_image()
        else:
            reward = 0
        
        next_state = self.get_view(self.position)
        
        done = False
        if self.episode_end == 'batch':
            done = decided_q
        elif self.episode_end == 'dataset':
            done = self.sample == None
            
        return next_state, reward, done
        
    def get_view(self, pos):
        # Bound point to positioning box
        bounded_pos = torch.minimum(torch.maximum(pos, self.box[0]), self.box[1])
        
        x, y, zoom = bounded_pos
        (x_min, y_min, zoom_min), (x_max, y_max, zoom_max) = self.box
        h_min, h_max = min(self.w, self.h) / 3, self.m
        
        scaled_zoom = torch.round(np.exp((np.log(h_min) * (zoom_max - zoom) + np.log(h_max) * (zoom - zoom_min)) / (zoom_max - zoom_min))).int()
        x_left = torch.round(self.w / 2 - scaled_zoom / 2 + x).int()
        y_top = torch.round(self.h / 2 - scaled_zoom / 2 - y).int()
        
        obs = transforms.functional.resized_crop(self.image, y_top, x_left, scaled_zoom, scaled_zoom, self.m)
        
        return obs
    
    def _next_image(self):
        # Have next observation ready
        self.sample = next(self.iter, None)
        
        if self.sample == None:
            self.iter = iter(self.data_loader)
            self.sample = next(self.iter, None)
            
        self.image = self.sample[0]
        self.label = self.sample[1]
        
        # Have agent state ready
        self.prediction = torch.ones(10) * 0.5
