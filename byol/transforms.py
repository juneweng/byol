import random
import paddle
import paddle.vision.transforms as T
import paddle.vision.transforms.functional as F 
from PIL import ImageFilter
import numpy as np


class RandomApply():
    def __init__(self, transforms, p):
        # super(RandomApply, self).__init__()
        self.transforms = transforms
        self.p = p
    # def forward(self, x):
    #     if random.random() > self.p:
    #         return x
    #     return self.fn(x)
    def __call__(self, x):
        if random.random() > self.p:
            return x
        for t in self.transforms:
            x = t(x)
        return x 


class RandomGrayscale(object):
    """Randomly convert image to grayscale with a probability of p (default 0.1).

    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        PIL Image: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        num_output_channels = 1 if img.mode == 'L' else 3
        if random.random() < self.p:
            return F.to_grayscale(img, num_output_channels=num_output_channels)
        return img



class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        q = np.array(q)
        k = np.array(k)
        return np.vstack((q, k))
