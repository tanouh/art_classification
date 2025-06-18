import torch
import torch.nn as nn
from datetime import datetime
import torchvision.transforms.functional as F
import random

def log_message(message, log_path):
    timestamp = datetime.now().strftime("%d/%m/%y, %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"{timestamp} : {message} \n")


class VGG16FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(VGG16FeatureExtractor, self).__init__()
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(*list(model.classifier.children())[:-3])  # Up to ReLU after first FC

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x  # Feature vector

class LetterboxPad:
    def __init__(self, size, fill_mode='reflect'):
        self.size = size  # (224, 224)
        self.fill_mode = fill_mode

    def __call__(self, img):
        # Resize while maintaining aspect ratio
        original_width, original_height = img.size
        target_width, target_height = self.size

        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        img = F.resize(img, (new_height, new_width))

        # Compute padding
        pad_left = (target_width - new_width) // 2
        pad_top = (target_height - new_height) // 2
        pad_right = target_width - new_width - pad_left
        pad_bottom = target_height - new_height - pad_top

        # Pad with reflection
        img = F.pad(img, padding=[pad_left, pad_top, pad_right, pad_bottom], padding_mode=self.fill_mode)

        return img


class RandomFixedCrop:
    def __init__(self, size):
        self.size = size  # (height, width) or int
    
    def __call__(self, img):
        w, h = img.size
        th, tw = self.size if isinstance(self.size, tuple) else (self.size, self.size)

        if w < tw or h < th:
            raise ValueError("Crop size must be smaller than image size")

        # 5 possible crop positions
        options = [
            (0, 0),                           # top-left
            (w - tw, 0),                      # top-right
            (0, h - th),                      # bottom-left
            (w - tw, h - th),                 # bottom-right
            ((w - tw) // 2, (h - th) // 2)    # center
        ]

        i, j = random.choice(options)
        return F.crop(img, j, i, th, tw)  # F.crop(img, top, left, height, width)
