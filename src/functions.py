import torch
import torch.nn as nn
from datetime import datetime
import numpy as np
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image


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


def radar_factory(num_vars, frame='circle'):
    """Créer un radar plot avec num_vars axes."""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        # Remplir le plot (zone fermée)
        def fill(self, *args, **kwargs):
            closed = kwargs.pop('closed', True)
            return super().fill(closed=closed, *args, **kwargs)
        # Tracer les lignes en radar
        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                line.set_clip_on(False)
            return lines
        # Fixer les labels des variables autour du radar
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

    register_projection(RadarAxes)
    return theta


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
