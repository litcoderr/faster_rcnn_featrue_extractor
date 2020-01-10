import os
import glob
from pathlib import Path

from torchvision import models

root = Path('./').resolve()  # Current Working Directory
image_root = os.path.join(str(root), '../dramaqa_loader/data/AnotherMissOh/AnotherMissOh_images/')

image_paths = []

for episode in [e for e in os.listdir(image_root) if e != '.DS_Store']:
    episode_path = os.path.join(image_root, episode)
    for scene in [s for s in os.listdir(episode_path) if s != '.DS_Store']:
        scene_path = os.path.join(episode_path, scene)
        for clip  in [c for c in os.listdir(scene_path) if c != '.DS_Store']:
            clip_path = os.path.join(scene_path, clip)
            image_path_partial = glob.glob('{}/*.jpg'.format(clip_path))
            image_paths += image_path_partial

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().cuda()

