import json
import cv2
import numpy as np
from torch.utils.data import Dataset

# Python script to prepare to source, target and prompt.json for training.
# This script was adopted and adapted from:
# https://github.com/lllyasviel/ControlNet/blob/main/tutorial_dataset.py

# Replace 'TrainingSet' in all 3 locations with the name of the folder containing 'source', 'target' and 'prompt.json'

class MyDataset(Dataset):
    def __init__(self):
        with open('./TrainingSet/prompt.json', 'rt') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./TrainingSet/' + source_filename)
        target = cv2.imread('./TrainingSet/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

