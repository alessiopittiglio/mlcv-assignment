import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.tv_tensors import Mask

IGNORE_INDEX = 255

class StreetHazardsDataset(Dataset):
    KNOWN_CLASSES = [
        'building', 'fence', 'other', 'pedestrian', 'pole', 'road line',
        'road', 'sidewalk', 'vegetation', 'car', 'wall', 'traffic sign'
    ]
    CLASSES = KNOWN_CLASSES + ['anomaly']
    CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

    RAW_TO_TARGET_MAPPING = {
        0: IGNORE_INDEX,
        1: IGNORE_INDEX,
        2: CLASS_TO_ID['building'],
        3: CLASS_TO_ID['fence'],
        4: CLASS_TO_ID['other'],
        5: CLASS_TO_ID['pedestrian'],
        6: CLASS_TO_ID['pole'],
        7: CLASS_TO_ID['road line'],
        8: CLASS_TO_ID['road'],
        9: CLASS_TO_ID['sidewalk'],
        10: CLASS_TO_ID['vegetation'],
        11: CLASS_TO_ID['car'],
        12: CLASS_TO_ID['wall'],
        13: CLASS_TO_ID['traffic sign'],
        14: CLASS_TO_ID['anomaly'],
    }

    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples = []

        if split not in ['train', 'val', 'test']:
            raise ValueError(
                f"Invalid split '{split}'. Expected one of 'train', 'val', or 'test'."
            )

        if split == 'val':
            split_folder = 'train'
            odgt_filename = 'validation.odgt'
        else:
            split_folder = split
            odgt_filename = f'{split}.odgt'
        
        odgt_path = os.path.join(root_dir, split_folder, odgt_filename)

        with open(odgt_path, 'r') as f:
            data = json.loads(f.read())
        
        for line in data:
            img_path = root_dir / split_folder / line['fpath_img']
            ann_path = root_dir / split_folder / line['fpath_segm']
            if os.path.exists(img_path) and os.path.exists(ann_path):
                self.samples.append({'img': img_path, 'mask': ann_path})
            else:
                print(f"File not found for line: {line.strip()}")
        
        if not self.samples:
            print(f"Warning: No samples found in {split} split. Check the dataset path and files.")
        
        self.num_classes = 13 if split == 'test' else 12

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        sample = self.samples[idx]
        img_path = sample['img']
        mask_path = sample['mask']
        
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert('L')

            mask_raw = np.array(mask)
            mask_target = np.full_like(mask, IGNORE_INDEX)

            unique_ids = np.unique(mask_raw)
            for raw_id in unique_ids:
                target_id = self.RAW_TO_TARGET_MAPPING.get(raw_id, IGNORE_INDEX)
                mask_target[mask_raw == raw_id] = target_id

        except FileNotFoundError:
            print(f"File not found: {img_path} or {mask_path}")
            return None

        if self.transform:
            image, mask = self.transform(image, Mask(mask_target))
        
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze(0).long()
        
        return image, mask

if __name__ == "__main__":
    from pathlib import Path
    train_dataset = StreetHazardsDataset(
        root_dir = Path('data/'),
        split = 'train',
    )
    
    val_dataset = StreetHazardsDataset(
        root_dir = Path('data/'),
        split = 'val',
    )

    test_dataset = StreetHazardsDataset(
        root_dir = Path('data/'),
        split = 'test',
    )
