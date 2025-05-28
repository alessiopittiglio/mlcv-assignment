import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.tv_tensors import Mask
from torchvision.transforms.v2 import functional as F_v2

IGNORE_INDEX = 255

class StreetHazardsDataset(Dataset):
    KNOWN_CLASSES = [
        'building', 'fence', 'other', 'pedestrian', 'pole', 'road line',
        'road', 'sidewalk', 'vegetation', 'car', 'wall', 'traffic sign'
    ]
    CLASSES = KNOWN_CLASSES + ['anomaly']
    CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
    ANOMALY_ID = CLASS_TO_ID['anomaly']

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
        
        odgt_path = self.root_dir / split_folder / odgt_filename

        try:
            with open(odgt_path, 'r') as f:
                samples = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"ODGT file not found: {odgt_path}")
        
        for sample in samples:
            img_path = self.root_dir / split_folder / sample['fpath_img']
            ann_path = self.root_dir / split_folder / sample['fpath_segm']

            if img_path.exists() and ann_path.exists():
                self.samples.append({'img': str(img_path), 'mask': str(ann_path)})
            else:
                print(f"File not found. Img: {img_path}, Mask: {ann_path}")
        
        if not self.samples:
            print(
                f"Warning: No samples found in {split} split. "
                "Check the dataset path and files."
            )

        self.raw_to_target = self.RAW_TO_TARGET_MAPPING.copy()
        
        if self.split == 'train' or self.split == 'val':
            self.raw_to_target[14] = IGNORE_INDEX
            self.num_classes = len(self.KNOWN_CLASSES)
        if self.split == 'test':
            self.raw_to_target[14] = self.ANOMALY_ID
            self.num_classes = len(self.CLASSES)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        sample = self.samples[idx]
        img_path = sample['img']
        mask_path = sample['mask']
        
        try:
            image_pil = Image.open(img_path).convert("RGB")
            mask_pil = Image.open(mask_path).convert('L')
        except FileNotFoundError:
            print(f"File not found: {img_path} or {mask_path}")
            return None

        mask_np = np.array(mask_pil)
        mask_target = np.full_like(mask_np, IGNORE_INDEX)

        for raw_id, target_id in self.raw_to_target.items():
            mask_target[mask_np == raw_id] = target_id

        if self.transform:
            transformed_image, transformed_mask = self.transform(
                image_pil,
                Mask(mask_target)
            )
        else:
            transformed_image = F_v2.to_image(image_pil)
            transformed_image = F_v2.to_dtype(
                transformed_image, torch.float32, scale=True
            )
            transformed_mask = torch.from_numpy(mask_target)
        
        if transformed_mask.dtype != torch.long:
            transformed_mask = transformed_mask.long()
        
        return transformed_image, transformed_mask

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
