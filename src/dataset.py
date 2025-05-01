import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class StreetHazardsDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []

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

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['img']
        mask_path = sample['mask']

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert('L')
        except FileNotFoundError:
            print(f"File not found: {img_path} or {mask_path}")
            return None
        
        if self.transform:
            image, mask = self.transform(image, mask)

        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze(0).long()
        elif isinstance(mask, Image.Image):
            mask = torch.from_numpy(np.array(mask)).long()

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
