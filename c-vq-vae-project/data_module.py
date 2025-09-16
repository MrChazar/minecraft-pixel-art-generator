import os
from PIL import Image
import io
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose
import torch
from datasets import load_from_disk


class MinecraftDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.transform = Compose([
            ToTensor()
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # get image
        img_bytes = self.dataframe.iloc[idx]['image']['bytes']
        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        img_tensor = self.transform(img)
        img_tensor = (img_tensor - 0.5) / 0.5  # normalization
        # get other columns
        is_block = float(self.dataframe.iloc[idx]['is_block'])
        is_block = torch.tensor(is_block, dtype=torch.float32)
        type_ = self.dataframe.iloc[idx]['type']
        type_tensor = torch.tensor(type_, dtype=torch.float32).flatten()
        colors = self.dataframe.iloc[idx]['colors']
        colors_tensor = torch.tensor(colors, dtype=torch.float32).flatten()

        return img_tensor, is_block, type_tensor, colors_tensor


def prepare_dataset_loader(ds, batch_size, num_workers=2):
    device = torch.accelerator.current_accelerator()
    if device is None:
        device = torch.device('cpu')
    dataset = MinecraftDataset(ds)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return device, dataset_loader


def download_file(file_path):
    if os.path.exists(file_path):
        dataset = load_from_disk(file_path)
        ds = dataset.to_pandas()
        return ds
    print("⚠️ The data folder couldn't be found, aborting download ⚠️")
    return None


def process_data(dataset):
    mask = []
    for index, row in dataset.iterrows():
        img = Image.open(io.BytesIO(row['image']['bytes']))
        mask.append((img.width == 16) & (img.height == 16))
    ds = dataset[mask]
    return ds
