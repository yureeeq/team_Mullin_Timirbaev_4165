"""
Загрузчик датасета пейзажей.
Ожидает папку с изображениями (jpg/png).
Автоматически скачивает датасет Landscape Pictures с Kaggle через kagglehub, если папка пустая.
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import kagglehub


class LandscapeDataset(Dataset):
    def __init__(self, root: str, image_size: int = 64, transform=None):
        self.root = Path(root)
        self.image_size = image_size
        self.image_paths = sorted(
            list(self.root.glob("*.jpg")) +
            list(self.root.glob("*.jpeg")) +
            list(self.root.glob("*.png"))
        )
        if len(self.image_paths) == 0:
            raise ValueError(f"В папке {root} не найдено изображений!")

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)


def download_landscape_dataset(output_dir: str = "data") -> str:
    """Скачивает датасет Landscape Pictures (arnaud58) и возвращает путь к папке с изображениями."""
    print("Скачивание датасета Landscape Pictures с Kaggle...")
    # force_download=True чтобы перезаписать если нужно
    path = kagglehub.dataset_download(
        "arnaud58/landscape-pictures",
        output_dir=output_dir,
        force_download=True,
    )
    print(f"Датасет скачан в: {path}")
    return path


def get_dataloader(
    data_root: str = "data",
    batch_size: int = 64,
    image_size: int = 64,
    num_workers: int = 2,
    auto_download: bool = True,
) -> DataLoader:
    data_path = Path(data_root)
    data_path.mkdir(parents=True, exist_ok=True)

    # Если папка пустая — скачиваем
    images = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png"))
    if len(images) == 0 and auto_download:
        download_landscape_dataset(str(data_path))

    dataset = LandscapeDataset(str(data_path), image_size=image_size)
    print(f"Загружено изображений: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader
