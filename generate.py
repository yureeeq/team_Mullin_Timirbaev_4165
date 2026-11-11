"""
Генерация пейзажей с помощью обученного генератора.
"""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image, make_grid

from models import Generator


def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели
    netG = Generator(nz=args.nz, ngf=args.ngf, nc=3).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "netG" in checkpoint:
        netG.load_state_dict(checkpoint["netG"])
    else:
        netG.load_state_dict(checkpoint)

    netG.eval()
    print(f"Модель загружена из {args.checkpoint}")

    # Генерация
    with torch.no_grad():
        noise = torch.randn(args.num_samples, args.nz, 1, 1, device=device)
        fake_images = netG(noise).cpu()

    # Сохранение сетки
    grid = make_grid(fake_images, nrow=int(args.num_samples ** 0.5), padding=2, normalize=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_path)
    print(f"Сгенерированные изображения сохранены в: {output_path}")

    # Отдельные изображения
    if args.save_individual:
        individual_dir = output_path.parent / "individual"
        individual_dir.mkdir(exist_ok=True)
        for i, img in enumerate(fake_images):
            save_image(img, individual_dir / f"sample_{i:03d}.png", normalize=True)
        print(f"Отдельные изображения сохранены в: {individual_dir}")


def main():
    parser = argparse.ArgumentParser(description="Генерация пейзажей DCGAN")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/generator_best.pt", help="Путь к чекпоинту генератора")
    parser.add_argument("--num-samples", type=int, default=64, help="Количество генерируемых изображений")
    parser.add_argument("--output", type=str, default="samples/generated_grid.png")
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--save-individual", action="store_true", help="Сохранить каждое изображение отдельно")
    args = parser.parse_args()

    generate(args)


if __name__ == "__main__":
    main()
