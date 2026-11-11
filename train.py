"""
Обучение DCGAN на пейзажах.
Поддерживает ClearML для логирования графиков (losses, samples).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image, make_grid
import numpy as np

from models import Generator, Discriminator
from dataset import get_dataloader

try:
    from clearml import Task
except ImportError:
    Task = None


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # ClearML
    task = None
    if args.use_clearml and Task is not None:
        task = Task.init(
            project_name=args.clearml_project,
            task_name=args.clearml_task,
            auto_connect_frameworks={"pytorch": True, "matplotlib": True},
        )
        task.connect(args)
        print("ClearML Task инициализирован")
    elif args.use_clearml:
        print("ClearML не установлен. Запустите: pip install clearml")

    # Данные
    dataloader = get_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        auto_download=True,
    )

    # Модели
    netG = Generator(nz=args.nz, ngf=args.ngf, nc=3).to(device)
    netD = Discriminator(ndf=args.ndf, nc=3).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss и оптимизаторы
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Фиксированный шум для визуализации прогресса
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    real_label = 1.0
    fake_label = 0.0

    # Папки
    checkpoint_dir = Path(args.checkpoint_dir)
    samples_dir = Path(args.samples_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    G_losses = []
    D_losses = []

    print("Начало обучения...")
    for epoch in range(args.epochs):
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            b_size = real_images.size(0)

            # ===== Train Discriminator =====
            netD.zero_grad()
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_images)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            # ===== Train Generator =====
            netG.zero_grad()
            label.fill_(real_label)  # Generator wants D to think fakes are real
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            # Логирование
            if i % args.log_interval == 0:
                print(
                    f"[{epoch+1}/{args.epochs}][{i}/{len(dataloader)}] "
                    f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}"
                )
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                if task is not None:
                    task.get_logger().report_scalar("Loss", "Discriminator", value=errD.item(), iteration=epoch * len(dataloader) + i)
                    task.get_logger().report_scalar("Loss", "Generator", value=errG.item(), iteration=epoch * len(dataloader) + i)

        # Сохранение сэмплов после каждой эпохи
        with torch.no_grad():
            fake_samples = netG(fixed_noise).detach().cpu()
            grid = make_grid(fake_samples, padding=2, normalize=True)
            sample_path = samples_dir / f"epoch_{epoch+1:03d}.png"
            save_image(grid, sample_path)
            print(f"Сохранён сэмпл: {sample_path}")

            if task is not None:
                task.get_logger().report_image(
                    "Generated Samples",
                    f"Epoch {epoch+1}",
                    iteration=epoch + 1,
                    image=grid.permute(1, 2, 0).numpy(),
                )

        # Чекпоинт
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save({
                "epoch": epoch + 1,
                "netG": netG.state_dict(),
                "netD": netD.state_dict(),
                "optimizerG": optimizerG.state_dict(),
                "optimizerD": optimizerD.state_dict(),
                "args": vars(args),
            }, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")
            # Лучший (последний)
            torch.save(netG.state_dict(), checkpoint_dir / "generator_best.pt")
            print(f"Чекпоинт сохранён (epoch {epoch+1})")

    # Финальный график (если нет ClearML, сохраним локально)
    if task is None and len(G_losses) > 0:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.title("Generator and Discriminator Loss")
            plt.plot(G_losses, label="G")
            plt.plot(D_losses, label="D")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(samples_dir / "loss_curve.png")
            print(f"График потерь сохранён: {samples_dir / 'loss_curve.png'}")
        except Exception:
            pass

    print("Обучение завершено!")
    if task is not None:
        task.close()


def main():
    parser = argparse.ArgumentParser(description="Обучение DCGAN на пейзажах")
    parser.add_argument("--data-root", type=str, default="data", help="Папка с изображениями")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--nz", type=int, default=100, help="Размер латентного вектора")
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--samples-dir", type=str, default="samples")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--use-clearml", action="store_true", help="Включить логирование в ClearML")
    parser.add_argument("--clearml-project", type=str, default="Landscape-DCGAN")
    parser.add_argument("--clearml-task", type=str, default="Train")
    args = parser.parse_args()

    # Фиксируем сид
    torch.manual_seed(42)
    np.random.seed(42)

    train(args)


if __name__ == "__main__":
    main()
