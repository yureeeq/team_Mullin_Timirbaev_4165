"""
Простой веб-интерфейс на Streamlit для генерации пейзажей.
Запуск: streamlit run frontend/app.py
"""

import streamlit as st
import torch
from torchvision.utils import make_grid
from PIL import Image
import io
from pathlib import Path
import sys

# Добавляем корень проекта в путь
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models import Generator


@st.cache_resource
def load_model(checkpoint_path: str, nz: int = 100, ngf: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator(nz=nz, ngf=ngf, nc=3).to(device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "netG" in ckpt:
            netG.load_state_dict(ckpt["netG"])
        else:
            netG.load_state_dict(ckpt)
        netG.eval()
        return netG, device
    except Exception as e:
        st.error(f"Не удалось загрузить модель: {e}")
        return None, device


def tensor_to_image(tensor):
    """Конвертирует тензор [-1,1] в PIL Image."""
    img = (tensor.clamp(-1, 1) + 1) / 2  # [0,1]
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype("uint8")
    return Image.fromarray(img)


def main():
    st.set_page_config(page_title="Landscape DCGAN", page_icon="🏔️")
    st.title("🏔️ Генерация пейзажей с помощью DCGAN")
    st.markdown("Простая генеративная модель (DCGAN) для создания пейзажей (горы, небо, море и т.д.)")

    checkpoint = st.sidebar.text_input("Путь к чекпоинту", "checkpoints/generator_best.pt")
    num_images = st.sidebar.slider("Количество изображений", 1, 64, 16)
    seed = st.sidebar.number_input("Seed (для воспроизводимости)", value=42, step=1)

    model, device = load_model(checkpoint)
    if model is None:
        st.warning("Положите обученный чекпоинт `generator_best.pt` в папку `checkpoints/`")
        return

    if st.button("Сгенерировать пейзажи"):
        torch.manual_seed(seed)
        with torch.no_grad():
            noise = torch.randn(num_images, 100, 1, 1, device=device)
            fake = model(noise)

        # Сетка
        grid = make_grid(fake, nrow=int(num_images ** 0.5), normalize=True, padding=2)
        grid_img = tensor_to_image(grid)
        st.image(grid_img, caption="Сгенерированные пейзажи", use_container_width=True)

        # Отдельные изображения
        st.subheader("Отдельные изображения")
        cols = st.columns(4)
        for i, img_tensor in enumerate(fake):
            with cols[i % 4]:
                pil_img = tensor_to_image(img_tensor)
                st.image(pil_img, use_container_width=True)
                # Кнопка скачивания
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                st.download_button(
                    label="Скачать",
                    data=buf.getvalue(),
                    file_name=f"landscape_{i:03d}.png",
                    mime="image/png",
                    key=f"dl_{i}",
                )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Чтобы обучить модель, выполните:\n\n"
        "`python train.py --use-clearml --epochs 50`\n\n"
        "Датасет скачается автоматически."
    )


if __name__ == "__main__":
    main()
