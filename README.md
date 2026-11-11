# Генерация простых пейзажей с помощью DCGAN

Проект по дисциплине «Прикладные методы искусственного интеллекта».

**Выполнили студенты группы 4265 Муллин Б.И. и Тимирбаев М.Ю.**

**Тема:** Генерация простых пейзажей (горы, небо, море и т.д.)  
**Архитектура:** DCGAN (Deep Convolutional Generative Adversarial Network)  
**Тип данных:** Изображения  
**Датасет:** [Landscape Pictures](https://www.kaggle.com/datasets/arnaud58/landscape-pictures) (~4.3k изображений с Flickr)


В проекте реализовано обучение DCGAN, генерация изображений, логирование экспериментов в **ClearML** (графики потерь, сэмплы).

## Структура проекта

```text
Landscape_DCGAN/
├── models.py          # Generator и Discriminator
├── dataset.py         # Загрузка датасета + автоскачивание с Kaggle
├── train.py           # Обучение модели
├── generate.py        # Генерация из чекпоинта
├── frontend/
│   └── app.py         # Веб-интерфейс на Streamlit
├── data/              # Сюда скачивается датасет (создаётся автоматически)
├── checkpoints/       # Сохранённые модели
├── samples/           # Сгенерированные сэмплы и графики
├── requirements.txt
└── README.md
```

## Установка

```bash
# Рекомендуется использовать виртуальное окружение
python -m venv venv
source venv/bin/activate          # Linux/macOS
# или venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### Настройка ClearML

1. Зарегистрируйтесь на [clear.ml](https://clear.ml)
2. Получите credentials: `clearml-init` (или создайте `~/clearml.conf`)
3. При обучении указывайте флаг `--use-clearml`

## Загрузка датасета

Датасет скачивается автоматически при первом запуске `train.py` через библиотеку `kagglehub`.

Если хотите скачать вручную:

```python
import kagglehub
path = kagglehub.dataset_download("arnaud58/landscape-pictures")
print(path)
```

Или положите все `.jpg` изображения в папку `data/`.

## Обучение

```bash
# Базовый запуск (50 эпох, автоматически скачает датасет)
python train.py --epochs 50

# С ClearML и кастомными параметрами
python train.py \
  --epochs 100 \
  --batch-size 64 \
  --lr 0.0002 \
  --use-clearml \
  --clearml-project "Landscape-DCGAN" \
  --clearml-task "Train-64x64"
```

### Что происходит во время обучения

- Каждые `log_interval` итераций выводятся потери Generator и Discriminator
- После каждой эпохи сохраняется сетка сгенерированных изображений в `samples/epoch_XXX.png`
- Если включён ClearML — потери и изображения логируются в веб-интерфейс ClearML
- Чекпоинты сохраняются каждые `save_every` эпох + финальный `generator_best.pt`

**Совет:** Для хорошего качества нужно 50–150 эпох на GPU. На CPU будет очень медленно.

## Генерация изображений

```bash
# Сгенерировать сетку из 64 изображений
python generate.py --checkpoint checkpoints/generator_best.pt --num-samples 64 --output samples/my_grid.png

# Сохранить каждое изображение отдельно
python generate.py --save-individual
```

## ClearML — графики

После обучения с флагом `--use-clearml` в интерфейсе ClearML вы увидите:

- Графики `Loss/Discriminator` и `Loss/Generator`
- Изображения сгенерированных пейзажей по эпохам
- Гиперпараметры эксперимента
- Автоматически подключённый PyTorch


## Веб-интерфейс

```bash
streamlit run frontend/app.py
```

Откройте браузер по адресу `http://localhost:8501`.  
Можно генерировать пейзажи, менять seed и скачивать результаты.

