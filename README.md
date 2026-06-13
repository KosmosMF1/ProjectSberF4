# SberAI-F4

## Краткое описание

SberAI-F4 - это веб-прототип для обработки фото и видео с покадровой отправкой в backend.

Сейчас приложение делает следующее:

- принимает изображение или видео через web-интерфейс;
- для видео извлекает кадры и отправляет их по одному в backend;
- backend возвращает для каждого кадра:
  - сам кадр,
  - маски трассы и колёс,
  - результат проверки выезда за пределы трассы.

Текущие модели:

- SAM 3 для маски поверхности трассы;
- YOLO segmentation для масок колёс.

## Быстрый запуск

Требования:

- Python 3.11+
- pip

Команды для Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env-example .env
uvicorn app.main:app --host localhost --port 8000 --reload
```

Если в системе есть `python`, можно использовать его вместо `python3`.

Для реального inference проверьте переменные в `.env`:

```env
WHEEL_MODEL_PATH=./app/models/yolo26_seg_best.pt
TRACK_MODEL_PATH=...
```

После запуска откройте в браузере:

```text
http://127.0.0.1:8000
```

## Как проверить работу

1. Нажмите "Выбрать медиафайл" и выберите фото или видео.
2. Нажмите "Запустить обработку".
3. В интерфейсе появится кадр с областью нарушения, а справа - ответ backend.

## API (кратко)

Endpoint для web-интерфейса:

```text
POST /api/infer/violation
```

Form-data параметры:

- `frame` (файл кадра),
- `frame_index` (номер кадра, int),
- `offtrack_threshold` (опционально, float от 0 до 1),
- `hard_violation_threshold` (опционально, float от 0 до 1).

Endpoint для сырых масок:

```text
POST /api/infer/frame
```

Form-data параметры:

- `frame` (файл кадра),
- `frame_index` (номер кадра, int).


## TODO:

- violation_score
- threshhold на бэкенд и в переменные
- threshhold на каждое нарушение
- сделать описание датасета на kaggle
- примеры работы и часть презы пихнуть в ридми