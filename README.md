# 🎵 MusicGen AI REST API

> Сервис генерации музыки на базе Meta MusicGen с LoRA файнтюнингом и интеграцией GPU-фермы RunPod.  
> Деплой на Replit · Авто-масштабирование GPU через RunPod · Встроенный Swagger UI

---

## 🏗 Архитектура

```
                    ┌─────────────────────────────┐
                    │      КЛИЕНТ (curl / app)     │
                    └──────────────┬──────────────┘
                                   │ POST /generate
                    ┌──────────────▼──────────────┐
                    │     Replit FastAPI Сервер    │
                    │   (REST API + Очередь задач) │
                    └──────┬───────────────┬───────┘
                           │               │
              duration≤15s │               │ duration>15s
              (или backend=local)    (или backend=runpod)
                           │               │
             ┌─────────────▼──┐   ┌────────▼──────────────────┐
             │  Локальный      │   │   RunPod Serverless GPU    │
             │  MusicGen       │   │   (большая модель + LoRA)  │
             │  CPU / free GPU │   │   RTX 4090 / A100          │
             └────────┬────────┘   └────────┬──────────────────┘
                      │                     │
                      └──────────┬──────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Файл .wav / .mp3      │
                    │   возвращается по URL   │
                    └─────────────────────────┘
```

### Компоненты проекта

| Компонент | Технология | Назначение |
|-----------|-----------|------------|
| `app/main.py` | FastAPI | REST API, роутинг, Swagger-документация |
| `app/models.py` | Pydantic | Схемы запросов и ответов |
| `app/job_store.py` | In-memory dict | Отслеживание состояния задач |
| `app/local_generator.py` | audiocraft/MusicGen | Локальная генерация на CPU/GPU |
| `app/runpod_client.py` | httpx + RunPod API | Управление GPU-фермой |
| `runpod_worker/handler.py` | runpod SDK | Выполняется НА GPU-инстансе |
| `finetune/finetune_lora.py` | PyTorch + LoRA | Скрипт файнтюнинга |

---

## 🚀 Быстрый старт (Replit)

### 1. Импорт и настройка

```
1. Импортируй репозиторий в Replit
2. Открой вкладку "Secrets" и задай переменные окружения (см. .env.example)
3. Нажми Run
```

### 2. Обязательные секреты (вкладка Secrets в Replit)

```
BASE_URL            = https://your-repl-name.your-username.repl.co
MUSICGEN_MODEL_SIZE = small         # "small" умещается в RAM бесплатного тарифа
RUNPOD_API_KEY      = rpa_xxxxx     # с сайта runpod.io (опционально)
RUNPOD_ENDPOINT_ID  = xxxxx         # serverless endpoint на runpod.io (опционально)
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

> ⚠️ audiocraft требует Python 3.9+ и ~2 ГБ диска. На бесплатном тарифе Replit
> API работает в режиме заглушки (возвращает тихий WAV) до загрузки audiocraft.

### 4. Запуск сервера

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Открыть Swagger UI: `https://your-repl.repl.co/docs`

---

## 📡 Справочник REST API

### `POST /generate` — Создать задачу генерации

```bash
curl -X POST https://your-api.repl.co/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "энергичная электронная танцевальная музыка с мощным басом, 128 BPM",
    "duration": 15,
    "temperature": 1.0,
    "cfg_coef": 3.5,
    "backend": "auto",
    "output_format": "wav"
  }'
```

**Ответ `202 Accepted`:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Job queued. Poll /jobs/{job_id} for status.",
  "estimated_seconds": 45
}
```

---

### `GET /jobs/{job_id}` — Проверить статус задачи

```bash
curl https://your-api.repl.co/jobs/550e8400-e29b-41d4-a716-446655440000
```

**Ответ после завершения:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "prompt": "энергичная электронная танцевальная музыка...",
  "duration": 15,
  "audio_url": "https://your-api.repl.co/outputs/550e8400.wav",
  "backend_used": "local",
  "created_at": "2024-01-01T12:00:00",
  "completed_at": "2024-01-01T12:01:30",
  "duration_seconds": 90.4
}
```

**Возможные значения `status`:** `queued` → `processing` → `completed` | `failed`

---

### `GET /jobs` — Список всех задач

```bash
curl "https://your-api.repl.co/jobs?limit=10&status=completed"
```

---

### `GET /audio/{filename}` — Скачать аудиофайл

```bash
curl -O https://your-api.repl.co/audio/550e8400.wav
```

---

### `GET /health` — Проверка состояния сервиса

```json
{
  "status": "ok",
  "version": "1.0.0",
  "local_model_loaded": true,
  "runpod_configured": false,
  "active_jobs": 2,
  "timestamp": "2024-01-01T12:00:00"
}
```

---

### Эндпоинты GPU-фермы

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/runpod/status` | Статус GPU-инстанса |
| `POST` | `/runpod/start` | Вручную запустить GPU-под |
| `POST` | `/runpod/stop` | Остановить GPU-под (экономия средств) |
| `GET` | `/models` | Доступные размеры моделей и характеристики |

---

## 🎛 Параметры генерации

| Параметр | По умолчанию | Диапазон | Описание |
|----------|-------------|----------|----------|
| `prompt` | обязательный | 3–512 символов | Текстовое описание музыки |
| `duration` | 10 | 1–30 сек | Длина трека |
| `temperature` | 1.0 | 0.1–2.0 | Креативность: выше = разнообразнее |
| `top_k` | 250 | 0–2048 | Разнообразие словаря |
| `cfg_coef` | 3.0 | 1.0–10.0 | Следование промпту: выше = точнее |
| `backend` | auto | local/runpod/auto | Бэкенд генерации |
| `output_format` | wav | wav/mp3 | Формат аудио |
| `seed` | null | int | Воспроизводимая генерация |

### Советы по составлению промптов

```
✅ Хорошие промпты:
  "кинематографический оркестр, эпические струнные и медь, драматично, 90 BPM, саундтрек"
  "lo-fi хип-хоп с джазовыми аккордами, виниловая текстура, расслабленно, 75 BPM"
  "тёмное техно, индустриальные удары, искажённые синтезаторы, берлинский клуб"

❌ Слишком расплывчато:
  "хорошая музыка"
  "что-нибудь крутое"
```

---

## 🧠 LoRA Файнтюнинг

### Зачем LoRA?

| Метод | Обучаемых параметров | VRAM | Качество |
|-------|---------------------|------|----------|
| Полный файнтюнинг | 100% (3.3B) | 40 ГБ+ | Максимальное |
| **LoRA (rank=8)** | **~0.1% (3.3M)** | **8 ГБ** | **~90% от полного** |
| Prompt tuning | 0% | 2 ГБ | Ограниченное |

### Подготовка датасета

```
my_dataset/
├── track_001.wav    ← аудиоклип 5–30 секунд, моно или стерео
├── track_001.txt    ← "джазовое фортепианное трио, 120 BPM, свинг"
├── track_002.wav
├── track_002.txt    ← "тёмный эмбиент, развивающиеся текстуры, 60 BPM"
...
```

Минимальный рекомендуемый размер: **200+ пар** для ощутимого результата.

### Запуск файнтюнинга

```bash
# На машине с GPU (8 ГБ+ VRAM для модели medium)
python finetune/finetune_lora.py \
  --model_size medium \
  --data_dir ./my_dataset \
  --output_dir ./lora_weights \
  --epochs 20 \
  --lora_rank 8 \
  --lr 3e-4
```

### Использование дообученных весов

```bash
# Задать переменную окружения перед запуском API
export LORA_WEIGHTS_PATH=./lora_weights/lora_best.pt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

---

## ⚡ Настройка GPU-фермы RunPod

### Шаг 1: Создать аккаунт RunPod
1. Зарегистрироваться на [runpod.io](https://runpod.io)
2. Получить API-ключ: `Settings → API Keys`

### Шаг 2: Собрать и загрузить Docker-образ воркера

```bash
cd runpod_worker

# Заменить на свой логин Docker Hub
docker build -t youruser/musicgen-worker:latest .
docker push youruser/musicgen-worker:latest
```

### Шаг 3: Создать Serverless Endpoint
1. Перейти в `Serverless → New Endpoint`
2. Container image: `youruser/musicgen-worker:latest`
3. GPU: RTX 4090 или A100
4. Скопировать **Endpoint ID**

### Шаг 4: Настроить API

```
RUNPOD_API_KEY=rpa_xxxxx
RUNPOD_ENDPOINT_ID=xxxxx
```

### Оптимизация расходов

- Воркер **автоматически засыпает** через 60 сек простоя (настраивается в панели RunPod)
- RTX 4090: ~$0.35/час · A100: ~$0.80/час
- Типичная стоимость генерации: **$0.001–0.005 за трек**
- Используй `backend=local` для коротких треков (≤15 сек) для экономии

---

## 📊 Сравнение качества моделей

| Модель | Качество vs Suno | Применение | VRAM |
|--------|----------------|------------|------|
| MusicGen small (базовая) | ~50% | Прототипирование | 2 ГБ |
| MusicGen medium (базовая) | ~70% | Общий API | 6 ГБ |
| MusicGen large (базовая) | ~80% | Продакшн | 12 ГБ |
| MusicGen large + LoRA | ~85–90% | Доменный файнтюнинг | 12 ГБ |

> Suno AI использует проприетарные модели (предположительно 10B+ параметров) с огромным объёмом обучающих данных.
> Достичь качества 85–90% реально с large + LoRA на доменно-специфических датасетах.

---

## 🔧 Локальная разработка

```bash
# Клонировать
git clone https://github.com/yourrepo/musicgen-api
cd musicgen-api

# Установить зависимости
pip install -r requirements.txt

# Скопировать конфиг
cp .env.example .env
# Отредактировать .env под свои значения

# Запустить (режим заглушки работает без GPU)
uvicorn app.main:app --reload --port 8000

# Открыть документацию
open http://localhost:8000/docs
```

---

## 📁 Структура проекта

```
musicgen-api/
├── app/
│   ├── main.py              # FastAPI приложение, все эндпоинты
│   ├── models.py            # Pydantic-схемы
│   ├── job_store.py         # Состояние задач в памяти
│   ├── local_generator.py   # MusicGen + LoRA инференс
│   └── runpod_client.py     # Интеграция с RunPod API
├── runpod_worker/
│   ├── handler.py           # Выполняется НА GPU-инстансе
│   ├── Dockerfile           # Описание контейнера
│   └── requirements_worker.txt
├── finetune/
│   └── finetune_lora.py     # Скрипт LoRA обучения
├── outputs/                 # Сгенерированные аудиофайлы
├── requirements.txt
├── .replit                  # Конфиг Replit
└── .env.example
```

---

## 🐛 Решение проблем

**"Model not loaded" / режим заглушки**
→ audiocraft не установлен или не импортируется. Проверь логи `pip install audiocraft`.

**Нехватка памяти на Replit**
→ Установи `MUSICGEN_MODEL_SIZE=small` и `PRELOAD_MODEL=false`

**RunPod "endpoint not found"**
→ Убедись, что `RUNPOD_ENDPOINT_ID` соответствует serverless endpoint (не pod ID)

**Долгая генерация локально**
→ Ожидаемо на CPU: ~3–5 минут для 15-секундного трека. Используй RunPod для быстрой генерации.

---

## 📜 Лицензия

MIT — создано на основе [audiocraft](https://github.com/facebookresearch/audiocraft) от Meta (лицензия MIT).
