# EEG preprocessing pipeline

Конфигурируемая предобработка Nihon/MNE: разметка DEPD → CSV и индекс скользящих окон. Выход — папки с тегом варианта обработки (без отдельной БД).

## Быстрый старт

```bash
cd pipeline
pip install -r requirements.txt
# при необходимости поправьте eeg_path, output_root, recordings
python run_preproc.py --config configs/preproc.yaml
```

Редактируйте **`configs/preproc.yaml`**: `output_root`, список **`datasets`**, общий **`preprocessing`**.

### Несколько датасетов (`datasets`)

Секция **`datasets`** — список блоков:

- **`id`** — имя датасета (попадает в `{dataset}` в `path_template` и в артефакты).
- **`root`** — корень файлов этого датасета; путь относительно каталога конфига (`configs/`), если не абсолютный.
- **`recordings`** — записи без поля `dataset` (оно подставляется из `id`). У каждой записи: `eeg_path` **относительно `root`**, `subject_id`, `session_date`, опционально `recording_id` (пустой → автоматический UUID), опционально **`tags`** — произвольная YAML-разметка (объект и/или список), целиком копируется в поле **`tags`** выходного `manifest.yaml` для каждого шага предобработки.

Один и тот же список **`preprocessing`** применяется ко **всем** записям во **всех** датасетах.

Устаревший (но поддерживаемый) вариант: верхнеуровневый **`recordings`**, у каждой строки своё поле **`dataset`**, `eeg_path` относительно каталога конфига. Нельзя смешивать **`datasets`** и **`recordings`** в одном файле.

`output_root` задаётся относительно каталога конфига, если не абсолютный.

### `path_template`

Допустимые плейсхолдеры: **`dataset`**, **`subject_id`**, **`session_date`**, **`recording_id`**, **`preproc_id`**. Другие имена в шаблоне приведут к ошибке при запуске.

## Что на выходе

Типичный путь: `output_root / {dataset} / {subject_id} / {session_date} / {preproc_id} /` (точный вид задаётся вашим `path_template`).

| `kind` | Файлы | Смысл |
|--------|--------|--------|
| `depd_intervals_csv` | `depd_intervals.csv`, `manifest.yaml` | Интервалы из аннотаций: канал, start/end (сек). |
| `sliding_windows_index` | `windows_index.csv`, `manifest.yaml` | Окна фиксированной длины и шага + метки `depd_ch_*` по каналам. |

Новый вариант (например, окна 1.73 с) — новый блок в `preprocessing` с другим `id` и параметрами; появится **отдельная** подпапка.

## Новый режим обработки

1. Файл в `preproc/postprocess/your_mode.py`: функция с сигнатурой  
   `(ctx: dict, params: dict, out_dir: Path) -> list[Path]`  
   и декоратор `@register_postprocessor("your_kind")`.
2. Импорт модуля в `preproc/postprocess/__init__.py`.
3. Запись в YAML: `kind: your_kind`, `params: { ... }`.

`ctx` содержит в том числе: `recording_id`, `dataset`, `subject_id`, `session_date`, `source_eeg_path`, `sfreq`, `n_times`, `depd_intervals`, `depd_annotations`.

## Зависимости

См. `requirements.txt` (MNE, NumPy, Pandas, PyYAML).
