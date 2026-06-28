# Интеграция PySide UI и PyVista 3D-сцены в существующую архитектуру

## Контекст

`oil_gestures` — pipeline бесконтактного жестового управления тренажёром нефтегазового объекта: `камера → MediaPipe → распознавание жестов → курсор/команды`. Коллеги разрабатывают отдельное приложение на **PySide (UI)** и **PyVista (3D-сцена)**: 3D-сцену объекта, интерактивные элементы, уровнемер, визуализацию состояния (задвижки, насосы, давление).

**Важное уточнение по факту изучения кода** (детали — в разделе «Что уже есть»): в репозитории уже существует **готовый архитектурный контракт именно для такой интеграции** — автономная ML-граница `oil_gestures/integration/*` (NDJSON/TCP) и документ [`docs/interaction_spec.md`](docs/interaction_spec.md), который явно описывает разделение ML / 3D / UI как **независимых процессов**. Пакеты `oil_gestures/ui/`, `oil_gestures/simulator/` и часть `oil_gestures/commands/` — это **пустые файлы-заглушки (0 байт)**, а не частично готовый, тесно связанный с CV-кодом UI-слой. Поэтому задача — не «подключить два готовых модуля», а **спроектировать и согласовать**, как код коллег ляжет в уже задуманную, но не реализованную автономную архитектуру, без нарушения границы ML ⇄ UI/3D.

## Цель задачи

Определить и зафиксировать архитектуру, по которой PySide UI + PyVista 3D-сцена коллег:

1. подключаются к ML-пайплайну **только** через существующий versioned-контракт (`oil_gestures/integration/*`, `contracts/ml_events.v1.schema.json`), а не через прямые импорты vision/MediaPipe/cursor/gestures;
2. реализуют собственную логику «жест → действие в сцене» и собственное состояние сцены (включая уровнемер) согласно [`docs/interaction_spec.md`](docs/interaction_spec.md);
3. используют (или осознанно заменяют) существующие пустые пакеты `oil_gestures/ui/`, `oil_gestures/simulator/`, `oil_gestures/commands/`;
4. не ломают текущий CV/gesture/cursor pipeline и существующие тесты.

## Что уже есть в проекте (факт, проверено по коду)

### Реально реализовано и используется в `app/main.py`

| Слой | Файлы | Состояние |
| --- | --- | --- |
| Vision | `oil_gestures/vision/camera.py`, `frame_processor.py`, `mediapipe_landmarker.py`, `mediapipe_gesture.py`, `drawing.py`, `landmark_utils.py` | Реализовано, используется |
| Static-жесты | `oil_gestures/gestures/static/static_recognizer.py` | Реализовано. Маппит **canned**-жесты MediaPipe (`Closed_Fist`, `Open_Palm`, `Thumb_Up`, `Victory`) на `GestureName`. `OK_SIGN` **не** является canned-жестом MediaPipe и нигде не реализован (ни как static rule, ни как dynamic) |
| Dynamic-жесты | `oil_gestures/gestures/dynamic/dynamic_recognizer.py`, `dynamic_model.py` (Protocol) | Это **фасад без модели**: `DynamicGestureRecognizer(model=None)` — в `app/main.py` модель не подключена, поэтому `dynamic_gesture` сегодня **всегда `None`** в живом демо. Обученного чекпойнта нет: `assets/models/pytorch/` содержит только `.gitkeep` |
| Тренировочный пайплайн | `dynamic_gestures/` (top-level, отдельно от `oil_gestures/`) — `runtime/`, `scripts/collect_dynamic_dataset.py`, `process_dynamic_dataset.py`, `train_dynamic_model.py`, `train_stgcn_model.py` | Это **сбор датасета и обучение**, не runtime-инференс в демо |
| Cursor-жесты | `oil_gestures/gestures/cursor/cursor_recognizer.py` | Реализовано: `INDEX_MCP`, `INDEX_SQUEEZE`, `INDEX_RELEASE`, `MIDDLE_PINCH` — **отдельный** словарь, не путать с «дикторскими» `SQUEEZE`/`RELEASE` из `GestureName` (те относятся к dynamic-каналу и пока ничего не производят) |
| Cursor pipeline | `oil_gestures/cursor/cursor_pipeline.py`, `action_mapper.py`, `cursor_smoothing.py`, `hand_pointer.py`, `mouse_controller.py`, `screen_mapper.py`, `backends/*` | Реализовано полностью. Переводит cursor-жесты в **реальные OS-события мыши** (или dry-run) |
| Toggle/cooldown | `oil_gestures/gestures/decision/gesture_toggle.py`, `cooldown.py` | Реализовано, используется (`VICTORY` включает/выключает курсорный режим) |
| **ML-контракт (автономная граница)** | `oil_gestures/integration/contracts.py`, `publisher.py`, `ndjson_server.py`, `client.py`, `contracts/ml_events.v1.schema.json` | **Реализовано и протестировано** (`tests/test_integration_contracts.py`). Это и есть существующая точка интеграции с UI/3D — см. ниже |
| Конфиг | `app/app_config.py` + `configs/default.yaml`, `gestures.yaml`, `model_config.yaml` | Реализовано, но только для камеры/MediaPipe/cursor. **Не знает** про UI/3D/сцену |

### Полностью пустые файлы-заглушки (0 байт, нигде не импортируются)

- `oil_gestures/ui/main_window.py`, `camera_widget.py`, `status_panel.py`, `debug_panel.py`, `ui_controller.py`
- `oil_gestures/simulator/simulator_controller.py`, `simulator_state.py`, `mock_simulator.py`, `blender_bridge.py`
- `oil_gestures/commands/command_dispatcher.py`, `command_history.py`

В проекте **нет PySide, PyVista или Qt** — ни в коде, ни в `requirements.txt`. Единственный UI сегодня — окно `cv2.imshow` в `app/main.py` (локальный preview ML-процесса, не имеет отношения к будущему PySide-приложению).

### Частично готово, но не подключено

- `oil_gestures/commands/command_mapper.py` — **реализован и покрыт тестом** (`tests/test_command_mapper.py`), маппит `GestureResult → CommandResult` по словарю `GestureName → CommandName`. Нигде не вызывается из `app/main.py`.
- `configs/commands.yaml` (`gesture_command_mode.mapping: FIST→GRAB_OBJECT, OPEN_PALM→RELEASE_OBJECT`) — **не читается** `app_config.py` (orphaned-конфиг).
- `oil_gestures/core/enums.py::InteractionMode` (`CURSOR_CONTROL`/`GESTURE_COMMAND`/`DEMO`) — определён, но **нигде не используется**.
- `configs/default.yaml::app.mode: "GESTURE_COMMAND"` — поле существует в YAML, но `app_config.py` его не парсит.
- `oil_gestures/core/types.py::SimulatorStateSnapshot` — dataclass с полями `selected_object, valve_open, valve_rotation_degrees, pump_running, emergency_stop, timestamp`. В docstring указаны producer'ы (`simulator/simulator_state.py`, `mock_simulator.py` — оба пустые) и consumer'ы (`ui/status_panel.py`, `ui/debug_panel.py` — оба пустые). **Поля для уровня (level) нет.**

### Ключевой существующий контракт интеграции

[`docs/integration_contract.md`](docs/integration_contract.md) и [`docs/interaction_spec.md`](docs/interaction_spec.md) уже фиксируют архитектурный принцип:

> «ML runtime — автономный продюсер. UI и 3D-приложения — автономные консьюмеры и **не должны импортировать** vision, MediaPipe, cursor или application runtime модули. Единственная общая зависимость — версионированный JSON-контракт».

Транспорт: TCP/NDJSON на `127.0.0.1:8765`, поднимается через `python scripts/run_demo.py --event-server [--publish-camera] [--headless]`. Есть готовый reference-консьюмер (`scripts/consume_ml_events.py`, `oil_gestures.integration.client.iter_events`) и **мок-продюсер для разработки UI/3D без камеры** (`scripts/mock_ml_events.py`) — он эмулирует весь «замороженный» словарь жестов из `interaction_spec.md`, включая ещё не реализованные ML-жесты.

`docs/interaction_spec.md` также явно закрепляет распределение ответственности:
- **ML** отдаёт только сырые жесты, не знает о сцене/режимах/командах;
- **3D-приложение** хранит состояние сцены и режим (`NAVIGATION`/`CONTROL`), реализует всю логику «жест → действие»;
- **UI** ничего не интерпретирует — только отображает телеметрию и человекочитаемые подписи.

## ⚠️ Несостыковка словаря жестов (важно для коллег)

В `oil_gestures/core/enums.py::GestureName` (код, которым говорит реальный ML-пайплайн) и в `docs/interaction_spec.md`/`scripts/mock_ml_events.py` (зафиксированный кросс-командный словарь) используются **разные имена для семантически одних и тех же жестов**:

| Смысл | Имя в `core/enums.py` (код) | Имя в `interaction_spec.md` / mock-продюсере |
| --- | --- | --- |
| Давление + / − | `ROTATE_CLOCKWISE` / `ROTATE_COUNTERCLOCKWISE` | `WRIST_ROTATE_CW` / `WRIST_ROTATE_CCW` |
| Указательный/выбор | `POINTING_INDEX` | `POINT` |
| Сжатие/разжатие (dynamic) | `SQUEEZE` / `RELEASE` | `CLENCH` / `SPREAD` |
| ОК-жест | — (не объявлен в `GestureName`) | `OK_SIGN` |

Семантика совпадает (например, и там и там «поворот кисти» = «давление», **не** «вращение вентиля» — это подтверждает корректность интерпретации задачи), но **строковые имена расходятся**. Это нужно явно зафиксировать как риск (см. ниже), а не предполагать, какое имя «правильное».

## Предлагаемая архитектура интеграции

Рекомендуемый подход — **не встраивать PySide/PyVista внутрь ML-процесса**, а закрепить его как второй, полностью автономный процесс-консьюмер, в соответствии с уже существующим `docs/integration_contract.md`. Конкретно:

1. **ML-процесс** (`app/main.py` / `scripts/run_demo.py --event-server`) остаётся без изменений — он уже умеет работать headless и публиковать NDJSON-контракт. Изменения в этой задаче сюда **не входят**.
2. **UI/3D-процесс** (новый, отдельный entry point) — то самое приложение коллег. Размещается внутри уже существующих, но пока пустых пакетов `oil_gestures/ui/` и `oil_gestures/simulator/` (естественное место по структуре проекта), и:
   - подключается к ML-контракту через `oil_gestures.integration.client.iter_events(...)` (единственная разрешённая зависимость от ML-стороны, согласно `integration_contract.md`) либо через собственную минимальную NDJSON/TCP-реализацию — это открытый вопрос (см. «Риски»);
   - реализует «жест → действие» и хранение состояния сцены **сам**, по таблице из `docs/interaction_spec.md`;
   - не импортирует `oil_gestures.vision`, `oil_gestures.gestures`, `oil_gestures.cursor`, `oil_gestures.mediapipe_*`.
3. Курсорный режим (`--cursor-on --real-mouse`) **уже сегодня** двигает реальный OS-курсор и кликает — если окно PySide/PyVista активно на десктопе, оно и сейчас нативно получает эти клики/перемещения от Qt, без какого-либо нового кода. Это нужно явно документировать как уже работающий, отдельный от NDJSON-контракта путь взаимодействия (`SQUEEZE`/`RELEASE`/`OK_SIGN` в этом пути отображаются на `INDEX_SQUEEZE`/`INDEX_RELEASE`/клик через cursor-канал, а не через dynamic-канал).

```text
                         Процесс 1 (существует, без изменений)
Camera → vision/* → static/dynamic/cursor recognizers → cursor/cursor_pipeline.py
                                  │                              │
                                  │ (OS-мышь, если --real-mouse) │
                                  ▼                              │
                    oil_gestures/integration/publisher.py ───────┘
                                  │ NDJSON/TCP (127.0.0.1:8765)
                                  ▼
                         Процесс 2 (новый, эта задача)
                    oil_gestures/integration/client.py (читает контракт)
                                  │
                                  ▼
                    simulator/simulator_controller.py  (жест → действие, по interaction_spec.md)
                                  │
                                  ▼
                    simulator/simulator_state.py  (состояние сцены + уровнемер)
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
        ui/ui_controller.py + PySide-панели   simulator/pyvista_scene.py (PyVista)
```

## Data flow

Фактическая схема (адаптирована под код, а не из примера в задаче):

```text
[Процесс 1 — ML, без изменений]
Camera frame
  -> vision/camera.py + vision/frame_processor.py
  -> vision/mediapipe_gesture.py (landmarks + canned static gesture)
  -> gestures/static/static_recognizer.py        \
  -> gestures/dynamic/dynamic_recognizer.py        }-> независимые GestureResult
  -> gestures/cursor/cursor_recognizer.py         /
  -> cursor/cursor_pipeline.py -> OS-мышь (если включён курсорный режим)
  -> integration/publisher.py -> integration/ndjson_server.py
                                       │
                                       │ NDJSON/TCP, contract v1
                                       ▼
[Процесс 2 — UI/3D, предмет этой задачи]
integration/client.py (iter_events)
  -> simulator/simulator_controller.py   (интерпретация жеста по interaction_spec.md,
                                           режимы NAVIGATION/CONTROL)
  -> simulator/simulator_state.py        (состояние сцены: выбранный объект, задвижка,
                                           насос, давление, уровень)
  -> ui/ui_controller.py
       ├─> ui/main_window.py + camera_widget.py + status_panel.py + debug_panel.py (PySide)
       └─> simulator/pyvista_scene.py + simulator/level_gauge.py (PyVista, embed в PySide)
```

## Файлы, которые нужно использовать/доработать

| Файл | Текущее состояние | Что нужно сделать |
| --- | --- | --- |
| `oil_gestures/ui/main_window.py` | Пустой (0 байт) | Реализовать `QMainWindow`, который компонует camera-виджет, панели и встроенную PyVista-сцену. Никакой gesture-логики |
| `oil_gestures/ui/camera_widget.py` | Пустой | Виджет, рисующий JPEG-кадры из `oil_gestures.ml.camera_frame` контракта (не из камеры напрямую) |
| `oil_gestures/ui/status_panel.py` | Пустой | Отображение текущего жеста/FPS/режима курсора из `MLRuntimeEvent`. Без интерпретации |
| `oil_gestures/ui/debug_panel.py` | Пустой | Сырые события контракта, история команд (если используется `command_history.py`) |
| `oil_gestures/ui/ui_controller.py` | Пустой | Связывает `simulator_state.py` с панелями PySide, без бизнес-логики |
| `oil_gestures/simulator/simulator_controller.py` | Пустой | Реализация таблицы «жест → действие» из `docs/interaction_spec.md`, владение режимом `NAVIGATION`/`CONTROL` |
| `oil_gestures/simulator/simulator_state.py` | Пустой | Конкретная реализация состояния сцены. Решить: расширять ли существующий `SimulatorStateSnapshot` из `core/types.py` или вести собственный state-класс в этом пакете (см. «Уровнемер») |
| `oil_gestures/simulator/mock_simulator.py` | Пустой | Генератор фейкового состояния сцены для разработки UI без реального симулятора, по аналогии с `scripts/mock_ml_events.py` |
| `oil_gestures/simulator/blender_bridge.py` | Пустой, имя противоречит выбору PyVista | Решить отдельно: переименовать/перепрофилировать под импорт ассетов (например, .glb/.obj, экспортированных из Blender, для использования в PyVista) либо удалить — **не входит в объём этой задачи**, но нужно зафиксировать решение |
| `oil_gestures/commands/command_mapper.py` | Реализован, не используется | Можно реализовать на нём связку `GestureContract.name → CommandName` внутри `simulator_controller.py`, если решено использовать `CommandName`-словарь вместо прямой строковой таблицы из `interaction_spec.md` |
| `oil_gestures/commands/command_dispatcher.py` | Пустой | При необходимости — диспетчер, вызывающий методы `simulator_controller.py` по `CommandResult` |
| `oil_gestures/commands/command_history.py` | Пустой | Хранение истории выполненных команд для `debug_panel.py` |
| `oil_gestures/core/types.py` | Реализован | **Не менять в этой задаче.** Решить вопрос расширения `SimulatorStateSnapshot` полем уровня — отдельным пунктом (см. «Уровнемер») |
| `oil_gestures/core/enums.py` | Реализован | Зафиксировать (без изменений сейчас), что `InteractionMode` и расхождение словаря жестов — открытые вопросы, требующие решения до или во время реализации |
| `app/app_config.py` / `app/main.py` | Реализован, продюсер ML | **Не изменять** — остаётся ML-продюсером. Новому UI/3D-процессу нужен **свой** entry point и **своя** конфигурация |
| `configs/commands.yaml` | Существует, не читается | Решить: либо начать читать его из нового UI-конфига, либо перенести маппинг в `interaction_spec.md`-driven код и убрать как orphaned |
| `requirements.txt` | Не содержит PySide/PyVista | Добавить зависимости UI-приложения отдельным requirements-файлом (например `requirements-ui.txt`), чтобы не тянуть тяжёлые GUI/3D-зависимости в headless ML-окружение |

## Новые файлы, которые возможно нужно создать

Не создаются автоматически — только предложение с обязанностями:

- `oil_gestures/simulator/pyvista_scene.py` — построение/обновление 3D-сцены нефтегазового объекта (меши скважины, задвижки, насоса, резервуара) средствами PyVista; не знает о MediaPipe/жестах, принимает только обновления состояния от `simulator_state.py`.
- `oil_gestures/simulator/scene_objects.py` — обёртки над PyVista-акторами для выбираемых/подсвечиваемых объектов сцены (хранение `actor`, `object_id`, состояние подсветки), используется `pyvista_scene.py` и `simulator_controller.py` (выбор объекта).
- `oil_gestures/simulator/level_gauge.py` — см. отдельный раздел ниже.
- `oil_gestures/ui/pyvista_widget.py` — `QtInteractor`/`QtPanel`-обёртка PyVista, встраиваемая как `QWidget` в `main_window.py`.
- `oil_gestures/ui/control_panel.py` — опциональные элементы управления (если нужно ручное тестирование сцены без жестов: кнопки "открыть задвижку", слайдер давления и т.п. для отладки).
- `oil_gestures/ui/scene_panel.py` — обёртка-контейнер, объединяющая `pyvista_widget.py` и подписи режима/выбранного объекта из `status_panel.py`.
- Новый entry point для UI-процесса (например `app/ui_main.py` или `scripts/run_ui.py`) и соответствующий `ui_app_config.py` (камера/MediaPipe-секции там не нужны — нужны host/port ML-контракта, путь к 3D-ассетам, конфигурация сцены).

## Модули UI

- PySide **не содержит** распознавание жестов и не парсит landmarks/raw-контракт глубже, чем чтение полей `gestures.*`, `hand.pointer`, `cursor.*`, `performance.*` из `MLRuntimeEvent`.
- `camera_widget.py` рисует уже готовый JPEG из `CameraFrameEvent` (нужно запускать продюсера с `--publish-camera`); если кадры не нужны — можно работать без них (UI получает только метаданные жестов, без видео).
- `status_panel.py`/`debug_panel.py` читают человекочитаемые подписи из `docs/interaction_spec.md` (или его машиночитаемой производной — см. риски), а не вычисляют их сами.
- `ui_controller.py` — единственная точка, которая знает и про PySide-виджеты, и про `simulator_state.py`; сам по себе не содержит правил «жест → действие».

## Модули 3D-сцены / simulator layer

- `simulator_controller.py` — единственное место, где реализована таблица `docs/interaction_spec.md` (режимы `NAVIGATION`/`CONTROL`, переходы, действия над объектами). Не зависит от PySide.
- `pyvista_scene.py`/`scene_objects.py`/`level_gauge.py` зависят только от `simulator_state.py`, не от событий контракта напрямую — так PyVista-слой остаётся тестируемым без сети и без ML.
- `mock_simulator.py` позволяет 3D-разработчикам тестировать сцену без `simulator_controller.py` и без живого ML-потока — генерирует фейковые `SimulatorStateSnapshot`/уровни.

Неправильные зависимости, которые нужно явно запретить (зафиксировать code review правилом / тестом на импорты):

```text
✗ PyVista-сцена импортирует cv2/mediapipe/oil_gestures.vision
✗ PyVista-сцена напрямую открывает камеру или сокет
✗ UI парсит landmarks или вычисляет жесты
✗ 3D-сцена содержит правила распознавания (geometry/threshold-логику)
```

## Интеграция уровнемера

Уровнемера в проекте сегодня **нет вообще** — ни в `core/types.py`, ни в `interaction_spec.md`, ни в схеме контракта. Это полностью новая концепция, которую нужно спроектировать:

- **Что это:** визуальный индикатор заполненности резервуара/скважины (0–100%) в 3D-сцене — отдельный объект сцены, который не выбирается и не управляется напрямую жестами (если ТЗ коллег не говорит иное), а **отражает** состояние, которое меняется как следствие других действий (например, открытие/закрытие задвижки, работа насоса).
- **Какое состояние хранить:** `level_percent: float` (0.0–100.0) плюс, возможно, `target_level_percent` для анимации сглаживания и `is_critical: bool` для аварийной подсветки.
- **Как отображать в PyVista:** отдельный меш-индикатор (цилиндр/прямоугольник с динамически обновляемой заливкой по высоте, либо текстовый/цветовой оверлей) в `level_gauge.py`, с методом вида `update(level_percent: float) -> None`, вызываемым из `pyvista_scene.py` при изменении `simulator_state.py`.
- **Какие жесты/команды могут менять его значение:** по `docs/interaction_spec.md` ни один жест не управляет уровнем напрямую — уровень должен быть **производным** от состояния насоса/задвижки внутри `simulator_controller.py` (бизнес-правило тренажёра, не жестовое). Это нужно явно согласовать с коллегами, а не считать решённым.
- **Куда положить код:** `oil_gestures/simulator/level_gauge.py` (3D-представление) + поле состояния в `simulator_state.py`.
- **Как связать с `SimulatorStateSnapshot`:** `SimulatorStateSnapshot` в `core/types.py` сегодня **не имеет** поля уровня и в принципе является, похоже, остатком более раннего (внутрипроцессного) варианта архитектуры — её producer/consumer-докстринги указывают на пустые модули. Два варианта, оба не реализуются в рамках этой задачи, а выносятся на решение:
  1. Расширить `SimulatorStateSnapshot` полем `level_percent: float` (и, возможно, `pressure: float`, которого там тоже нет) — отдельной небольшой PR-задачей до начала реализации UI;
  2. Завести в `simulator/simulator_state.py` собственный, более широкий state-класс (например `SceneStateSnapshot`), не привязанный к `core/types.py`, поскольку состояние сцены — внутренняя забота консьюмер-процесса и не обязано жить в `core` (который сегодня выглядит как общий контракт ML-стороны).

  Рекомендация: вариант 2 более согласован с принципом автономности консьюмера из `integration_contract.md`, но финальное решение — за командой.

## Интеграция gesture/cursor pipeline с UI

- NDJSON-контракт уже несёт всё, что нужно UI для отображения: текущий жест (`gestures.static/dynamic/cursor`), состояние курсора (`cursor.enabled/pressed/action/screen_position`), FPS/производительность, наличие руки и нормализованную точку указателя.
- Курсорный режим (`SQUEEZE`/`RELEASE`/`OK_SIGN` в терминах задачи → фактически `INDEX_SQUEEZE`/`INDEX_RELEASE`/клик в коде) воздействует на PySide-окно как на любое другое десктоп-окно через ОС, если активирован `--real-mouse`; никакого дополнительного связывающего кода для этого пути не требуется — только документация и явный тест (фокус окна, корректность зоны экрана).
- `ROTATE_CLOCKWISE`/`ROTATE_COUNTERCLOCKWISE` (давление) сегодня **не производятся** реальным пайплайном (dynamic-модель не подключена) — UI должен быть готов получать их по контракту, но не может полагаться на них до обучения и подключения модели (вне объёма этой задачи). До этого момента — тестировать через `scripts/mock_ml_events.py`, который уже эмулирует `WRIST_ROTATE_CW/CCW`.

## Интеграция command mapping с 3D scene

- `interaction_spec.md` явно отдаёт логику «жест → команда сцены» 3D-приложению. Рекомендуется реализовать её в `simulator_controller.py` как прямую таблицу (жест/режим → действие), а не через `commands/command_mapper.py` + `CommandName`, **если** только команда не хочет единый типизированный словарь команд — тогда `command_mapper.py` можно переиспользовать как промежуточный слой (`GestureContract.name → CommandName`), а `simulator_controller.py` дальше исполняет `CommandName`. Выбор одного из двух подходов — открытый вопрос, см. ниже.
- В любом случае результат — изменение `simulator_state.py`, который затем читают и PySide-панели, и PyVista-сцена.

## Конфиги

- `configs/default.yaml`, `gestures.yaml`, `model_config.yaml` — относятся только к ML-продюсеру, не трогать.
- `configs/commands.yaml` — сейчас orphaned (не читается кодом). Нужно решить: использовать его как источник для `simulator_controller.py`/`command_mapper.py`, или считать устаревшим и убрать после переноса логики в `interaction_spec.md`-driven код.
- Для нового UI-процесса нужен отдельный конфиг (новый YAML-файл, например `configs/ui.yaml`, и/или `ui_app_config.py`) с секциями: адрес/порт ML-контракта, путь к 3D-ассетам сцены, параметры уровнемера/сцены. Не входит в объём данной задачи — только проектирование, не реализация.

## Acceptance criteria

- [ ] У UI/3D-приложения есть собственный entry point, отдельный от `app/main.py` (тот остаётся ML-продюсером и не запускает PySide).
- [ ] Приложение коллег **не импортирует** `oil_gestures.vision`, `oil_gestures.gestures`, `oil_gestures.cursor`, `mediapipe` — проверяется явным правилом/тестом на импорты.
- [ ] PySide-окно открывается и встраивает PyVista-сцену 3D-объекта (через `QtInteractor` или аналог).
- [ ] Camera feed в UI отображается из `oil_gestures.ml.camera_frame` контракта (`--publish-camera`), а не из прямого доступа к камере.
- [ ] Текущий жест/FPS/режим курсора отображаются в `status_panel`/`debug_panel` на основе `oil_gestures.ml.runtime` контракта.
- [ ] Объект сцены можно выбрать/подсветить через `simulator_controller.py` по правилам `docs/interaction_spec.md` (`POINT`/`POINTING_INDEX` → выбор объекта → переход `NAVIGATION → CONTROL`).
- [ ] Курсорный режим ML (`INDEX_SQUEEZE`/`INDEX_RELEASE`/`MIDDLE_PINCH`) корректно взаимодействует с UI-окном как с обычным десктоп-приложением при включённом `--real-mouse`.
- [ ] `ROTATE_CLOCKWISE`/`ROTATE_COUNTERCLOCKWISE` (или согласованное имя — см. риски) меняют состояние давления в `simulator_state.py`, протестировано через `scripts/mock_ml_events.py` (без необходимости в обученной dynamic-модели).
- [ ] Уровнемер отображается в 3D-сцене и обновляется из `simulator_state.py`/`level_gauge.py`.
- [ ] PySide-код не содержит правил распознавания жестов или интерпретации MediaPipe-данных.
- [ ] PyVista-сцена не содержит правил распознавания жестов и не открывает сетевые/камера-соединения напрямую (только через `simulator_state.py`).
- [ ] ML-пайплайн (`app/main.py`, `oil_gestures/vision`, `gestures`, `cursor`) остаётся неизменным и переиспользуемым как самостоятельный продукт.
- [ ] Существующие тесты (`tests/*`, включая `test_integration_contracts.py`, `test_command_mapper.py`) проходят без изменений.
- [ ] Зафиксировано решение по расхождению словаря жестов (`ROTATE_CLOCKWISE` vs `WRIST_ROTATE_CW` и т.п.) — либо правкой `interaction_spec.md`, либо явным маппингом на стороне UI.

## План реализации по шагам

1. **Согласование контракта** — зафиксировать с командой 3D/UI: имена жестов (расхождение из раздела выше), формат состояния уровнемера, выбор `SimulatorStateSnapshot` vs новый state-класс, нужен ли `command_mapper.py`/`CommandName` или прямая таблица.
2. **Каркас процесса** — новый entry point + конфиг UI-приложения, подключение к ML-контракту через `integration/client.py`, проверка на mock-продюсере (`scripts/mock_ml_events.py`) без реальной камеры.
3. **Simulator layer** — `simulator_state.py`, `simulator_controller.py`, `mock_simulator.py`: логика режимов `NAVIGATION`/`CONTROL`, выбор объекта, изменение давления/задвижки/насоса/уровня по таблице `interaction_spec.md`.
4. **PyVista-сцена** — `pyvista_scene.py`, `scene_objects.py`, `level_gauge.py`: статичная сцена объекта, без привязки к жестам, с ручным/мок-управлением состоянием для отладки.
5. **PySide-каркас** — `main_window.py`, `pyvista_widget.py`, `camera_widget.py`, `status_panel.py`, `debug_panel.py`: сборка окна, встраивание сцены, привязка к `ui_controller.py`.
6. **Связка** — `ui_controller.py` соединяет `simulator_state.py` с PySide-виджетами и PyVista-сценой; полный прогон на mock-продюсере.
7. **Прогон на реальном ML** — `scripts/run_demo.py --event-server --publish-camera` + UI-процесс, проверка acceptance criteria вживую (камера, статичные жесты, курсорный режим).
8. **Документация** — обновить `docs/interaction_spec.md` (если менялись имена/режимы) и добавить README для нового UI-процесса (запуск, зависимости, конфиг).

## Что не входит в задачу

- Обучение dynamic-модели и подключение обученного чекпойнта (`dynamic_gestures/` тренировочный пайплайн).
- Сбор/доразметка датасета жестов.
- Изменение MediaPipe pipeline, static/dynamic/cursor recognizer'ов.
- Изменение `app/main.py`, `app/app_config.py`, `oil_gestures/integration/*` и схемы `contracts/ml_events.v1.schema.json` (если расширение контракта понадобится — отдельная задача с версионированием v2).
- Финальный визуальный дизайн интерфейса и 3D-ассетов.
- Реальная промышленная интеграция симулятора (физика, реальные датчики).
- Упаковка приложения (инсталлятор, CI-сборка бинарника).
- Реализация `oil_gestures/commands/command_dispatcher.py`/`command_history.py`, если в итоге решено обойтись без `CommandName`-слоя (см. открытый вопрос).

## Риски и спорные места

1. **Расхождение словаря жестов** между `core/enums.py::GestureName` (код) и `docs/interaction_spec.md`/mock-продюсером (кросс-командный контракт). Нужно явное решение, кто под кого подстраивается, до начала реализации `simulator_controller.py`.
2. **`OK_SIGN`** заявлен в задаче и в `interaction_spec.md` как `planned`, но не существует ни как MediaPipe canned-жест, ни как правило в `static_recognizer.py`/`dynamic`. Нельзя тестировать вживую до его появления — только через мок.
3. **`SimulatorStateSnapshot`** в `core/types.py` выглядит как остаток более раннего (внутрипроцессного) дизайна — её докстринги указывают consumer'ами пустые `ui/status_panel.py`/`debug_panel.py`, что подразумевает прямой импорт `core.types` из UI-процесса. Это потенциально противоречит принципу «консьюмер не импортирует application runtime модули» из `integration_contract.md`. Нужно явно решить, разрешён ли импорт `oil_gestures.core.types`/`core.enums` из UI/3D-процесса (это не vision/MediaPipe/cursor, но формально это «application runtime»), или эти типы нужно дублировать/выносить в общий пакет без зависимостей.
4. **`blender_bridge.py`** — само название файла предполагает другой 3D-движок (Blender), чем фактически выбранный (PyVista). Требует явного решения (переименовать/удалить/перепрофилировать), иначе вводит в заблуждение.
5. **`configs/commands.yaml` и `InteractionMode`** — orphaned-конфиг и неиспользуемый enum создают иллюзию готовой инфраструктуры command-mode, которой по факту нет. Нужно либо реализовать, либо явно задокументировать как устаревшее.
6. **Выбор транспорта для уровнемера/состояния сцены** — контракт ML принципиально не должен знать про уровень/давление/задвижки (по `interaction_spec.md`, ML не знает о сцене). Если 3D-приложению понадобится publish/sync состояния сцены наружу (например, для UI как отдельного третьего процесса), нужен **отдельный** контракт, не путать с `ml_events.v1`.
7. **Производительность embedding PyVista в PySide** — `QtInteractor` тянет vtk; нужно заранее проверить совместимость версий PyVista/vtk/PySide и стоимость рендера при одновременном чтении NDJSON-потока на 30 FPS.
8. **Открытый вопрос из `interaction_spec.md`**: переход `CONTROL → NAVIGATION` (снятие выбора объекта) не определён — нужно решить вместе с командой 3D до реализации `simulator_controller.py`.

## Definition of Done

- Архитектурное решение по всем пунктам раздела «Риски» зафиксировано (в этом issue, в `interaction_spec.md` или в отдельном ADR/доке).
- UI/3D-процесс запускается отдельно от ML-продюсера, подключается к NDJSON-контракту, не импортирует ML-внутренности.
- Все acceptance criteria выполнены и проверены вручную (камера + жесты, и через `scripts/mock_ml_events.py` без камеры).
- Существующий ML pipeline и его тесты не изменены и проходят как прежде.
- Документация (`README.md` нового UI-процесса, при необходимости — обновлённый `docs/interaction_spec.md`) отражает финальное распределение ответственности между ML/3D/UI.
