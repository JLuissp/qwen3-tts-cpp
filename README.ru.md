# C++ Voice Design (Qwen3-TTS)

## Назначение
`qwen3_tts_cpp` — C++ библиотека и примеры для генерации речи через ONNX-экспорт Qwen3-TTS.

Текущий стек:
- `QWEN3TTS::Voice` — runtime (load/generate/unload)
- `QWEN3TTS::VoiceTokenizer` — токенизация `text + instruct` в ID
- `QWEN3TTSUTILS` — утилиты (WAV, tensors, sampling, parsing)

## Примечание о разработке
Проект разработан с использованием `gpt-5.3-codex` и затем вручную проверен/доработан.

## Структура
- `src/voice.h`, `src/voice.cpp`
  Основной runtime API.
- `src/tokenizer.h`, `src/tokenizer.cpp`
  Токенайзер для Qwen3-TTS prompt формата.
- `src/utils.h`, `src/utils.cpp`
  Вспомогательные функции (включая `WriteWavPcm16`).
- `examples/voice_design_cli_example.cpp`
  CLI пример.
- `examples/voice_design_timing_example.cpp`
  Несколько генераций подряд + тайминги.
- `examples/voice_design_full_profile_example.cpp`
  Профиль одного прогона.
- `CMakeLists.txt`
  Сборка библиотеки `qwen3_tts_cpp` и примеров.

## Зависимости
- C++17
- CMake >= 3.18
- ONNX Runtime:
  - headers (`onnxruntime_cxx_api.h`)
  - shared library (`libonnxruntime.so` / `libonnxruntime.so.*`)
- `Threads` (pthread)
- Для CUDA: ONNX Runtime GPU build + рабочий NVIDIA runtime

## Версия ONNX Runtime
- Рекомендуемая / проверенная: `1.24.1`
- Минимально требуемая для этого экспорта модели: `>= 1.20` (поддержка IR v10)
- На более старых версиях `load()` может падать с ошибкой:
  `Unsupported model IR version: 10, max supported IR version: 9`

## Ожидаемые файлы в `onnx_dir`
- `prefill_builder.onnx`
- `talker_prefill_cache.onnx`
- `talker_decode_cache.onnx`
- `code_predictor_dynamic.onnx` (или step-модели по шаблону)
- `speech_tokenizer_decode.onnx`
- `vocab.json`
- `merges.txt`
- `tokenizer_config.json`

## Файлы модели
- Репозиторий на Hugging Face: https://huggingface.co/abrakadobr/qwen3-tts-onnx-cpp
- Страница файлов: https://huggingface.co/abrakadobr/qwen3-tts-onnx-cpp/tree/main

## Сборка
### Вариант 1: ONNX Runtime из локальной установки
```bash
cmake -S . -B build \
  -DONNX_DIR=/path/to/onnxruntime
cmake --build build -j
```

### Вариант 2: runtime из `venv`
`CMakeLists.txt` умеет автоматически подхватить runtime-библиотеку из:
`venv/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so*`

Но Python wheel обычно не содержит C++ headers, поэтому include-путь нужно передать отдельно:
```bash
cmake -S . -B build \
  -DONNX_INCLUDE_DIR=/path/to/onnxruntime/include
cmake --build build -j
```

## Интеграция в CMake
```cmake
add_subdirectory(path/to/qwen3_tts_cpp)

target_link_libraries(my_tts_app PRIVATE qwen3_tts_cpp)
```

## Минимальный пример API
```cpp
#include "voice.h"
#include "utils.h"

int main() {
  QWEN3TTS::TtsConfig cfg;
  cfg.model.path = "path/to/onnx/model";
  cfg.device = "cpu";
  cfg.intra_threads = 6;
  cfg.inter_threads = 1;

  QWEN3TTS::GenerationParams gen;
  gen.text = "Это тестовая фраза.";
  gen.instruct = "Говори спокойным мягким голосом.";
  gen.max_steps = 160;

  auto* voice = new QWEN3TTS::Voice();
  if (!voice->load(cfg)) {
    // voice->lastErrorCode() / voice->lastErrorMessage()
    delete voice;
    return 1;
  }
  auto pcm = voice->generateVoice(gen);
  if (pcm.size() == 1 && pcm[0] < 0.0f) {
    // код ошибки генерации находится в pcm[0]
    delete voice;
    return 2;
  }
  QWEN3TTSUTILS::WriteWavPcm16("api_example.wav", pcm, 24000);
  voice->unload();
  delete voice;
  return 0;
}
```

## Обработка ошибок
- `Voice::load(...)` возвращает `bool`:
  - `true` при успехе
  - `false` при ошибке (детали: `lastErrorCode()` / `lastErrorMessage()`)
- `Voice::generateVoice(...)` возвращает:
  - обычный PCM-массив при успехе
  - массив из одного отрицательного значения (код ошибки) при провале

### Exit-коды CLI
| Код | Значение |
|---|---|
| `0` | успех |
| `1` | ошибка вызова/usage |
| `2` | некорректные аргументы CLI (парсинг/валидация) |
| `3` | ошибка runtime/модели/генерации |
| `4` | ошибка записи выходного файла (WAV) |

### Частые коды ошибок runtime
| Код | Значение |
|---|---|
| `-1001` | runtime не загружен |
| `-1002` | не инициализирован `MemoryInfo` |
| `-1101` | ошибка токенизатора/входных id |
| `-1102` | некорректный `temperature` |
| `-1103` | некорректный `top-k` |
| `-1104` | некорректные параметры `tail-stop` |
| `-1105` | некорректный `eos_min_steps` |
| `-1106` | некорректный `steps` |
| `-1201` | не сгенерированы аудио-коды |
| `-1202` | после trim не осталось кадров |
| `-1203` | предсказанный код вне диапазона |
| `-1204` | не удалось выбрать первый talker-код |
| `-1301` | ошибка CUDA/provider |
| `-1302` | ошибка ONNX/decode runtime |
| `-1303` | ошибка записи файла codes |
| `-1401` | ошибка загрузки токенизатора |
| `-1402` | ошибка построения id токенизатором |
| `-3001` | некорректный путь модели в `load()` |
| `-3002` | ошибка загрузки модели/сессии |
| `-3003` | неизвестная ошибка `load()` |
| `-3004` | запрошен недоступный CUDA EP |

## Готовые примеры запуска
```bash
./build/qwen3_tts_cpp_cli_example \
  --onnx-dir /path/to/model_dir \
  --text "Привет" \
  --instruct "Говори спокойно." \
  --output-wav artifacts/audio/cli_example.wav \
  --max-steps 120
```

Проверенный запуск (английская фраза):
```bash
./build/qwen3_tts_cpp_cli_example \
  --onnx-dir /path/to/model_dir \
  --text "welcome to qwen3 text to speech model with c++" \
  --instruct "Speak clearly, natural female voice." \
  --output-wav /tmp/qwen3tts_welcome_en.wav \
  --max-steps 80 \
  --device cpu \
  --intra-threads 2 \
  --inter-threads 1
```

## Примечание по совместимости ORT
Если при `load()` появляется ошибка вида
`Unsupported model IR version: 10, max supported IR version: 9`,
значит используется слишком старая версия ONNX Runtime для этой модели.

## Лицензия
Проект распространяется под Apache License 2.0. См. файл `LICENSE`.
