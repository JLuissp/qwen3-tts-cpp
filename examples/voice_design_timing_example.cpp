#include "voice.h"
#include "utils.h"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double ElapsedSec(const Clock::time_point& start, const Clock::time_point& end) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

void PrintStepTime(const std::string& name, const Clock::time_point& start, const Clock::time_point& end) {
  std::cout << "[time] " << name << ": " << std::fixed << std::setprecision(3)
            << ElapsedSec(start, end) << " sec\n" << std::flush;
}

bool IsErrorPcm(const std::vector<float>& pcm, float* code) {
  if (pcm.size() == 1 && pcm[0] < 0.0f) {
    *code = pcm[0];
    return true;
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  const std::string onnx_dir = (argc > 1) ? argv[1] : "onnx_out_v11_min";
  const std::filesystem::path out_dir = std::filesystem::path("artifacts") / "audio";
  std::error_code mkerr;
  std::filesystem::create_directories(out_dir, mkerr);
  if (mkerr) {
    std::cerr << "Error: failed to create output dir: " << mkerr.message() << "\n";
    return 2;
  }

  const auto total_start = Clock::now();

  QWEN3TTS::TtsConfig cfg;
  cfg.model.path = onnx_dir;
  cfg.device = "cpu";
  cfg.intra_threads = 6;
  cfg.inter_threads = 1;

  QWEN3TTS::Voice* voice = new QWEN3TTS::Voice();
  const auto load_start = Clock::now();
  if (!voice->load(cfg)) {
    std::cerr << "Load failed with error code: " << voice->lastErrorCode()
              << " (" << voice->lastErrorMessage() << ")\n";
    delete voice;
    return 3;
  }
  const auto load_end = Clock::now();
  PrintStepTime("load", load_start, load_end);

  auto run_generation = [&](const std::string& step_name,
                            const std::string& text,
                            const std::string& instruct,
                            const std::string& wav_name,
                            int max_steps,
                            int eos_min_steps) -> bool {
    QWEN3TTS::GenerationParams p;
    p.text = text;
    p.instruct = instruct;
    p.max_steps = max_steps;
    p.eos_min_steps = eos_min_steps;
    p.tail_stop_repeat_frames = 0;
    p.tail_stop_min_steps = 0;
    p.trim_tail_repeat_min = 24;
    p.trim_tail_keep = 1;

    const auto t0 = Clock::now();
    auto pcm = voice->generateVoice(p);
    const auto t1 = Clock::now();
    float err_code = 0.0f;
    if (IsErrorPcm(pcm, &err_code)) {
      std::cerr << "Generation failed in step '" << step_name
                << "' with error code: " << static_cast<int>(err_code) << "\n";
      return false;
    }
    std::string wav_err;
    if (!QWEN3TTSUTILS::WriteWavPcm16Safe((out_dir / wav_name).string(), pcm, 24000, &wav_err)) {
      std::cerr << "WAV write failed in step '" << step_name << "': " << wav_err << "\n";
      return false;
    }
    PrintStepTime(step_name, t0, t1);
    return true;
  };

  if (!run_generation(
        "generate_short_1",
        "Привет. Это первый короткий тест после загрузки модели.",
        "Говори спокойным, мягким, женским голосом, естественно и ровно.",
        "timing_short_1.wav",
        160,
        32)) {
    delete voice;
    return 3;
  }

  if (!run_generation(
        "generate_long",
        "Сегодня мы запускаем длинную тестовую фразу, чтобы проверить стабильность генерации в одном процессе: "
        "модель уже загружена, поэтому нам важно видеть время на повторные запросы, качество речи и отсутствие "
        "обрывов в середине предложения.",
        "Говори спокойным, уверенным голосом, с чёткой дикцией и плавной интонацией.",
        "timing_long.wav",
        420,
        48)) {
    delete voice;
    return 3;
  }

  if (!run_generation(
        "generate_short_2",
        "Финальный короткий тест после длинной фразы.",
        "Говори спокойно и разборчиво.",
        "timing_short_2.wav",
        160,
        32)) {
    delete voice;
    return 3;
  }

  if (!run_generation(
        "generate_short_3",
        "И ещё один короткий тест, чтобы проверить хвост после нескольких последовательных запросов.",
        "Говори спокойно и мягко.",
        "timing_short_3.wav",
        180,
        32)) {
    delete voice;
    return 3;
  }

  const auto unload_start = Clock::now();
  voice->unload();
  delete voice;
  const auto unload_end = Clock::now();
  PrintStepTime("unload", unload_start, unload_end);

  const auto total_end = Clock::now();
  PrintStepTime("total", total_start, total_end);

  std::cout << "Saved files:\n";
  std::cout << "  " << (out_dir / "timing_short_1.wav").string() << "\n";
  std::cout << "  " << (out_dir / "timing_long.wav").string() << "\n";
  std::cout << "  " << (out_dir / "timing_short_2.wav").string() << "\n";
  std::cout << "  " << (out_dir / "timing_short_3.wav").string() << "\n";
  return 0;
}
