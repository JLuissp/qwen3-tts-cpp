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

double Sec(const Clock::time_point& a, const Clock::time_point& b) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(b - a).count();
}

void PrintTime(const std::string& name, const Clock::time_point& a, const Clock::time_point& b) {
  std::cout << "[time] " << name << ": " << std::fixed << std::setprecision(3) << Sec(a, b) << " sec\n";
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
  const std::filesystem::path out_wav = std::filesystem::path("artifacts") / "audio" / "full_profile_example.wav";
  std::error_code mkerr;
  std::filesystem::create_directories(out_wav.parent_path(), mkerr);
  if (mkerr) {
    std::cerr << "Error: failed to create output dir: " << mkerr.message() << "\n";
    return 2;
  }

  QWEN3TTS::TtsConfig cfg;
  cfg.model.path = onnx_dir;
  cfg.device = "cpu";
  cfg.intra_threads = 6;
  cfg.inter_threads = 1;

  QWEN3TTS::Voice* voice = new QWEN3TTS::Voice();

  const auto t_load_0 = Clock::now();
  if (!voice->load(cfg)) {
    std::cerr << "Load failed with error code: " << voice->lastErrorCode()
              << " (" << voice->lastErrorMessage() << ")\n";
    delete voice;
    return 3;
  }
  const auto t_load_1 = Clock::now();
  PrintTime("load", t_load_0, t_load_1);

  QWEN3TTS::GenerationParams p;
  p.text = "Это полный профилировочный прогон на новом стеке Voice плюс Tokenizer плюс Utils.";
  p.instruct = "Говори спокойно, мягко и разборчиво.";
  p.max_steps = 120;
  p.eos_min_steps = 40;
  p.tail_stop_repeat_frames = 0;

  const auto t_gen_0 = Clock::now();
  auto pcm = voice->generateVoice(p);
  const auto t_gen_1 = Clock::now();
  PrintTime("generate", t_gen_0, t_gen_1);

  float err_code = 0.0f;
  if (IsErrorPcm(pcm, &err_code)) {
    std::cerr << "Generation failed with error code: " << static_cast<int>(err_code) << "\n";
    delete voice;
    return 3;
  }

  std::string wav_err;
  if (!QWEN3TTSUTILS::WriteWavPcm16Safe(out_wav.string(), pcm, 24000, &wav_err)) {
    std::cerr << "WAV write failed: " << wav_err << "\n";
    delete voice;
    return 4;
  }

  const auto t_unload_0 = Clock::now();
  voice->unload();
  delete voice;
  const auto t_unload_1 = Clock::now();
  PrintTime("unload", t_unload_0, t_unload_1);

  std::cout << "Saved: " << out_wav.string() << "\n";
  return 0;
}
