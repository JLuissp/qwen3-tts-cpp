#pragma once

#if __has_include(<onnxruntime_cxx_api.h>)
#include <onnxruntime_cxx_api.h>
#elif __has_include(<onnxruntime/onnxruntime_cxx_api.h>)
#include <onnxruntime/onnxruntime_cxx_api.h>
#else
#error "onnxruntime_cxx_api.h not found. Set include path to ONNX Runtime headers."
#endif

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace QWEN3TTS {

  struct ModelConfig {
    std::string path;
    std::string vocab_file = "vocab.json";
    std::string merges_file = "merges.txt";
    std::string tokenizer_config_file = "tokenizer_config.json";
    std::string prefill_builder_file = "prefill_builder.onnx";
    std::string talker_prefill_file = "talker_prefill_cache.onnx";
    std::string talker_decode_file = "talker_decode_cache.onnx";
    std::string speech_tokenizer_file = "speech_tokenizer_decode.onnx";
    std::string cp_dynamic_file = "code_predictor_dynamic.onnx";
    std::string cp_step_pattern = "code_predictor_step_%02d.onnx";
 
    bool auto_cuda_talker_fp16_fallback = true;
    std::string cuda_talker_fallback_onnx_dir;
    std::string cuda_talker_prefill_fallback_file = "talker_prefill_cache.onnx";
    std::string cuda_talker_decode_fallback_file = "talker_decode_cache.onnx";
  };

  struct TtsConfig {
    ModelConfig             model;
    GraphOptimizationLevel  ort_opt = GraphOptimizationLevel::ORT_ENABLE_ALL;
    int                     intra_threads = 0;
    int                     inter_threads = 0;
    std::string             device = "cpu";
    std::string             prefill_device = "auto";
    std::string             talker_device = "auto";
    std::string             cp_device = "auto";
    std::string             vocoder_device = "auto";
    int                     gpu_device_id = 0;
    int64_t                 gpu_mem_limit_mb = 0;

  };

  struct GenerationParams {
    std::string             text = "";
    std::string             instruct = "";
    std::vector<int64_t>    codec_lang = {-1};
    std::string             wav_out = "./output.wav";
    int                     steps = 0;
    int                     max_steps = 0;
    std::string             codes_out;
    int                     auto_stop_first_code_run = 0;
    int                     auto_stop_min_steps = 40;
    int                     tail_stop_repeat_frames = 8;
    int                     tail_stop_min_steps = 32;
    int                     trim_tail_repeat_min = 24;
    int                     trim_tail_keep = 1;
    int                     eos_min_steps = 0;
    bool                    do_sample = false;
    float                   temperature = 1.0f;
    int                     top_k = 0;
    int64_t                 seed = -1;
  };

  class Voice {
    protected:

  public:
        Voice();
        ~Voice();
      bool load(const TtsConfig& cfg);
      std::vector<float> generateVoice(GenerationParams &params);
      void unload();

      bool isLoaded() const;
      int lastErrorCode() const;
      const std::string& lastErrorMessage() const;

  protected:
      bool BuildVoiceDesignIds();

    private:
        TtsConfig               _config;
        GenerationParams        _params;
        bool                    _loaded = false;
        int                     _last_error_code = 0;
        std::string             _last_error_message;
        std::vector<int64_t>    _input_ids;
        std::vector<int64_t>    _instruct_ids;


    private:
        static constexpr int64_t kBatch = 1;
        static constexpr int64_t kCodeGroups = 16;
        static constexpr int64_t kHidden = 2048;
        static constexpr int64_t kTalkerVocab = 3072;
        static constexpr int64_t kCpVocab = 2048;
        static constexpr int64_t kCodecEosId = 2150;
        static constexpr int kSampleRate = 24000;

        std::unique_ptr<Ort::Env> env_;
        std::optional<Ort::MemoryInfo> mi_;

        std::unique_ptr<Ort::Session> prefill_builder_;
        std::unique_ptr<Ort::Session> talker_prefill_;
        std::unique_ptr<Ort::Session> talker_;
        std::unique_ptr<Ort::Session> vocoder_;
        std::unique_ptr<Ort::Session> cp_dynamic_;
        std::vector<std::unique_ptr<Ort::Session>> cp_steps_;
        bool has_cp_dynamic_ = false;
        bool use_kv_cache_ = false;

    };

}
