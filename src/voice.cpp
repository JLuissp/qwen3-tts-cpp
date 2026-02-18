#include "voice.h"
#include "tokenizer.h"
#include "utils.h"
#include <filesystem>
#include <iostream>
#include <regex>
#include <array>
#include <chrono>
#include <limits>
#include <numeric>
#include <random>
#include <thread>

using namespace QWEN3TTSUTILS;

namespace QWEN3TTS {

Voice::Voice() { }

Voice::~Voice()
{
    unload();
}

bool Voice::load(const TtsConfig &cfg)
{
    _last_error_code = 0;
    _last_error_message.clear();
    if (_loaded) return true;

    auto fail_load = [&](int code, const std::string& msg) -> bool {
        _last_error_code = code;
        _last_error_message = msg;
        std::cerr << "Voice::load failed: " << msg << "\n";
        unload();
        return false;
    };

    try {
    if (cfg.model.path.empty()) {
        return fail_load(-3001, "VoiceDesignConfig.onnx_dir must not be empty");
    }
    const std::filesystem::path base(cfg.model.path);
    const std::filesystem::path fallback(cfg.model.cuda_talker_fallback_onnx_dir);
    _config.model.path = base.string();
    _config.model.cuda_talker_fallback_onnx_dir = fallback.string();
    _config.model.tokenizer_config_file = (base / cfg.model.tokenizer_config_file).string();
    _config.model.speech_tokenizer_file = (base / cfg.model.speech_tokenizer_file).string();
    _config.model.cp_dynamic_file = cfg.model.cp_dynamic_file;
    _config.model.cp_step_pattern = cfg.model.cp_step_pattern;
    _config.model.merges_file = (base / cfg.model.merges_file).string();
    _config.model.prefill_builder_file = (base / cfg.model.prefill_builder_file).string();
    _config.model.talker_prefill_file = (base / cfg.model.talker_prefill_file).string();
    _config.model.talker_decode_file = (base / cfg.model.talker_decode_file).string();
    _config.model.cuda_talker_decode_fallback_file = cfg.model.cuda_talker_decode_fallback_file;
    _config.model.cuda_talker_prefill_fallback_file = cfg.model.cuda_talker_prefill_fallback_file;
    _config.model.auto_cuda_talker_fp16_fallback = cfg.model.auto_cuda_talker_fp16_fallback;

    _config.talker_device = cfg.talker_device;
    _config.cp_device = cfg.cp_device;
    _config.device = cfg.device;
    _config.vocoder_device = cfg.vocoder_device;
    _config.prefill_device = cfg.prefill_device;
    _config.gpu_device_id = cfg.gpu_device_id;
    _config.gpu_mem_limit_mb = cfg.gpu_mem_limit_mb;
    _config.inter_threads = cfg.inter_threads;
    _config.intra_threads = cfg.intra_threads;
    _config.ort_opt = cfg.ort_opt;


    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "qwen3_tts_smoke");
    mi_.emplace(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

    auto configure_device = [&](Ort::SessionOptions& so_local, const std::string& dev_name) -> bool {
        if (dev_name != "cuda") return true;
        if (!HasExecutionProvider("CUDAExecutionProvider")) {
            _last_error_code = -3004;
            _last_error_message = "CUDAExecutionProvider is not available in this onnxruntime build";
            return false;
        }
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = _config.gpu_device_id;
        if (_config.gpu_mem_limit_mb > 0) {
            cuda_opts.gpu_mem_limit = static_cast<size_t>(_config.gpu_mem_limit_mb) * 1024ULL * 1024ULL;
        }
        Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA(so_local, &cuda_opts));
        return true;
    };

    const std::string prefill_device_resolved = (_config.prefill_device == "auto") ? _config.device : _config.prefill_device;
    const std::string talker_device_resolved = (_config.talker_device == "auto") ? _config.device : _config.talker_device;
    const std::string cp_device_resolved =
        (_config.cp_device == "auto") ? ((talker_device_resolved == "cuda") ? "cuda" : _config.device) : _config.cp_device;
    const std::string vocoder_device_resolved = (_config.vocoder_device == "auto") ? _config.device : _config.vocoder_device;
    std::string talker_prefill_path = _config.model.talker_prefill_file;
    std::string talker_path = _config.model.talker_decode_file;

    auto to_lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    };
    auto looks_fp16 = [&](const std::string& s) {
        const std::string low = to_lower(s);
        return low.find("fp16") != std::string::npos || low.find("float16") != std::string::npos;
    };
    auto try_fallback_dir = [&](const std::filesystem::path& dir) -> bool {
        if (dir.empty() || !std::filesystem::exists(dir)) return false;
        const auto prefill_candidate = (dir / _config.model.cuda_talker_prefill_fallback_file).string();
        const auto decode_candidate = (dir / _config.model.cuda_talker_decode_fallback_file).string();
        if (!std::filesystem::exists(prefill_candidate) || !std::filesystem::exists(decode_candidate)) return false;
        if (prefill_candidate == talker_prefill_path && decode_candidate == talker_path) return false;
        talker_prefill_path = prefill_candidate;
        talker_path = decode_candidate;
        std::cout << "[warn] talker-device=cuda with fp16-like bundle detected; "
                  << "falling back to fp32 talker models from: " << dir.string() << "\n";
        return true;
    };
    if (_config.model.auto_cuda_talker_fp16_fallback &&
        talker_device_resolved == "cuda" &&
        (looks_fp16(_config.model.talker_prefill_file) || looks_fp16(_config.model.talker_decode_file))) {
        bool fallback_applied = false;
        if (!_config.model.cuda_talker_fallback_onnx_dir.empty()) {
            fallback_applied = try_fallback_dir(std::filesystem::path(_config.model.cuda_talker_fallback_onnx_dir));
        }
        if (!fallback_applied) {
            const std::filesystem::path base_dir = std::filesystem::path(_config.model.talker_prefill_file).parent_path();
            const std::string base_name = base_dir.filename().string();
            std::smatch m;
            if (std::regex_search(base_name, m, std::regex("onnx_out_v([0-9]+).*fp16", std::regex::icase))) {
                const std::string min_name = "onnx_out_v" + m[1].str() + "_min";
                fallback_applied = try_fallback_dir(base_dir.parent_path() / min_name);
            }
        }
        if (!fallback_applied) {
            std::cout << "[warn] fp16-like talker bundle on CUDA detected but no fp32 fallback found; "
                      << "continuing with configured talker files.\n";
        }
    }

    Ort::SessionOptions so_prefill;
    so_prefill.SetGraphOptimizationLevel(_config.ort_opt);
    if (_config.intra_threads > 0) so_prefill.SetIntraOpNumThreads(_config.intra_threads);
    if (_config.inter_threads > 0) so_prefill.SetInterOpNumThreads(_config.inter_threads);
    if (!configure_device(so_prefill, prefill_device_resolved)) {
        return fail_load(_last_error_code, _last_error_message);
    }

    Ort::SessionOptions so_talker;
    so_talker.SetGraphOptimizationLevel(_config.ort_opt);
    if (_config.intra_threads > 0) so_talker.SetIntraOpNumThreads(_config.intra_threads);
    if (_config.inter_threads > 0) so_talker.SetInterOpNumThreads(_config.inter_threads);
    if (!configure_device(so_talker, talker_device_resolved)) {
        return fail_load(_last_error_code, _last_error_message);
    }

    Ort::SessionOptions so_cp;
    so_cp.SetGraphOptimizationLevel(_config.ort_opt);
    if (_config.intra_threads > 0) so_cp.SetIntraOpNumThreads(_config.intra_threads);
    if (_config.inter_threads > 0) so_cp.SetInterOpNumThreads(_config.inter_threads);
    if (!configure_device(so_cp, cp_device_resolved)) {
        return fail_load(_last_error_code, _last_error_message);
    }

    Ort::SessionOptions so_vocoder;
    so_vocoder.SetGraphOptimizationLevel(_config.ort_opt);
    if (_config.intra_threads > 0) so_vocoder.SetIntraOpNumThreads(_config.intra_threads);
    if (_config.inter_threads > 0) so_vocoder.SetInterOpNumThreads(_config.inter_threads);
    if (!configure_device(so_vocoder, vocoder_device_resolved)) {
        return fail_load(_last_error_code, _last_error_message);
    }

    prefill_builder_ = std::make_unique<Ort::Session>(*env_, _config.model.prefill_builder_file.c_str(), so_prefill);
    talker_prefill_ = std::make_unique<Ort::Session>(*env_, talker_prefill_path.c_str(), so_talker);
    talker_ = std::make_unique<Ort::Session>(*env_, talker_path.c_str(), so_talker);

    const std::string cp_dynamic_path = (std::filesystem::path(_config.model.path) / _config.model.cp_dynamic_file).string();
    has_cp_dynamic_ = std::filesystem::exists(cp_dynamic_path);
    if (has_cp_dynamic_) {
        cp_dynamic_ = std::make_unique<Ort::Session>(*env_, cp_dynamic_path.c_str(), so_cp);
        std::cout << "[cp] using shared dynamic model: " << cp_dynamic_path << "\n";
    } else {
        cp_steps_.clear();
        cp_steps_.reserve(static_cast<size_t>(kCodeGroups - 1));
        for (int g = 0; g < kCodeGroups - 1; ++g) {
            char suffix[64];
            std::snprintf(suffix, sizeof(suffix), _config.model.cp_step_pattern.c_str(), g);
            const std::string cp_path = (std::filesystem::path(_config.model.path) / suffix).string();
            cp_steps_.emplace_back(std::make_unique<Ort::Session>(*env_, cp_path.c_str(), so_cp));
        }
        std::cout << "[cp] using legacy fixed-step models from: " << _config.model.path << "\n";
    }

    vocoder_ = std::make_unique<Ort::Session>(*env_, _config.model.speech_tokenizer_file.c_str(), so_vocoder);
    use_kv_cache_ = (talker_prefill_->GetOutputCount() >= 4 && talker_->GetInputCount() >= 5);


    _loaded = true;
    _last_error_code = 0;
    _last_error_message.clear();
    return true;
    } catch (const std::exception& e) {
        return fail_load(-3002, e.what());
    } catch (...) {
        return fail_load(-3003, "unknown exception");
    }
    return false;
}

std::vector<float> Voice::generateVoice(GenerationParams &params)
{
    auto err_pcm = [](float code) { return std::vector<float>{code}; };
    auto fail_gen = [&](int code, const std::string& msg) {
        std::cerr << "Voice::generateVoice failed: " << msg << "\n";
        _last_error_code = code;
        _last_error_message = msg;
        return err_pcm(static_cast<float>(code));
    };
    try {
    if (!_loaded) {
        return fail_gen(-1001, "runtime is not loaded");
    }
    if (!mi_.has_value()) {
        return fail_gen(-1002, "memory info is not initialized");
    }
    _params = params;
    if (!BuildVoiceDesignIds()) return fail_gen(_last_error_code, _last_error_message);

    if (_input_ids.empty() || _instruct_ids.empty()) return fail_gen(-1101, "Empty input_ids or instruct_ids");
    if (_params.temperature < 0.0f) return fail_gen(-1102, "temperature must be >= 0");
    if (_params.top_k < 0) return fail_gen(-1103, "top_k must be >= 0");
    if (_params.tail_stop_repeat_frames < 0) return fail_gen(-1104, "tail_stop_repeat_frames must be >= 0");
    if (_params.tail_stop_min_steps < 0) return fail_gen(-1104, "tail_stop_min_steps must be >= 0");
    if (_params.eos_min_steps < 0) return fail_gen(-1105, "eos_min_steps must be >= 0");

    int steps = _params.steps;
    if (steps <= 0) {
        if (_params.max_steps > 0) {
            steps = _params.max_steps;
            std::cout << "[auto-steps] selected=max_steps=" << steps
                      << " (EOS/tail-stop may finish earlier)\n";
        } else {
            steps = 2000;
            std::cout << "[auto-steps] selected=tail-mode, safety_cap=" << steps
                      << " (set max_steps to limit)\n";
        }
    }
    if (steps <= 0) return fail_gen(-1106, "steps must be > 0");

    uint64_t seed = 0;
    if (_params.seed >= 0) {
        seed = static_cast<uint64_t>(_params.seed);
    } else {
        std::random_device rd;
        seed = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
    }
    std::mt19937_64 rng(seed);

    std::vector<int64_t> input_ids = _input_ids;
    std::vector<int64_t> instruct_ids = _instruct_ids;
    std::vector<int64_t> codec_lang = _params.codec_lang;
    if (codec_lang.empty()) {
        codec_lang = {-1};
    }

    auto input_ids_tensor = MakeTensorI64(*mi_, input_ids, {1, static_cast<int64_t>(input_ids.size())});
    auto instruct_ids_tensor = MakeTensorI64(*mi_, instruct_ids, {1, static_cast<int64_t>(instruct_ids.size())});
    auto lang_tensor = MakeTensorI64(*mi_, codec_lang, {1});
    const char* pb_in_names[] = {"input_ids", "instruct_ids", "codec_language_token_id"};
    const char* pb_out_names[] = {"prefill_embeds", "tts_pad_embed"};
    std::array<Ort::Value, 3> pb_inputs = {
        std::move(input_ids_tensor), std::move(instruct_ids_tensor), std::move(lang_tensor)};
    auto pb_out = prefill_builder_->Run(
        Ort::RunOptions{nullptr}, pb_in_names, pb_inputs.data(), pb_inputs.size(), pb_out_names, 2);

    Ort::Value prefill_embeds = std::move(pb_out[0]);
    Ort::Value tts_pad_embed_val = std::move(pb_out[1]);
    float* tts_pad_ptr = tts_pad_embed_val.GetTensorMutableData<float>();
    std::vector<float> trailing_step(tts_pad_ptr, tts_pad_ptr + kHidden);

    auto prefill_shape = prefill_embeds.GetTensorTypeAndShapeInfo().GetShape();
    const int64_t prefill_len = prefill_shape[1];
    const int64_t prefill_elems = std::accumulate(
        prefill_shape.begin(), prefill_shape.end(), int64_t{1}, std::multiplies<int64_t>());
    float* prefill_ptr = prefill_embeds.GetTensorMutableData<float>();
    std::vector<float> prefill_copy(prefill_ptr, prefill_ptr + prefill_elems);
    auto prefill_tensor = MakeTensorF32(*mi_, prefill_copy, prefill_shape);

    const char* tp_in_names[] = {"prefill_embeds"};
    std::vector<Ort::Value> tp_out;
    std::array<Ort::Value, 1> tp_inputs = {std::move(prefill_tensor)};
    if (use_kv_cache_) {
        const char* tp_out_names_cache[] = {"logits", "last_hidden", "present_k", "present_v"};
        tp_out = talker_prefill_->Run(
            Ort::RunOptions{nullptr}, tp_in_names, tp_inputs.data(), 1, tp_out_names_cache, 4);
    } else {
        const char* tp_out_names[] = {"logits", "last_hidden"};
        tp_out = talker_prefill_->Run(
            Ort::RunOptions{nullptr}, tp_in_names, tp_inputs.data(), 1, tp_out_names, 2);
    }

    float* prefill_logits_ptr = tp_out[0].GetTensorMutableData<float>();
    int64_t first_code = SelectTalkerFirstCode(
        prefill_logits_ptr,
        kTalkerVocab,
        kCpVocab,
        kCodecEosId,
        0 >= _params.eos_min_steps,
        _params.do_sample,
        _params.temperature,
        _params.top_k,
        &rng);
    if (first_code < 0 || first_code >= kTalkerVocab) {
        return fail_gen(-1204, "Failed to select first talker code");
    }
    float* prefill_last_hidden_ptr = tp_out[1].GetTensorMutableData<float>();

    std::vector<int64_t> codec_ids(kCodeGroups, 1);
    std::vector<int64_t> all_codes;
    all_codes.reserve(static_cast<size_t>(steps * kCodeGroups));
    std::vector<int64_t> prev_codes(kCodeGroups - 2, 0);
    std::vector<int64_t> first_code_vec(1, 0);
    std::vector<int64_t> codec_step_vec(kCodeGroups, 0);
    std::vector<float> trailing_step_vec = trailing_step;
    std::vector<int64_t> cache_pos_vec(1, 0);
    std::vector<int64_t> step_id_vec(1, 0);
    const char* cp_in_names[] = {"past_hidden", "first_code_id", "prev_codes"};
    const char* cp_out_names[] = {"logits"};
    const char* cp_dyn_in_names[] = {"past_hidden", "first_code_id", "prev_codes", "step_id"};

    std::vector<float> current_past_hidden(prefill_last_hidden_ptr, prefill_last_hidden_ptr + kHidden);
    Ort::Value past_k_cache{nullptr};
    Ort::Value past_v_cache{nullptr};
    if (use_kv_cache_) {
        past_k_cache = std::move(tp_out[2]);
        past_v_cache = std::move(tp_out[3]);
    }
    int64_t current_first_code = first_code;
    int64_t prev_generated_first_code = std::numeric_limits<int64_t>::min();
    int same_first_code_run = 0;
    std::vector<int64_t> prev_frame(kCodeGroups, std::numeric_limits<int64_t>::min());
    int same_frame_run = 0;


    for (int s = 0; s < steps; ++s) {
        if (s > 0 && current_first_code == kCodecEosId && s >= _params.eos_min_steps) {
            break;
        }
        codec_ids[0] = current_first_code;
        std::fill(prev_codes.begin(), prev_codes.end(), 0);

        for (int g = 0; g < kCodeGroups - 1; ++g) {
            auto past_hidden_tensor = MakeTensorF32(*mi_, current_past_hidden, {kBatch, 1, kHidden});
            first_code_vec[0] = codec_ids[0];
            auto first_code_tensor = MakeTensorI64(*mi_, first_code_vec, {kBatch, 1});
            auto prev_codes_tensor = MakeTensorI64(*mi_, prev_codes, {kBatch, kCodeGroups - 2});
            std::array<Ort::Value, 3> cp_inputs = {
                std::move(past_hidden_tensor), std::move(first_code_tensor), std::move(prev_codes_tensor)};
            std::vector<Ort::Value> cp_out;
            if (has_cp_dynamic_) {
                step_id_vec[0] = g;
                auto step_id_tensor = MakeTensorI64(*mi_, step_id_vec, {1});
                std::array<Ort::Value, 4> cp_dyn_inputs = {
                    std::move(cp_inputs[0]), std::move(cp_inputs[1]), std::move(cp_inputs[2]), std::move(step_id_tensor)};
                cp_out = cp_dynamic_->Run(
                    Ort::RunOptions{nullptr},
                    cp_dyn_in_names,
                    cp_dyn_inputs.data(),
                    cp_dyn_inputs.size(),
                    cp_out_names,
                    1);
            } else {
                cp_out = cp_steps_[static_cast<size_t>(g)]->Run(
                    Ort::RunOptions{nullptr}, cp_in_names, cp_inputs.data(), cp_inputs.size(), cp_out_names, 1);
            }

            float* cp_logits_ptr = cp_out[0].GetTensorMutableData<float>();
            const int64_t pred = SelectCpCode(
                cp_logits_ptr, kCpVocab, _params.do_sample, _params.temperature, _params.top_k, &rng);
            if (pred < 0 || pred >= kCpVocab) return fail_gen(-1203, "Predicted cp code out of range");
            codec_ids[g + 1] = pred;
            if (g < kCodeGroups - 2) {
                prev_codes[g] = pred;
            }
        }

        all_codes.insert(all_codes.end(), codec_ids.begin(), codec_ids.end());
        if (codec_ids[0] == prev_generated_first_code) {
            ++same_first_code_run;
        } else {
            same_first_code_run = 1;
            prev_generated_first_code = codec_ids[0];
        }
        if (s > 0 && codec_ids == prev_frame) {
            ++same_frame_run;
        } else {
            same_frame_run = 1;
        }
        prev_frame = codec_ids;
        const int generated_now = s + 1;
        if (_params.tail_stop_repeat_frames > 0 &&
            generated_now >= _params.tail_stop_min_steps &&
            same_frame_run >= _params.tail_stop_repeat_frames) {
            std::cout << "[tail-stop] repeated full frame " << same_frame_run
                      << " times at step=" << generated_now << "\n";
            break;
        }
        if (_params.auto_stop_first_code_run > 0 &&
            generated_now >= _params.auto_stop_min_steps &&
            same_first_code_run >= _params.auto_stop_first_code_run) {
            std::cout << "[auto-stop] repeated first code " << same_first_code_run
                      << " times at step=" << generated_now << "\n";
            break;
        }

        if (s == steps - 1) {
            break;
        }

        std::vector<Ort::Value> talker_out;
        if (use_kv_cache_) {
            codec_step_vec = codec_ids;
            cache_pos_vec[0] = prefill_len + static_cast<int64_t>(s);
            auto codec_step_tensor = MakeTensorI64(*mi_, codec_step_vec, {kBatch, 1, kCodeGroups});
            auto trailing_step_tensor = MakeTensorF32(*mi_, trailing_step_vec, {kBatch, 1, kHidden});
            auto cache_pos_tensor = MakeTensorI64(*mi_, cache_pos_vec, {1});
            const char* talker_in_names[] = {"codec_ids_step", "trailing_text_step", "past_k", "past_v", "cache_position"};
            const char* talker_out_names[] = {"logits", "last_hidden", "present_k", "present_v"};
            std::array<Ort::Value, 5> talker_inputs = {
                std::move(codec_step_tensor),
                std::move(trailing_step_tensor),
                std::move(past_k_cache),
                std::move(past_v_cache),
                std::move(cache_pos_tensor),
            };
            talker_out = talker_->Run(
                Ort::RunOptions{nullptr}, talker_in_names, talker_inputs.data(), talker_inputs.size(), talker_out_names, 4);
        } else {
            const int64_t hist_len = static_cast<int64_t>(s + 1);
            std::vector<int64_t> codec_hist(
                all_codes.end() - static_cast<long>(hist_len * kCodeGroups), all_codes.end());
            std::vector<float> trailing_hist(static_cast<size_t>(hist_len * kHidden), 0.0f);
            for (int64_t t = 0; t < hist_len; ++t) {
                std::copy(trailing_step.begin(), trailing_step.end(), trailing_hist.begin() + t * kHidden);
            }
            std::vector<float> prefill_run = prefill_copy;
            auto prefill_hist_tensor = MakeTensorF32(*mi_, prefill_run, prefill_shape);
            auto codec_tensor = MakeTensorI64(*mi_, codec_hist, {kBatch, hist_len, kCodeGroups});
            auto trailing_tensor = MakeTensorF32(*mi_, trailing_hist, {kBatch, hist_len, kHidden});
            const char* talker_in_names[] = {"prefill_embeds", "codec_ids", "trailing_text"};
            const char* talker_out_names[] = {"logits", "last_hidden"};
            std::array<Ort::Value, 3> talker_inputs = {
                std::move(prefill_hist_tensor), std::move(codec_tensor), std::move(trailing_tensor)};
            talker_out = talker_->Run(
                Ort::RunOptions{nullptr}, talker_in_names, talker_inputs.data(), talker_inputs.size(), talker_out_names, 2);
        }

        float* logits_ptr = talker_out[0].GetTensorMutableData<float>();
        current_first_code = SelectTalkerFirstCode(
            logits_ptr,
            kTalkerVocab,
            kCpVocab,
            kCodecEosId,
            (s + 1) >= _params.eos_min_steps,
            _params.do_sample,
            _params.temperature,
            _params.top_k,
            &rng);
        if (current_first_code < 0 || current_first_code >= kTalkerVocab) {
            return fail_gen(-1204, "Failed to select first talker code");
        }

        float* last_hidden_ptr = talker_out[1].GetTensorMutableData<float>();
        current_past_hidden.assign(last_hidden_ptr, last_hidden_ptr + kHidden);
        if (use_kv_cache_) {
            past_k_cache = std::move(talker_out[2]);
            past_v_cache = std::move(talker_out[3]);
        }
    }

    int generated_steps = static_cast<int>(all_codes.size() / static_cast<size_t>(kCodeGroups));
    if (generated_steps <= 0) return fail_gen(-1201, "No audio codes generated (EOS too early or decoding failed)");

    std::vector<int64_t> audio_codes = all_codes;
    if (_params.trim_tail_repeat_min > 0) {
        const int before_steps = generated_steps;
        generated_steps = TrimRepeatingTailFrames(
            &audio_codes, static_cast<int>(kCodeGroups), _params.trim_tail_repeat_min, _params.trim_tail_keep);
        if (generated_steps < before_steps) {
            std::cout << "[trim] removed tail repeated frames=" << (before_steps - generated_steps)
                      << ", remaining_steps=" << generated_steps << "\n";
        }
    }
    if (generated_steps <= 0) return fail_gen(-1202, "All generated frames were trimmed; adjust trim settings.");

    if (!_params.codes_out.empty()) {
        std::string write_codes_err;
        if (!WriteCodesTxtSafe(_params.codes_out, audio_codes, generated_steps, static_cast<int>(kCodeGroups), &write_codes_err)) {
            return fail_gen(-1303, write_codes_err.empty() ? "failed to write codes" : write_codes_err);
        }
    }
    std::vector<float> wav;
    std::string decode_err;
    if (!DecodeAudioCodesSafe(*vocoder_, *mi_, audio_codes, generated_steps, static_cast<int>(kCodeGroups), &wav, &decode_err)) {
        return fail_gen(-1302, decode_err.empty() ? "failed to decode audio codes" : decode_err);
    }

    std::cout << "Samples: " << static_cast<int64_t>(wav.size()) << ", sample_rate: " << kSampleRate << "\n";
    std::cout << "Decoder path: AR code predictor step model enabled"
              << (use_kv_cache_ ? " + talker KV cache.\n" : ".\n");
    _last_error_code = 0;
    _last_error_message.clear();
    return wav;
    } catch (const std::exception& e) {
        std::cerr << "Voice::generateVoice failed: " << e.what() << "\n";
        const std::string msg = e.what();
        if (msg.find("input_ids") != std::string::npos || msg.find("instruct_ids") != std::string::npos) {
            _last_error_code = -1101;
            _last_error_message = msg;
            return err_pcm(-1101.0f);
        }
        if (msg.find("temperature") != std::string::npos) {
            _last_error_code = -1102;
            _last_error_message = msg;
            return err_pcm(-1102.0f);
        }
        if (msg.find("top_k") != std::string::npos) {
            _last_error_code = -1103;
            _last_error_message = msg;
            return err_pcm(-1103.0f);
        }
        if (msg.find("tail_stop") != std::string::npos) {
            _last_error_code = -1104;
            _last_error_message = msg;
            return err_pcm(-1104.0f);
        }
        if (msg.find("eos_min_steps") != std::string::npos) {
            _last_error_code = -1105;
            _last_error_message = msg;
            return err_pcm(-1105.0f);
        }
        if (msg.find("No audio codes generated") != std::string::npos) {
            _last_error_code = -1201;
            _last_error_message = msg;
            return err_pcm(-1201.0f);
        }
        if (msg.find("All generated frames were trimmed") != std::string::npos) {
            _last_error_code = -1202;
            _last_error_message = msg;
            return err_pcm(-1202.0f);
        }
        if (msg.find("CUDA") != std::string::npos) {
            _last_error_code = -1301;
            _last_error_message = msg;
            return err_pcm(-1301.0f);
        }
        if (msg.find("onnx") != std::string::npos || msg.find("Ort") != std::string::npos) {
            _last_error_code = -1302;
            _last_error_message = msg;
            return err_pcm(-1302.0f);
        }
        _last_error_code = -1999;
        _last_error_message = msg;
        return err_pcm(-1999.0f);
    } catch (...) {
        std::cerr << "Voice::generateVoice failed: unknown exception\n";
        _last_error_code = -2000;
        _last_error_message = "unknown exception";
        return err_pcm(-2000.0f);
    }
}

void Voice::unload()
{
    cp_steps_.clear();
    cp_dynamic_.reset();
    vocoder_.reset();
    talker_.reset();
    talker_prefill_.reset();
    prefill_builder_.reset();
    env_.reset();
    has_cp_dynamic_ = false;
    use_kv_cache_ = false;
    mi_.reset();
    _loaded = false;
}

bool Voice::isLoaded() const
{
        return _loaded;
}

int Voice::lastErrorCode() const
{
    return _last_error_code;
}

const std::string& Voice::lastErrorMessage() const
{
    return _last_error_message;
}

bool Voice::BuildVoiceDesignIds()
{
    const auto vocab_file = std::filesystem::path(_config.model.vocab_file).filename().string();
    const auto merges_file = std::filesystem::path(_config.model.merges_file).filename().string();
    const auto tokenizer_config_file = std::filesystem::path(_config.model.tokenizer_config_file).filename().string();

    VoiceTokenizer tok;
    std::string tok_err;
    if (!tok.LoadSafe(_config.model.path, vocab_file, merges_file, tokenizer_config_file, &tok_err)) {
        _last_error_code = -1401;
        _last_error_message = tok_err.empty() ? "tokenizer load failed" : tok_err;
        return false;
    }
    if (!tok.BuildVoiceDesignIdsSafe(_params.text, _params.instruct, &_input_ids, &_instruct_ids, &tok_err)) {
        _last_error_code = -1402;
        _last_error_message = tok_err.empty() ? "tokenizer build ids failed" : tok_err;
        return false;
    }
    return true;
}

}
