#include "utils.h"


#include <algorithm>
#include <cstdint>
// #include <cctype>
// #include <cstring>
// #include <cmath>
// #include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
// #include <limits>
// #include <numeric>
// #include <optional>
// #include <random>

#include <stdexcept>
#include <string>
// #include <memory>
// #include <chrono>
// #include <thread>
#include <set>
// #include <unordered_map>
#include <vector>
// #include <array>

namespace QWEN3TTSUTILS {


int64_t Argmax(const float* data, int64_t size) {
    int64_t best = 0;
    float best_val = data[0];
    for (int64_t i = 1; i < size; ++i) {
        if (data[i] > best_val) {
            best_val = data[i];
            best = i;
        }
    }
    return best;
}

int64_t ArgmaxTalkerFirstCode(
    const float* data,
    int64_t talker_vocab,
    int64_t cp_vocab,
    int64_t codec_eos_id,
    bool allow_eos) {
    bool has_best = false;
    int64_t best = 0;
    float best_val = 0.0f;
    for (int64_t i = 0; i < talker_vocab; ++i) {
        const bool is_eos = (i == codec_eos_id);
        const bool suppressed = (i >= cp_vocab) && (!is_eos || !allow_eos);
        if (suppressed) continue;
        if (!has_best || data[i] > best_val) {
            best_val = data[i];
            best = i;
            has_best = true;
        }
    }
    return has_best ? best : -1;
}

int64_t SampleFromCandidates(
    const std::vector<std::pair<float, int64_t>>& candidates,
    float temperature,
    int top_k,
    std::mt19937_64* rng) {
    if (candidates.empty()) return -1;
    if (temperature <= 0.0f) {
        float best_val = candidates[0].first;
        int64_t best_idx = candidates[0].second;
        for (size_t i = 1; i < candidates.size(); ++i) {
            if (candidates[i].first > best_val) {
                best_val = candidates[i].first;
                best_idx = candidates[i].second;
            }
        }
        return best_idx;
    }

    std::vector<std::pair<float, int64_t>> filtered = candidates;
    if (top_k > 0 && top_k < static_cast<int>(filtered.size())) {
        std::nth_element(
            filtered.begin(),
            filtered.begin() + top_k,
            filtered.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        filtered.resize(static_cast<size_t>(top_k));
    }

    float max_scaled = -std::numeric_limits<float>::infinity();
    for (const auto& item : filtered) {
        const float scaled = item.first / temperature;
        if (scaled > max_scaled) max_scaled = scaled;
    }
    std::vector<double> weights;
    weights.reserve(filtered.size());
    for (const auto& item : filtered) {
        const double scaled = static_cast<double>(item.first / temperature - max_scaled);
        weights.push_back(std::exp(scaled));
    }

    if (!rng) return -1;
    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    const size_t sampled = dist(*rng);
    return filtered[sampled].second;
}

int64_t SelectTalkerFirstCode(
    const float* data,
    int64_t talker_vocab,
    int64_t cp_vocab,
    int64_t codec_eos_id,
    bool allow_eos,
    bool do_sample,
    float temperature,
    int top_k,
    std::mt19937_64* rng) {
    if (!do_sample || temperature <= 0.0f) {
        return ArgmaxTalkerFirstCode(data, talker_vocab, cp_vocab, codec_eos_id, allow_eos);
    }
    std::vector<std::pair<float, int64_t>> candidates;
    candidates.reserve(static_cast<size_t>(cp_vocab + 1));
    for (int64_t i = 0; i < talker_vocab; ++i) {
        const bool is_eos = (i == codec_eos_id);
        const bool suppressed = (i >= cp_vocab) && (!is_eos || !allow_eos);
        if (suppressed) continue;
        candidates.emplace_back(data[i], i);
    }
    return SampleFromCandidates(candidates, temperature, top_k, rng);
}

int64_t SelectCpCode(
    const float* data,
    int64_t cp_vocab,
    bool do_sample,
    float temperature,
    int top_k,
    std::mt19937_64* rng) {
    if (!do_sample || temperature <= 0.0f) {
        return Argmax(data, cp_vocab);
    }
    std::vector<std::pair<float, int64_t>> candidates;
    candidates.reserve(static_cast<size_t>(cp_vocab));
    for (int64_t i = 0; i < cp_vocab; ++i) {
        candidates.emplace_back(data[i], i);
    }
    return SampleFromCandidates(candidates, temperature, top_k, rng);
}

void WriteWavPcm16(const std::string& path, const std::vector<float>& samples, int sample_rate) {
    std::vector<float> processed = samples;
    const size_t tail_probe = std::min<size_t>(processed.size(), static_cast<size_t>(sample_rate / 100));  // 10 ms
    float tail_peak = 0.0f;
    for (size_t i = 0; i < tail_probe; ++i) {
        const float v = std::abs(processed[processed.size() - tail_probe + i]);
        if (v > tail_peak) tail_peak = v;
    }

    // Stronger default fade to suppress end-clicks.
    size_t fade_ms = 40;
    // If tail is still hot, apply an even stronger fade window.
    if (tail_peak > 0.35f) fade_ms = 80;
    const size_t fade_samples = std::min<size_t>(processed.size(), static_cast<size_t>((sample_rate * fade_ms) / 1000));
    if (fade_samples > 1) {
        const size_t start = processed.size() - fade_samples;
        for (size_t i = 0; i < fade_samples; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(fade_samples - 1);
            const float gain = (1.0f - t) * (1.0f - t);  // Quadratic fade-out.
            processed[start + i] *= gain;
        }
    }

    // Add a short silence pad so players don't cut exactly on a non-zero edge.
    const size_t pad_samples = static_cast<size_t>((sample_rate * 30) / 1000);  // 30 ms
    processed.insert(processed.end(), pad_samples, 0.0f);

    std::vector<int16_t> pcm(processed.size());
    for (size_t i = 0; i < processed.size(); ++i) {
        float v = std::max(-1.0f, std::min(1.0f, processed[i]));
        pcm[i] = static_cast<int16_t>(v * 32767.0f);
    }

    const uint32_t data_bytes = static_cast<uint32_t>(pcm.size() * sizeof(int16_t));
    const uint32_t riff_size = 36u + data_bytes;
    const uint16_t audio_format = 1;
    const uint16_t num_channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    const uint16_t block_align = num_channels * bits_per_sample / 8;

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "WriteWavPcm16 failed to open output wav: " << path << "\n";
        return;
    }

    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char*>(&riff_size), sizeof(riff_size));
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    const uint32_t fmt_size = 16;
    out.write(reinterpret_cast<const char*>(&fmt_size), sizeof(fmt_size));
    out.write(reinterpret_cast<const char*>(&audio_format), sizeof(audio_format));
    out.write(reinterpret_cast<const char*>(&num_channels), sizeof(num_channels));
    out.write(reinterpret_cast<const char*>(&sample_rate), sizeof(sample_rate));
    out.write(reinterpret_cast<const char*>(&byte_rate), sizeof(byte_rate));
    out.write(reinterpret_cast<const char*>(&block_align), sizeof(block_align));
    out.write(reinterpret_cast<const char*>(&bits_per_sample), sizeof(bits_per_sample));
    out.write("data", 4);
    out.write(reinterpret_cast<const char*>(&data_bytes), sizeof(data_bytes));
    out.write(reinterpret_cast<const char*>(pcm.data()), static_cast<std::streamsize>(data_bytes));
}

bool WriteWavPcm16Safe(
    const std::string& path,
    const std::vector<float>& samples,
    int sample_rate,
    std::string* error) {
    std::ofstream probe(path, std::ios::binary | std::ios::app);
    if (!probe) {
        if (error) *error = "Failed to open output wav: " + path;
        return false;
    }
    probe.close();
    WriteWavPcm16(path, samples, sample_rate);
    if (error) error->clear();
    return true;
}

Ort::Value MakeTensorI64(const Ort::MemoryInfo& mi, std::vector<int64_t>& data, const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<int64_t>(mi, data.data(), data.size(), shape.data(), shape.size());
}

Ort::Value MakeTensorF32(const Ort::MemoryInfo& mi, std::vector<float>& data, const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<float>(mi, data.data(), data.size(), shape.data(), shape.size());
}

GraphOptimizationLevel ParseGraphOptimizationLevel(const std::string& s) {
    if (s == "disable") return GraphOptimizationLevel::ORT_DISABLE_ALL;
    if (s == "basic") return GraphOptimizationLevel::ORT_ENABLE_BASIC;
    if (s == "extended") return GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
    if (s == "all") return GraphOptimizationLevel::ORT_ENABLE_ALL;
    return GraphOptimizationLevel::ORT_ENABLE_ALL;
}

bool HasExecutionProvider(const std::string &ep_name) {
    const auto providers = Ort::GetAvailableProviders();
    std::set<std::string> uniq(providers.begin(), providers.end());
    return uniq.find(ep_name) != uniq.end();
}

std::vector<float> DecodeAudioCodes(
    Ort::Session& vocoder,
    const Ort::MemoryInfo& mi,
    std::vector<int64_t>& audio_codes,
    int steps,
    int groups) {
    if (steps <= 0) return {};
    auto audio_codes_tensor = MakeTensorI64(mi, audio_codes, {1, steps, groups});
    const char* voc_in_names[] = {"audio_codes"};
    const char* voc_out_names[] = {"audio_values", "audio_lengths"};
    std::array<Ort::Value, 1> voc_inputs = {std::move(audio_codes_tensor)};
    auto voc_out = vocoder.Run(Ort::RunOptions{nullptr}, voc_in_names, voc_inputs.data(), 1, voc_out_names, 2);
    float* audio_ptr = voc_out[0].GetTensorMutableData<float>();
    int64_t* len_ptr = voc_out[1].GetTensorMutableData<int64_t>();
    const int64_t n = len_ptr[0];
    return std::vector<float>(audio_ptr, audio_ptr + n);
}

bool DecodeAudioCodesSafe(
    Ort::Session& vocoder,
    const Ort::MemoryInfo& mi,
    std::vector<int64_t>& audio_codes,
    int steps,
    int groups,
    std::vector<float>* out,
    std::string* error) {
    try {
        auto v = DecodeAudioCodes(vocoder, mi, audio_codes, steps, groups);
        if (out) *out = std::move(v);
        if (error) error->clear();
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    } catch (...) {
        if (error) *error = "unknown decode audio error";
        return false;
    }
}

std::string ReadAll(const std::string& path) {
    std::ifstream in(path);
    if (!in) return {};
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void WriteCodesTxt(const std::string& path, const std::vector<int64_t>& codes, int steps, int groups) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "WriteCodesTxt failed to open output codes: " << path << "\n";
        return;
    }
    for (int s = 0; s < steps; ++s) {
        for (int g = 0; g < groups; ++g) {
            if (g) out << ' ';
            out << codes[static_cast<size_t>(s * groups + g)];
        }
        out << '\n';
    }
}

bool WriteCodesTxtSafe(
    const std::string& path,
    const std::vector<int64_t>& codes,
    int steps,
    int groups,
    std::string* error) {
    std::ofstream probe(path, std::ios::app);
    if (!probe) {
        if (error) *error = "Failed to open output codes: " + path;
        return false;
    }
    probe.close();
    WriteCodesTxt(path, codes, steps, groups);
    if (error) error->clear();
    return true;
}

int TrimRepeatingTailFrames(std::vector<int64_t>* codes, int groups, int min_repeat, int keep_last) {
    if (min_repeat <= 0) return static_cast<int>(codes->size() / static_cast<size_t>(groups));
    const int steps = static_cast<int>(codes->size() / static_cast<size_t>(groups));
    if (steps <= 0) return 0;
    if (steps <= keep_last) return steps;
    // Keep a small minimum tail to avoid over-trimming into near-empty audio.
    constexpr int kMinStepsAfterTrim = 8;
    const int keep_effective = std::max(keep_last, kMinStepsAfterTrim);
    if (steps <= keep_effective) return steps;

    const int last = steps - 1;
    int run = 1;
    const int64_t last_first_code = (*codes)[static_cast<size_t>(last * groups)];
    for (int s = steps - 2; s >= 0; --s) {
        if ((*codes)[static_cast<size_t>(s * groups)] == last_first_code) {
            ++run;
        } else {
            break;
        }
    }
    if (run < min_repeat) return steps;
    const int removable = run - keep_effective;
    if (removable <= 0) return steps;
    const int new_steps = steps - removable;
    codes->resize(static_cast<size_t>(new_steps * groups));
    return new_steps;
}

std::vector<int64_t> ParseIntArray(const std::string& src, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    const size_t pos = src.find(marker);
    if (pos == std::string::npos) return {};
    const size_t lb = src.find('[', pos);
    const size_t rb = src.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos || rb <= lb) {
        return {};
    }
    std::vector<int64_t> out;
    int64_t cur = 0;
    bool in_num = false;
    bool neg = false;
    for (size_t i = lb + 1; i < rb; ++i) {
        const char c = src[i];
        if (c == '-') {
            neg = true;
            in_num = true;
            cur = 0;
        } else if (c >= '0' && c <= '9') {
            if (!in_num) {
                in_num = true;
                neg = false;
                cur = 0;
            }
            cur = cur * 10 + static_cast<int64_t>(c - '0');
        } else {
            if (in_num) {
                out.push_back(neg ? -cur : cur);
                in_num = false;
                neg = false;
                cur = 0;
            }
        }
    }
    if (in_num) out.push_back(neg ? -cur : cur);
    return out;
}

int64_t ParseIntScalar(const std::string& src, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    const size_t pos = src.find(marker);
    if (pos == std::string::npos) return 0;
    const size_t colon = src.find(':', pos);
    if (colon == std::string::npos) return 0;
    size_t i = colon + 1;
    while (i < src.size() && (src[i] == ' ' || src[i] == '\n' || src[i] == '\r' || src[i] == '\t')) ++i;
    bool neg = false;
    if (i < src.size() && src[i] == '-') {
        neg = true;
        ++i;
    }
    int64_t val = 0;
    bool has = false;
    while (i < src.size() && src[i] >= '0' && src[i] <= '9') {
        has = true;
        val = val * 10 + static_cast<int64_t>(src[i] - '0');
        ++i;
    }
    if (!has) return 0;
    return neg ? -val : val;
}

void AppendUtf8(uint32_t cp, std::string& out) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

uint32_t DecodeUtf8At(const std::string& s, size_t i, size_t* next_i) {
    if (!next_i) return 0;
    if (i >= s.size()) {
        *next_i = s.size();
        return 0;
    }
    const unsigned char c0 = static_cast<unsigned char>(s[i]);
    if (c0 < 0x80) {
        *next_i = i + 1;
        return c0;
    }
    if ((c0 >> 5) == 0x6 && i + 1 < s.size()) {
        const unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
        *next_i = i + 2;
        return ((c0 & 0x1F) << 6) | (c1 & 0x3F);
    }
    if ((c0 >> 4) == 0xE && i + 2 < s.size()) {
        const unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
        const unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
        *next_i = i + 3;
        return ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
    }
    if ((c0 >> 3) == 0x1E && i + 3 < s.size()) {
        const unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
        const unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
        const unsigned char c3 = static_cast<unsigned char>(s[i + 3]);
        *next_i = i + 4;
        return ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
    }
    *next_i = i + 1;
    return static_cast<uint32_t>('?');
}

bool IsAsciiLetter(uint32_t cp) {
    return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
}

bool IsAsciiDigit(uint32_t cp) {
    return (cp >= '0' && cp <= '9');
}

bool IsNewline(uint32_t cp) {
    return cp == '\n' || cp == '\r';
}

bool IsWhitespaceNonNewline(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\v' || cp == '\f';
}

bool IsLetter(uint32_t cp) {
    if (cp < 128) return IsAsciiLetter(cp);
    return true;  // Good approximation for Cyrillic/Unicode letters used in TTS texts.
}

bool IsNumber(uint32_t cp) {
    return IsAsciiDigit(cp);
}


} // namespace QWEN3TTSUTILS
