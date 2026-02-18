#pragma once
#if __has_include(<onnxruntime_cxx_api.h>)
#include <onnxruntime_cxx_api.h>
#elif __has_include(<onnxruntime/onnxruntime_cxx_api.h>)
#include <onnxruntime/onnxruntime_cxx_api.h>
#else
#error "onnxruntime_cxx_api.h not found. Set include path to ONNX Runtime headers."
#endif
#include <cstdint>
// #include <memory>
#include <random>
#include <string>
#include <vector>


namespace QWEN3TTSUTILS {



GraphOptimizationLevel ParseGraphOptimizationLevel(const std::string& s);
std::string ReadAll(const std::string& path);
std::vector<int64_t> ParseIntArray(const std::string& src, const std::string& key);
int64_t ParseIntScalar(const std::string& src, const std::string& key);


int64_t Argmax(const float* data, int64_t size);

int64_t ArgmaxTalkerFirstCode(const float* data, int64_t talker_vocab, int64_t cp_vocab, int64_t codec_eos_id, bool allow_eos = true);

int64_t SampleFromCandidates(const std::vector<std::pair<float, int64_t>>& candidates, float temperature, int top_k, std::mt19937_64* rng);
int64_t SelectTalkerFirstCode(const float* data, int64_t talker_vocab,int64_t cp_vocab, int64_t codec_eos_id, bool allow_eos, bool do_sample, float temperature, int top_k, std::mt19937_64* rng);

int64_t SelectCpCode(const float* data, int64_t cp_vocab, bool do_sample, float temperature, int top_k, std::mt19937_64* rng);

void WriteWavPcm16(const std::string& path, const std::vector<float>& samples, int sample_rate);
bool WriteWavPcm16Safe(const std::string& path, const std::vector<float>& samples, int sample_rate, std::string* error);
Ort::Value MakeTensorI64(const Ort::MemoryInfo& mi, std::vector<int64_t>& data, const std::vector<int64_t>& shape);

Ort::Value MakeTensorF32(const Ort::MemoryInfo& mi, std::vector<float>& data, const std::vector<int64_t>& shape);

GraphOptimizationLevel ParseGraphOptimizationLevel(const std::string& s);

bool HasExecutionProvider(const std::string &ep_name);

std::vector<float> DecodeAudioCodes(Ort::Session& vocoder, const Ort::MemoryInfo& mi, std::vector<int64_t>& audio_codes, int steps, int groups);
bool DecodeAudioCodesSafe(
    Ort::Session& vocoder,
    const Ort::MemoryInfo& mi,
    std::vector<int64_t>& audio_codes,
    int steps,
    int groups,
    std::vector<float>* out,
    std::string* error);

std::string ReadAll(const std::string& path);

void WriteCodesTxt(const std::string& path, const std::vector<int64_t>& codes, int steps, int groups);
bool WriteCodesTxtSafe(const std::string& path, const std::vector<int64_t>& codes, int steps, int groups, std::string* error);

int TrimRepeatingTailFrames(std::vector<int64_t>* codes, int groups, int min_repeat, int keep_last);

std::vector<int64_t> ParseIntArray(const std::string& src, const std::string& key);

int64_t ParseIntScalar(const std::string& src, const std::string& key);

void AppendUtf8(uint32_t cp, std::string& out);

uint32_t DecodeUtf8At(const std::string& s, size_t i, size_t* next_i);

bool IsAsciiLetter(uint32_t cp);

bool IsAsciiDigit(uint32_t cp);

bool IsNewline(uint32_t cp);

bool IsWhitespaceNonNewline(uint32_t cp);

bool IsLetter(uint32_t cp);

bool IsNumber(uint32_t cp);




} // namespace QWEN3TTS
