#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace QWEN3TTS {

class VoiceTokenizer {
 public:
  void Load(
      const std::string& tokenizer_dir,
      const std::string& vocab_file = "vocab.json",
      const std::string& merges_file = "merges.txt",
      const std::string& tokenizer_config_file = "tokenizer_config.json");
  bool LoadSafe(
      const std::string& tokenizer_dir,
      const std::string& vocab_file,
      const std::string& merges_file,
      const std::string& tokenizer_config_file,
      std::string* error);

  void BuildVoiceDesignIds(
      const std::string& text,
      const std::string& instruct,
      std::vector<int64_t>* input_ids,
      std::vector<int64_t>* instruct_ids);
  bool BuildVoiceDesignIdsSafe(
      const std::string& text,
      const std::string& instruct,
      std::vector<int64_t>* input_ids,
      std::vector<int64_t>* instruct_ids,
      std::string* error);

 private:
  static void SkipWs(const std::string& s, size_t* i);
  static bool ParseHex4(const std::string& s, size_t i, uint32_t* out);
  static bool ParseJsonString(const std::string& s, size_t* i, std::string* out);
  static bool ParseJsonInt(const std::string& s, size_t* i, int64_t* out);
  static int64_t FindAddedTokenId(const std::string& json, const std::string& content);

  void InitByteEncoder();
  std::vector<std::string> SplitUtf8Chars(const std::string& s) const;
  std::string ByteEncodeToken(const std::string& tok) const;
  std::string Bpe(const std::string& token);
  std::vector<std::string> RegexLikeSplit(const std::string& s) const;
  std::vector<int64_t> Encode(const std::string& text);

 private:
  std::unordered_map<std::string, int64_t> vocab_;
  std::unordered_map<std::string, int> bpe_ranks_;
  std::unordered_map<std::string, std::string> bpe_cache_;
  std::array<std::string, 256> byte_encoder_;
  int64_t im_start_id_ = -1;
  int64_t im_end_id_ = -1;
  int64_t endoftext_id_ = -1;
  int64_t assistant_id_ = -1;
  int64_t user_id_ = -1;
  int64_t newline_id_ = -1;
  std::string last_error_;
};

}  // namespace QWEN3TTS
