#include "tokenizer.h"
#include "utils.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace QWEN3TTS {

void VoiceTokenizer::SkipWs(const std::string& s, size_t* i) {
  while (*i < s.size() && std::isspace(static_cast<unsigned char>(s[*i]))) ++(*i);
}

bool VoiceTokenizer::ParseHex4(const std::string& s, size_t i, uint32_t* out) {
  if (!out) return false;
  if (i + 4 > s.size()) return false;
  uint32_t v = 0;
  for (size_t k = i; k < i + 4; ++k) {
    v <<= 4;
    const char c = s[k];
    if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
    else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(10 + c - 'a');
    else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(10 + c - 'A');
    else return false;
  }
  *out = v;
  return true;
}

bool VoiceTokenizer::ParseJsonString(const std::string& s, size_t* i, std::string* out) {
  if (!i || !out) return false;
  out->clear();
  if (*i >= s.size() || s[*i] != '"') return false;
  ++(*i);
  while (*i < s.size()) {
    const char c = s[*i];
    if (c == '"') {
      ++(*i);
      return true;
    }
    if (c != '\\') {
      out->push_back(c);
      ++(*i);
      continue;
    }
    ++(*i);
    if (*i >= s.size()) return false;
    const char e = s[*i];
    ++(*i);
    switch (e) {
      case '"': out->push_back('"'); break;
      case '\\': out->push_back('\\'); break;
      case '/': out->push_back('/'); break;
      case 'b': out->push_back('\b'); break;
      case 'f': out->push_back('\f'); break;
      case 'n': out->push_back('\n'); break;
      case 'r': out->push_back('\r'); break;
      case 't': out->push_back('\t'); break;
      case 'u': {
        uint32_t cp = 0;
        if (!ParseHex4(s, *i, &cp)) return false;
        *i += 4;
        QWEN3TTSUTILS::AppendUtf8(cp, *out);
        break;
      }
      default:
        return false;
    }
  }
  return false;
}

bool VoiceTokenizer::ParseJsonInt(const std::string& s, size_t* i, int64_t* out) {
  if (!i || !out) return false;
  SkipWs(s, i);
  bool neg = false;
  if (*i < s.size() && s[*i] == '-') {
    neg = true;
    ++(*i);
  }
  if (*i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[*i]))) {
    return false;
  }
  int64_t v = 0;
  while (*i < s.size() && std::isdigit(static_cast<unsigned char>(s[*i]))) {
    v = v * 10 + static_cast<int64_t>(s[*i] - '0');
    ++(*i);
  }
  *out = neg ? -v : v;
  return true;
}

int64_t VoiceTokenizer::FindAddedTokenId(const std::string& json, const std::string& content) {
  const std::string needle = "\"content\": \"" + content + "\"";
  const size_t pos = json.find(needle);
  if (pos == std::string::npos) return -1;
  const size_t obj_start = json.rfind('{', pos);
  if (obj_start == std::string::npos) return -1;
  const size_t q2 = json.rfind('"', obj_start);
  if (q2 == std::string::npos) return -1;
  const size_t q1 = json.rfind('"', q2 - 1);
  if (q1 == std::string::npos || q1 + 1 >= q2) return -1;
  return std::stoll(json.substr(q1 + 1, q2 - q1 - 1));
}

void VoiceTokenizer::InitByteEncoder() {
  std::vector<int> bs;
  for (int i = 33; i <= 126; ++i) bs.push_back(i);
  for (int i = 161; i <= 172; ++i) bs.push_back(i);
  for (int i = 174; i <= 255; ++i) bs.push_back(i);
  std::vector<int> cs = bs;
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n);
      ++n;
    }
  }
  for (size_t i = 0; i < bs.size(); ++i) {
    std::string ch;
    QWEN3TTSUTILS::AppendUtf8(static_cast<uint32_t>(cs[i]), ch);
    byte_encoder_[static_cast<size_t>(bs[i])] = ch;
  }
}

void VoiceTokenizer::Load(
    const std::string& tokenizer_dir,
    const std::string& vocab_file,
    const std::string& merges_file,
    const std::string& tokenizer_config_file) {
  last_error_.clear();
  InitByteEncoder();
  vocab_.clear();
  bpe_ranks_.clear();
  bpe_cache_.clear();

  const std::string vocab_json = QWEN3TTSUTILS::ReadAll((std::filesystem::path(tokenizer_dir) / vocab_file).string());
  if (vocab_json.empty()) {
    last_error_ = "Failed to read vocab json";
    return;
  }
  size_t i = 0;
  SkipWs(vocab_json, &i);
  if (i >= vocab_json.size() || vocab_json[i] != '{') {
    last_error_ = "vocab.json: expected object";
    return;
  }
  ++i;
  while (true) {
    SkipWs(vocab_json, &i);
    if (i < vocab_json.size() && vocab_json[i] == '}') {
      ++i;
      break;
    }
    std::string key;
    if (!ParseJsonString(vocab_json, &i, &key)) {
      last_error_ = "vocab.json: invalid key json string";
      return;
    }
    SkipWs(vocab_json, &i);
    if (i >= vocab_json.size() || vocab_json[i] != ':') {
      last_error_ = "vocab.json: expected ':'";
      return;
    }
    ++i;
    int64_t val = 0;
    if (!ParseJsonInt(vocab_json, &i, &val)) {
      last_error_ = "vocab.json: invalid int value";
      return;
    }
    vocab_.emplace(key, val);
    SkipWs(vocab_json, &i);
    if (i < vocab_json.size() && vocab_json[i] == ',') {
      ++i;
      continue;
    }
    if (i < vocab_json.size() && vocab_json[i] == '}') {
      ++i;
      break;
    }
  }

  std::ifstream merges((std::filesystem::path(tokenizer_dir) / merges_file).string());
  if (!merges) {
    last_error_ = "Failed to open merges.txt";
    return;
  }
  std::string line;
  int rank = 0;
  while (std::getline(merges, line)) {
    if (line.empty() || line[0] == '#') continue;
    size_t sp = line.find(' ');
    if (sp == std::string::npos) continue;
    const std::string a = line.substr(0, sp);
    const std::string b = line.substr(sp + 1);
    bpe_ranks_.emplace(a + "\t" + b, rank++);
  }

  const std::string tok_cfg = QWEN3TTSUTILS::ReadAll((std::filesystem::path(tokenizer_dir) / tokenizer_config_file).string());
  if (tok_cfg.empty()) {
    last_error_ = "Failed to read tokenizer_config.json";
    return;
  }
  im_start_id_ = FindAddedTokenId(tok_cfg, "<|im_start|>");
  im_end_id_ = FindAddedTokenId(tok_cfg, "<|im_end|>");
  endoftext_id_ = FindAddedTokenId(tok_cfg, "<|endoftext|>");

  auto it_ass = vocab_.find("assistant");
  auto it_user = vocab_.find("user");
  if (it_ass != vocab_.end()) assistant_id_ = it_ass->second;
  if (it_user != vocab_.end()) user_id_ = it_user->second;
  auto newline_ids = Encode("\n");
  if (newline_ids.empty()) {
    last_error_ = "Failed to encode newline token";
    return;
  }
  newline_id_ = newline_ids.at(0);

  if (im_start_id_ < 0 || im_end_id_ < 0 || assistant_id_ < 0 || user_id_ < 0 || newline_id_ < 0) {
    last_error_ = "Tokenizer special ids resolution failed";
    return;
  }
}

bool VoiceTokenizer::LoadSafe(
    const std::string& tokenizer_dir,
    const std::string& vocab_file,
    const std::string& merges_file,
    const std::string& tokenizer_config_file,
    std::string* error) {
  Load(tokenizer_dir, vocab_file, merges_file, tokenizer_config_file);
  if (!last_error_.empty()) {
    if (error) *error = last_error_;
    return false;
  }
  if (error) error->clear();
  return true;
}

std::vector<std::string> VoiceTokenizer::SplitUtf8Chars(const std::string& s) const {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    size_t j = i;
    QWEN3TTSUTILS::DecodeUtf8At(s, i, &j);
    out.push_back(s.substr(i, j - i));
    i = j;
  }
  return out;
}

std::string VoiceTokenizer::ByteEncodeToken(const std::string& tok) const {
  std::string out;
  out.reserve(tok.size() * 2);
  for (unsigned char b : tok) {
    out += byte_encoder_[b];
  }
  return out;
}

std::string VoiceTokenizer::Bpe(const std::string& token) {
  auto c = bpe_cache_.find(token);
  if (c != bpe_cache_.end()) return c->second;
  std::vector<std::string> word = SplitUtf8Chars(token);
  if (word.size() == 1) {
    bpe_cache_[token] = token;
    return token;
  }
  while (true) {
    int best_rank = std::numeric_limits<int>::max();
    int best_i = -1;
    for (int i = 0; i + 1 < static_cast<int>(word.size()); ++i) {
      auto it = bpe_ranks_.find(word[static_cast<size_t>(i)] + "\t" + word[static_cast<size_t>(i + 1)]);
      if (it != bpe_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
      }
    }
    if (best_i < 0) break;
    std::vector<std::string> nw;
    nw.reserve(word.size());
    for (int i = 0; i < static_cast<int>(word.size());) {
      if (i == best_i) {
        nw.push_back(word[static_cast<size_t>(i)] + word[static_cast<size_t>(i + 1)]);
        i += 2;
      } else {
        nw.push_back(word[static_cast<size_t>(i)]);
        ++i;
      }
    }
    word.swap(nw);
    if (word.size() == 1) break;
  }
  std::string out;
  for (size_t i = 0; i < word.size(); ++i) {
    if (i) out.push_back(' ');
    out += word[i];
  }
  bpe_cache_[token] = out;
  return out;
}

std::vector<std::string> VoiceTokenizer::RegexLikeSplit(const std::string& s) const {
  std::vector<std::string> out;
  size_t i = 0;
  auto contraction_len = [&](size_t pos) -> size_t {
    static const char* cs[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    for (const char* c : cs) {
      const size_t n = std::strlen(c);
      if (pos + n > s.size()) continue;
      bool ok = true;
      for (size_t k = 0; k < n; ++k) {
        const char a = static_cast<char>(std::tolower(static_cast<unsigned char>(s[pos + k])));
        if (a != c[k]) { ok = false; break; }
      }
      if (ok) return n;
    }
    return 0;
  };
  while (i < s.size()) {
    const size_t c_len = contraction_len(i);
    if (c_len > 0) {
      out.push_back(s.substr(i, c_len));
      i += c_len;
      continue;
    }
    size_t j = i;
    const uint32_t cp = QWEN3TTSUTILS::DecodeUtf8At(s, i, &j);
    const bool is_l = QWEN3TTSUTILS::IsLetter(cp);
    const bool is_n = QWEN3TTSUTILS::IsNumber(cp);
    const bool is_nl = QWEN3TTSUTILS::IsNewline(cp);
    const bool is_ws = QWEN3TTSUTILS::IsWhitespaceNonNewline(cp);

    if (!is_nl && !is_l && !is_n) {
      size_t j2 = j;
      if (j2 < s.size()) {
        size_t k = j2;
        const uint32_t cp2 = QWEN3TTSUTILS::DecodeUtf8At(s, j2, &k);
        if (QWEN3TTSUTILS::IsLetter(cp2)) {
          size_t e = k;
          while (e < s.size()) {
            size_t npos = e;
            const uint32_t cpx = QWEN3TTSUTILS::DecodeUtf8At(s, e, &npos);
            if (!QWEN3TTSUTILS::IsLetter(cpx)) break;
            e = npos;
          }
          out.push_back(s.substr(i, e - i));
          i = e;
          continue;
        }
      }
    }

    if (is_l) {
      size_t e = j;
      while (e < s.size()) {
        size_t npos = e;
        const uint32_t cpx = QWEN3TTSUTILS::DecodeUtf8At(s, e, &npos);
        if (!QWEN3TTSUTILS::IsLetter(cpx)) break;
        e = npos;
      }
      out.push_back(s.substr(i, e - i));
      i = e;
      continue;
    }

    if (is_n) {
      out.push_back(s.substr(i, j - i));
      i = j;
      continue;
    }

    if (is_ws || is_nl) {
      if (is_ws) {
        size_t e = j;
        while (e < s.size()) {
          size_t npos = e;
          const uint32_t cpx = QWEN3TTSUTILS::DecodeUtf8At(s, e, &npos);
          if (!QWEN3TTSUTILS::IsWhitespaceNonNewline(cpx)) break;
          e = npos;
        }
        if (e < s.size()) {
          size_t npos = e;
          const uint32_t cpx = QWEN3TTSUTILS::DecodeUtf8At(s, e, &npos);
          if (QWEN3TTSUTILS::IsNewline(cpx)) {
            size_t nle = npos;
            while (nle < s.size()) {
              size_t np2 = nle;
              const uint32_t cy = QWEN3TTSUTILS::DecodeUtf8At(s, nle, &np2);
              if (!QWEN3TTSUTILS::IsNewline(cy)) break;
              nle = np2;
            }
            out.push_back(s.substr(i, nle - i));
            i = nle;
            continue;
          }
          if (QWEN3TTSUTILS::IsLetter(cpx) ||
              (!QWEN3TTSUTILS::IsNewline(cpx) &&
               !QWEN3TTSUTILS::IsLetter(cpx) &&
               !QWEN3TTSUTILS::IsNumber(cpx) &&
               !QWEN3TTSUTILS::IsWhitespaceNonNewline(cpx))) {
            if (e - i > 1) {
              out.push_back(s.substr(i, (e - i) - 1));
              i = e - 1;
              continue;
            }
          }
        }
        out.push_back(s.substr(i, e - i));
        i = e;
        continue;
      }
      size_t e = j;
      while (e < s.size()) {
        size_t npos = e;
        const uint32_t cpx = QWEN3TTSUTILS::DecodeUtf8At(s, e, &npos);
        if (!QWEN3TTSUTILS::IsNewline(cpx)) break;
        e = npos;
      }
      out.push_back(s.substr(i, e - i));
      i = e;
      continue;
    }

    size_t e = j;
    while (e < s.size()) {
      size_t npos = e;
      const uint32_t cpx = QWEN3TTSUTILS::DecodeUtf8At(s, e, &npos);
      if (QWEN3TTSUTILS::IsNewline(cpx) ||
          QWEN3TTSUTILS::IsWhitespaceNonNewline(cpx) ||
          QWEN3TTSUTILS::IsLetter(cpx) ||
          QWEN3TTSUTILS::IsNumber(cpx)) break;
      e = npos;
    }
    while (e < s.size()) {
      size_t npos = e;
      const uint32_t cpx = QWEN3TTSUTILS::DecodeUtf8At(s, e, &npos);
      if (!QWEN3TTSUTILS::IsNewline(cpx)) break;
      e = npos;
    }
    out.push_back(s.substr(i, e - i));
    i = e;
  }
  return out;
}

std::vector<int64_t> VoiceTokenizer::Encode(const std::string& text) {
  std::vector<int64_t> ids;
  const auto chunks = RegexLikeSplit(text);
  for (const auto& ch : chunks) {
    const std::string be = ByteEncodeToken(ch);
    const std::string bpe = Bpe(be);
    size_t i = 0;
    while (i < bpe.size()) {
      size_t j = bpe.find(' ', i);
      if (j == std::string::npos) j = bpe.size();
      const std::string tok = bpe.substr(i, j - i);
      auto it = vocab_.find(tok);
      if (it != vocab_.end()) {
        ids.push_back(it->second);
      } else if (endoftext_id_ >= 0) {
        ids.push_back(endoftext_id_);
      } else {
        last_error_ = "Tokenizer OOV and no unk token id";
        return {};
      }
      i = j + 1;
    }
  }
  return ids;
}

void VoiceTokenizer::BuildVoiceDesignIds(
    const std::string& text,
    const std::string& instruct,
    std::vector<int64_t>* input_ids,
    std::vector<int64_t>* instruct_ids) {
  auto text_ids = Encode(text);
  auto instr_ids = Encode(instruct);
  input_ids->clear();
  instruct_ids->clear();
  *input_ids = {im_start_id_, assistant_id_, newline_id_};
  input_ids->insert(input_ids->end(), text_ids.begin(), text_ids.end());
  input_ids->push_back(im_end_id_);
  input_ids->push_back(newline_id_);
  input_ids->push_back(im_start_id_);
  input_ids->push_back(assistant_id_);
  input_ids->push_back(newline_id_);

  *instruct_ids = {im_start_id_, user_id_, newline_id_};
  instruct_ids->insert(instruct_ids->end(), instr_ids.begin(), instr_ids.end());
  instruct_ids->push_back(im_end_id_);
  instruct_ids->push_back(newline_id_);
}

bool VoiceTokenizer::BuildVoiceDesignIdsSafe(
    const std::string& text,
    const std::string& instruct,
    std::vector<int64_t>* input_ids,
    std::vector<int64_t>* instruct_ids,
    std::string* error) {
  BuildVoiceDesignIds(text, instruct, input_ids, instruct_ids);
  if (!last_error_.empty()) {
    if (error) *error = last_error_;
    return false;
  }
  if (error) error->clear();
  return true;
}

}  // namespace QWEN3TTS
