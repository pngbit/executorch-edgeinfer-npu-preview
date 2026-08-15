/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama3.2 runner that includes preprocessing and post processing
// logic. The module takes in a string as input and emits a string as output.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#if defined(EXECUTORCH_BUILD_EDGEINFER)
#include <unordered_set>
#include <vector>
#endif

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/attention_sink_rope_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/cache_utils.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.h>
#if defined(EXECUTORCH_BUILD_EDGEINFER)
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/edgeinfer_prompt_processor.h>
#endif
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/prompt_processor.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/token_generator.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace example {

enum DecoderModelVersion {
  kLlama2 = 0,
  kLlama3,
  kGemma,
  kGemma3,
  kGranite,
  kPhi4,
  kQwen2_5,
  kQwen3,
  kSmollm2_135m,
  kSmollm3,
  kCodegen,
  kGlm,
  kGemma2,
};

class Runner : public executorch::extension::llm::IRunner {
 public:
  enum EvalMode {
    kKVCached = 0,
    kHybrid,
    kLookaheadDecoding,
    kUnsupported,
  };

  explicit Runner(
      std::unique_ptr<executorch::extension::Module> module,
      const std::string& decoder_model,
      const std::string& model_path,
      const std::string& tokenizer_path,
      const std::string& performance_output_path,
      const std::string& dump_logits_path,
      const float temperature = 0.8f,
      const int eval_mode = EvalMode::kHybrid,
      const bool shared_buffer = false,
      const int ngram = 0,
      const int window = 0,
      const int gcap = 0,
      std::unique_ptr<tokenizers::Tokenizer> tokenizer = nullptr,
      std::unique_ptr<executorch::extension::Module>
          attention_sink_rope_module = nullptr
#if defined(EXECUTORCH_BUILD_EDGEINFER)
      ,
      std::unique_ptr<executorch::extension::Module>
          edgeinfer_composable_module = nullptr,
      std::vector<std::unique_ptr<executorch::extension::Module>>
          edgeinfer_composable_layer_modules = {},
      std::unique_ptr<executorch::extension::Module>
          edgeinfer_decode_composable_module = nullptr,
      std::vector<std::unique_ptr<executorch::extension::Module>>
          edgeinfer_decode_composable_layer_modules = {},
      EdgeInferPrefillOptions edgeinfer_prefill_options = {}
#endif
  );

  bool is_loaded() const override;
  executorch::runtime::Error load() override;
  // TODO: Support echo and warming
  executorch::runtime::Error generate(
      const std::string& prompt,
      const executorch::extension::llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const executorch::llm::Stats&)> stats_callback = {})
      override;

  executorch::runtime::Error generate_from_prompt_or_file(
      const std::string& prompt,
      bool tokenized_prompt,
      const executorch::extension::llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const executorch::llm::Stats&)> stats_callback = {});
  void stop() override {};
  void reset() override {};
  executorch::runtime::Result<DecoderModelVersion> get_decoder_model_version();

 private:
  std::unique_ptr<executorch::extension::Module> module_;
  std::unique_ptr<executorch::extension::Module> attention_sink_rope_module_;
#if defined(EXECUTORCH_BUILD_EDGEINFER)
  std::unique_ptr<executorch::extension::Module> edgeinfer_composable_module_;
  std::vector<std::unique_ptr<executorch::extension::Module>>
      edgeinfer_composable_layer_modules_;
  std::unique_ptr<executorch::extension::Module>
      edgeinfer_decode_composable_module_;
  std::vector<std::unique_ptr<executorch::extension::Module>>
      edgeinfer_decode_composable_layer_modules_;
  EdgeInferPrefillOptions edgeinfer_prefill_options_;
#endif
  int32_t context_len_{0};

  int ngram_{0};
  int window_{0};
  int gcap_{0};

  // Defaults to StaticCahce, indicating that the model does not use a
  // global/local architecture.
  CacheMode cache_mode_{CacheMode::StaticCahce};
  int64_t cur_pos_{0};

  std::string tokenizer_path_;
  std::string performance_output_path_;
  std::string dump_logits_path_;
  float temperature_;
  EvalMode eval_mode_;
  bool shared_buffer_;

  DecoderModelVersion decoder_model_version_;
  std::unique_ptr<IMemAlloc> buffer_manager_;
  std::unique_ptr<KVManager> kv_manager_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<DecoderRunner> decoder_runner_;
  std::unique_ptr<AttentionSinkRopeRunner> attention_sink_rope_runner_;
  std::unique_ptr<PromptProcessor> prompt_processor_;
#if defined(EXECUTORCH_BUILD_EDGEINFER)
  EdgeInferPromptProcessor* edgeinfer_prompt_processor_{nullptr};
#endif
  std::unique_ptr<TokenGenerator> token_generator_;
#if defined(EXECUTORCH_BUILD_EDGEINFER)
  std::unordered_set<uint64_t> edgeinfer_eos_ids_;
#endif

  // stats
  executorch::llm::Stats stats_;
};

} // namespace example
