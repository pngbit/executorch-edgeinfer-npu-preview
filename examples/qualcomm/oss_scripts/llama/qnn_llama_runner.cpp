/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 *
 * This tool can run Llama2 110M, Llama3.2 1B / 3B, Gemma 2B, Gemma2 2B, Gemma3
 * 1B, Granite3.3 2B, phi4-mini-instruct, Qwen2.5 0.5B / 1.5B, Qwen3 0.6B
 * / 1.7B, SmolLM2 135M, SmolLM3 3B with Qualcomm AI Engine Direct.
 *
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/runtime/platform/log.h>
#include <gflags/gflags.h>
#if defined(EXECUTORCH_BUILD_EDGEINFER)
#include <algorithm>
#include <cstdlib>
#endif
#include <fstream>
#if defined(EXECUTORCH_BUILD_EDGEINFER)
#include <limits>
#include <sstream>
#endif
#include <vector>

DEFINE_string(decoder_model_version, "llama2", "The decoder model to execute.");
DEFINE_string(
    model_path,
    "kv_llama_qnn.pte",
    "Model serialized in flatbuffer format.");
DEFINE_string(
    attention_sink_rope_path,
    "",
    "[Attention Sink] The Attention Sink Rope Model is serialized using the flatbuffer format. If specified, seq_len can exceed the context length defined in the model.");
DEFINE_string(
    output_path,
    "outputs.txt",
    "Executorch inference data output path.");
DEFINE_string(
    performance_output_path,
    "inference_speed.txt",
    "Records inference speed. For CI purpose.");
DEFINE_string(
    dump_logits_path,
    "",
    "If path is provided, program will dump all logits generated. This option is for analysis purpose. It is not recommended for general usage as it will cause token rate drop and increase in memory usage.");
DEFINE_string(tokenizer_path, "tokenizer.bin", "Tokenizer stuff.");
DEFINE_string(
    prompt,
    "The answer to the ultimate question is",
    "User prompts for Llama. When multiple prompts are entered, a multi-turn conversation will be initiated. Note that this feature is currently for testing purposes only.");
DEFINE_string(
    tokenized_prompt,
    "",
    "This is an alternative of passing prompts. Users could provide this in a raw file, with tokens saved in uint64 format.");
DEFINE_string(
    system_prompt,
    "",
    "Tells the model what kind of assistant it should be. For example, You are a helpful AI assistant for travel tips and recommendations. Default is None");
DEFINE_double(
    temperature,
    0.0f,
    "Temperature; Default is 0.0f. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");
DEFINE_int32(
    seq_len,
    128,
    "Total number of tokens to generate (prompt + output).");
DEFINE_int32(
    eval_mode,
    1,
    "0: TokenGenerator(kv) / 1: HybridMode (prefill+kv) / 2: Lookahead Decoding");
DEFINE_bool(
    shared_buffer,
    false,
    "Specifies to use shared buffers for zero-copy use case between the application and device/co-processor associated with the backend.");
DEFINE_int32(num_iters, 1, "total num of iterations to run.");
DEFINE_int32(
    ngram,
    0,
    "[Lookahead Decoding] Represents the size of the n-grams used in the lookahead process.");
DEFINE_int32(
    window,
    0,
    "[Lookahead Decoding] Determines how many future tokens the algorithm attempts to predict in each step.");
DEFINE_int32(
    gcap,
    0,
    "[Lookahead Decoding] Represents the maximum number of speculations or candidate n-grams that the algorithm considers in each step for verification. It balances the trade-off between computation efficiency and exploring more possibilities.");
#if defined(EXECUTORCH_BUILD_EDGEINFER)
DEFINE_string(
    edgeinfer_composable_model_path,
    "",
    "Optional split-PTE path for EdgeInfer Prefill. Empty preserves the "
    "original decoder-PTE PromptProcessor.");
DEFINE_string(
    edgeinfer_decode_model_path,
    "",
    "Optional R=1 split-PTE path for EdgeInfer Decode. Empty reuses the "
    "Prefill split PTE.");
DEFINE_string(
    edgeinfer_layer_model_paths,
    "",
    "Optional comma-separated EdgeInfer Prefill layer-shard PTE paths in "
    "ascending layer order.");
DEFINE_uint32(
    edgeinfer_layers_per_shard,
    0,
    "Number of contiguous Transformer layers in each EdgeInfer Prefill "
    "layer-shard PTE. Zero selects the single-PTE layout.");
DEFINE_string(
    edgeinfer_decode_layer_model_paths,
    "",
    "Optional comma-separated EdgeInfer Decode layer-shard PTE paths in "
    "ascending layer order.");
DEFINE_uint32(
    edgeinfer_decode_layers_per_shard,
    0,
    "Number of contiguous Transformer layers in each EdgeInfer Decode "
    "layer-shard PTE. Zero selects the single-PTE layout.");
DEFINE_string(
    edgeinfer_widths,
    "",
    "Comma-separated power-of-two K/V widths exported in the EdgeInfer "
    "split PTE.");
DEFINE_string(
    edgeinfer_decode_widths,
    "",
    "Comma-separated power-of-two K/V widths exported in the optional "
    "EdgeInfer Decode PTE. Empty reuses --edgeinfer_widths.");
DEFINE_uint32(edgeinfer_layers, 0, "Number of layers in the split PTE.");
DEFINE_uint32(edgeinfer_dim, 0, "Hidden dimension in the split PTE.");
DEFINE_uint32(edgeinfer_query_heads, 0, "Query heads in the split PTE.");
DEFINE_uint32(edgeinfer_kv_heads, 0, "K/V heads in the split PTE.");
DEFINE_uint32(edgeinfer_head_dim, 0, "Head dimension in the split PTE.");
DEFINE_uint32(
    edgeinfer_vocab_size,
    0,
    "Split-PTE vocabulary size. Zero derives the value from the decoder PTE.");
DEFINE_double(edgeinfer_rope_theta, 10000.0, "RoPE theta for the split PTE.");
DEFINE_string(
    edgeinfer_rope_style,
    "llama",
    "RoPE layout for the split PTE: llama or hf.");
DEFINE_bool(
    edgeinfer_pre_attention_rope,
    false,
    "Use the opt-in pre(hidden, cos, sin)->Q,K,V split-PTE ABI. This keeps "
    "Q/K normalization, layout conversion, and RoPE inside each layer graph.");
DEFINE_uint32(
    edgeinfer_profile_warmup,
    1,
    "Warm-up calls per EdgeInfer static Attention method.");
DEFINE_uint32(
    edgeinfer_profile_iterations,
    3,
    "Timed calls per EdgeInfer static Attention method.");

namespace {

std::vector<size_t> ParseEdgeInferWidths(const std::string& values) {
  std::vector<size_t> result;
  std::stringstream stream(values);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      return {};
    }
    char* end = nullptr;
    const unsigned long long value = std::strtoull(token.c_str(), &end, 10);
    if (end == token.c_str() || *end != '\0' || value == 0 ||
        value > std::numeric_limits<size_t>::max()) {
      return {};
    }
    const size_t width = static_cast<size_t>(value);
    if ((width & (width - 1)) != 0) {
      return {};
    }
    result.push_back(width);
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

std::vector<std::string> ParseEdgeInferPaths(const std::string& values) {
  std::vector<std::string> result;
  if (values.empty()) {
    return result;
  }
  std::stringstream stream(values);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      return {};
    }
    result.emplace_back(std::move(token));
  }
  return result;
}

std::vector<std::unique_ptr<executorch::extension::Module>> MakeModules(
    const std::vector<std::string>& paths) {
  std::vector<std::unique_ptr<executorch::extension::Module>> modules;
  modules.reserve(paths.size());
  for (const std::string& path : paths) {
    modules.emplace_back(std::make_unique<executorch::extension::Module>(
        path.c_str(),
        executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors));
  }
  return modules;
}

} // namespace
#endif

std::vector<std::string> CollectPrompts(int argc, char** argv) {
  // Collect all prompts from command line, example usage:
  // --prompt "prompt1" --prompt "prompt2" --prompt "prompt3"
  std::vector<std::string> prompts;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--prompt" && i + 1 < argc) {
      prompts.push_back(argv[i + 1]);
      i++; // Skip the next argument
    }
  }
  return prompts;
}

std::string get_formatted_prompt(
    const std::string& prompt,
    const std::string& system_prompt,
    example::DecoderModelVersion decoder_model_version) {
  std::string formatted_prompt;
  switch (decoder_model_version) {
    case example::DecoderModelVersion::kLlama2:
    case example::DecoderModelVersion::kQwen2_5:
    case example::DecoderModelVersion::kCodegen:
      formatted_prompt.append(prompt);
      break;
    case example::DecoderModelVersion::kLlama3:
      if (!system_prompt.empty()) {
        formatted_prompt.append(
            "<|start_header_id|>system<|end_header_id|>\n\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|eot_id|>");
      }
      formatted_prompt.append("<|start_header_id|>user<|end_header_id|>\n\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append(
          "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
      break;
    case example::DecoderModelVersion::kGemma:
    case example::DecoderModelVersion::kGemma3:
      formatted_prompt.append("<start_of_turn>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<end_of_turn>\n");
      formatted_prompt.append("<start_of_turn>model\n");
      if (!system_prompt.empty()) {
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<end_of_turn>\n");
      }
      break;
    case example::DecoderModelVersion::kGemma2:
      formatted_prompt.append("<start_of_turn>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<end_of_turn>\n");
      formatted_prompt.append("<start_of_turn>model\n");
      break;
    case example::DecoderModelVersion::kGranite:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|start_of_role|>system<|end_of_role|>");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|end_of_text|>\n");
      }
      formatted_prompt.append("<|start_of_role|>user<|end_of_role|>");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|end_of_text|>\n");
      formatted_prompt.append("<|start_of_role|>assistant<|end_of_role|>");
      break;
    case example::DecoderModelVersion::kPhi4:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|system|>");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|end|>");
      }
      formatted_prompt.append("<|user|>");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|end|><|assistant|>");
      break;
    case example::DecoderModelVersion::kQwen3:
      formatted_prompt.append("<|im_start|>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|im_end|>\n");
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|im_end|>\n");
      }
      formatted_prompt.append("<|im_start|>assistant");
      break;
    case example::DecoderModelVersion::kSmollm2_135m:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("<|im_end|>\n");
      }
      formatted_prompt.append("<|im_start|>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|im_end|>\n");
      formatted_prompt.append("<|im_start|>assistant\n\n");
      break;
    case example::DecoderModelVersion::kSmollm3:
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|im_start|>system\n");
        formatted_prompt.append(system_prompt);
        formatted_prompt.append("\n\n");
      }
      formatted_prompt.append("<|im_start|>user\n");
      formatted_prompt.append(prompt);
      formatted_prompt.append("<|im_end|>\n");
      formatted_prompt.append("<|im_start|>assistant\n");
      break;
    case example::DecoderModelVersion::kGlm:
      formatted_prompt.append("<|user|>\n");
      formatted_prompt.append(prompt);
      if (!system_prompt.empty()) {
        formatted_prompt.append("<|system|>\n");
        formatted_prompt.append(system_prompt);
      }
      formatted_prompt.append("<|assistant|>\n");
      break;
    default:
      ET_CHECK_MSG(false, "unsupported llama version");
      break;
  }
  return formatted_prompt;
}

void start_runner(
    std::unique_ptr<executorch::extension::Module> module,
    std::vector<std::string>& prompts,
    std::unique_ptr<executorch::extension::Module> attention_sink_rope_module
#if defined(EXECUTORCH_BUILD_EDGEINFER)
    ,
    std::unique_ptr<executorch::extension::Module> edgeinfer_composable_module,
    std::vector<std::unique_ptr<executorch::extension::Module>>
        edgeinfer_composable_layer_modules,
    std::unique_ptr<executorch::extension::Module>
        edgeinfer_decode_composable_module,
    std::vector<std::unique_ptr<executorch::extension::Module>>
        edgeinfer_decode_composable_layer_modules
#endif
) {
  bool use_tokenized_prompt =
      gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default ? false
                                                                         : true;
#if defined(EXECUTORCH_BUILD_EDGEINFER)
  example::EdgeInferPrefillOptions edgeinfer_options;
  if (edgeinfer_composable_module != nullptr) {
    edgeinfer_options.widths = ParseEdgeInferWidths(FLAGS_edgeinfer_widths);
    edgeinfer_options.decode_widths = FLAGS_edgeinfer_decode_widths.empty()
        ? edgeinfer_options.widths
        : ParseEdgeInferWidths(FLAGS_edgeinfer_decode_widths);
    ET_CHECK_MSG(
        !edgeinfer_options.widths.empty() &&
            !edgeinfer_options.decode_widths.empty() &&
            FLAGS_edgeinfer_dim > 0 && FLAGS_edgeinfer_query_heads > 0 &&
            FLAGS_edgeinfer_kv_heads > 0 && FLAGS_edgeinfer_head_dim > 0 &&
            FLAGS_edgeinfer_rope_theta > 0.0 &&
            (FLAGS_edgeinfer_rope_style == "llama" ||
             FLAGS_edgeinfer_rope_style == "hf"),
        "EdgeInfer split PTE requires widths, dimensions, and a valid RoPE layout.");
    edgeinfer_options.layers = FLAGS_edgeinfer_layers;
    edgeinfer_options.dim = FLAGS_edgeinfer_dim;
    edgeinfer_options.query_heads = FLAGS_edgeinfer_query_heads;
    edgeinfer_options.kv_heads = FLAGS_edgeinfer_kv_heads;
    edgeinfer_options.head_dim = FLAGS_edgeinfer_head_dim;
    edgeinfer_options.vocab_size = FLAGS_edgeinfer_vocab_size;
    edgeinfer_options.rope_theta = FLAGS_edgeinfer_rope_theta;
    edgeinfer_options.hf_rope = FLAGS_edgeinfer_rope_style == "hf";
    edgeinfer_options.pre_attention_rope = FLAGS_edgeinfer_pre_attention_rope;
    edgeinfer_options.layers_per_shard = FLAGS_edgeinfer_layers_per_shard;
    edgeinfer_options.decode_layers_per_shard =
        FLAGS_edgeinfer_decode_layers_per_shard;
    edgeinfer_options.profile_warmup = FLAGS_edgeinfer_profile_warmup;
    edgeinfer_options.profile_iterations = FLAGS_edgeinfer_profile_iterations;
  }

  // create llama runner
  example::Runner runner(
      std::move(module),
      FLAGS_decoder_model_version.c_str(),
      FLAGS_model_path.c_str(),
      FLAGS_tokenizer_path.c_str(),
      FLAGS_dump_logits_path.c_str(),
      FLAGS_performance_output_path.c_str(),
      FLAGS_temperature,
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      FLAGS_ngram,
      FLAGS_window,
      FLAGS_gcap,
      nullptr,
      std::move(attention_sink_rope_module),
      std::move(edgeinfer_composable_module),
      std::move(edgeinfer_composable_layer_modules),
      std::move(edgeinfer_decode_composable_module),
      std::move(edgeinfer_decode_composable_layer_modules),
      std::move(edgeinfer_options));
#else
  example::Runner runner(
      std::move(module),
      FLAGS_decoder_model_version.c_str(),
      FLAGS_model_path.c_str(),
      FLAGS_tokenizer_path.c_str(),
      FLAGS_dump_logits_path.c_str(),
      FLAGS_performance_output_path.c_str(),
      FLAGS_temperature,
      FLAGS_eval_mode,
      FLAGS_shared_buffer,
      FLAGS_ngram,
      FLAGS_window,
      FLAGS_gcap,
      nullptr,
      std::move(attention_sink_rope_module));
#endif
  auto decoder_model_version = runner.get_decoder_model_version();
  std::vector<char> buf;
  buf.reserve(5 * FLAGS_seq_len); // assume each token is around 5 char
  std::ofstream fout(FLAGS_output_path.c_str());
  auto callback = [&](const std::string& piece) {
    for (const char c : piece) {
      buf.push_back(c);
    }
  };
  executorch::extension::llm::GenerationConfig config{
      true,
      "",
      "",
      false,
      -1,
      false,
      FLAGS_seq_len,
      static_cast<float>(FLAGS_temperature),
      0,
      0};
  if (use_tokenized_prompt) {
    runner.generate_from_prompt_or_file(
        FLAGS_tokenized_prompt.c_str(), use_tokenized_prompt, config, callback);
  } else {
    // generate tokens & store inference output
    for (int i = 0; i < FLAGS_num_iters; i++) {
      for (const auto& prompt : prompts) {
        std::string formatted_prompt;
        formatted_prompt = get_formatted_prompt(
            prompt, FLAGS_system_prompt, decoder_model_version.get());
        runner.generate_from_prompt_or_file(
            formatted_prompt.c_str(), use_tokenized_prompt, config, callback);
      }
    }
  }

  fout.write(buf.data(), buf.size());
  fout.close();
}

int main(int argc, char** argv) {
  std::vector<std::string> prompts = CollectPrompts(argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (!gflags::GetCommandLineFlagInfoOrDie("prompt").is_default &&
      !gflags::GetCommandLineFlagInfoOrDie("tokenized_prompt").is_default) {
    ET_CHECK_MSG(false, "Only provide prompt or tokenized_input but not both.");
  }
  if (!gflags::GetCommandLineFlagInfoOrDie("dump_logits_path").is_default &&
      FLAGS_eval_mode != 0) {
    ET_CHECK_MSG(
        false, "Only TokenGenerator(kv) mode is supported to dump all logits.");
  }

  std::unique_ptr<executorch::extension::Module> module =
      std::make_unique<executorch::extension::Module>(
          FLAGS_model_path.c_str(),
          executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
  std::unique_ptr<executorch::extension::Module> attention_sink_rope_module;
  if (!FLAGS_attention_sink_rope_path.empty()) {
    attention_sink_rope_module =
        std::make_unique<executorch::extension::Module>(
            FLAGS_attention_sink_rope_path.c_str(),
            executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
  }
#if defined(EXECUTORCH_BUILD_EDGEINFER)
  std::unique_ptr<executorch::extension::Module> edgeinfer_composable_module;
  if (!FLAGS_edgeinfer_composable_model_path.empty()) {
    edgeinfer_composable_module =
        std::make_unique<executorch::extension::Module>(
            FLAGS_edgeinfer_composable_model_path.c_str(),
            executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
  }
  std::unique_ptr<executorch::extension::Module>
      edgeinfer_decode_composable_module;
  if (!FLAGS_edgeinfer_decode_model_path.empty()) {
    edgeinfer_decode_composable_module =
        std::make_unique<executorch::extension::Module>(
            FLAGS_edgeinfer_decode_model_path.c_str(),
            executorch::extension::Module::LoadMode::MmapUseMlockIgnoreErrors);
  }
  const auto edgeinfer_layer_paths =
      ParseEdgeInferPaths(FLAGS_edgeinfer_layer_model_paths);
  const auto edgeinfer_decode_layer_paths =
      ParseEdgeInferPaths(FLAGS_edgeinfer_decode_layer_model_paths);
  ET_CHECK_MSG(
      FLAGS_edgeinfer_composable_model_path.empty() ==
              (edgeinfer_composable_module == nullptr) &&
          (FLAGS_edgeinfer_layer_model_paths.empty() ==
           (FLAGS_edgeinfer_layers_per_shard == 0)) &&
          (FLAGS_edgeinfer_decode_layer_model_paths.empty() ==
           (FLAGS_edgeinfer_decode_layers_per_shard == 0)) &&
          (FLAGS_edgeinfer_layer_model_paths.empty() ||
           !edgeinfer_layer_paths.empty()) &&
          (FLAGS_edgeinfer_decode_layer_model_paths.empty() ||
           !edgeinfer_decode_layer_paths.empty()) &&
          (FLAGS_edgeinfer_decode_model_path.empty() ||
           edgeinfer_decode_composable_module != nullptr) &&
          (FLAGS_edgeinfer_decode_model_path.empty() ||
           edgeinfer_composable_module != nullptr) &&
          (FLAGS_edgeinfer_decode_layer_model_paths.empty() ||
           edgeinfer_decode_composable_module != nullptr) &&
          ((FLAGS_edgeinfer_layer_model_paths.empty() &&
            FLAGS_edgeinfer_decode_layer_model_paths.empty()) ||
           edgeinfer_composable_module != nullptr),
      "EdgeInfer layer-shard paths require a matching positive "
      "layers-per-shard value and an EdgeInfer core PTE.");
  auto edgeinfer_layer_modules = MakeModules(edgeinfer_layer_paths);
  auto edgeinfer_decode_layer_modules =
      MakeModules(edgeinfer_decode_layer_paths);
  start_runner(
      std::move(module),
      prompts,
      std::move(attention_sink_rope_module),
      std::move(edgeinfer_composable_module),
      std::move(edgeinfer_layer_modules),
      std::move(edgeinfer_decode_composable_module),
      std::move(edgeinfer_decode_layer_modules));
#else
  start_runner(
      std::move(module), prompts, std::move(attention_sink_rope_module));
#endif

  return 0;
}
