# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# model sharding with custom op
set(CUSTOM_OP_SRCS_FILE
    "${EXECUTORCH_SOURCE_DIR}/extension/llm/custom_ops/op_fallback.cpp"
)
set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)

option(
  EXECUTORCH_BUILD_EDGEINFER
  "Build EdgeInfer composable static Attention support"
  OFF
)

add_library(custom_ops ${CUSTOM_OP_SRCS_FILE})
target_include_directories(custom_ops PUBLIC "${_common_include_directories}")
target_include_directories(
  custom_ops PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../include"
)
target_link_libraries(custom_ops PUBLIC executorch_core full_portable_ops_lib)
executorch_target_link_options_shared_lib(custom_ops)

# preprocess qnn runner src files for llama
set(_llama_runner__srcs ${_llama_runner__srcs})
list(TRANSFORM _llama_runner__srcs PREPEND "${EXECUTORCH_SOURCE_DIR}/")
list(FILTER _llama_runner__srcs EXCLUDE REGEX ".*(/runner/).*")
list(
  PREPEND
  _llama_runner__srcs
  ${CMAKE_CURRENT_LIST_DIR}/qnn_llama_runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/runner.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/cache_utils.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/decoder_runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/decoder_runner.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/prompt_processor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/prompt_processor.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/token_generator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/token_generator.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/imem_alloc.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/client_mem.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/lhd_token_generator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/lhd_token_generator.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/attention_sink_rope_runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/attention_sink_rope_runner.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/rpc_mem.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/rpc_mem.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/kv_manager.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/kv_manager.h
  ${EXECUTORCH_SOURCE_DIR}/examples/models/llama/runner/runner.cpp
  ${EXECUTORCH_SOURCE_DIR}/examples/models/llama/runner/runner.h
)

list(APPEND _llama_runner__srcs)

# build qnn llama runner
add_executable(qnn_llama_runner ${_llama_runner__srcs})
target_include_directories(
  qnn_llama_runner PUBLIC ${_common_include_directories}
)

executorch_target_link_options_shared_lib(quantized_ops_lib)

target_link_libraries(
  qnn_llama_runner
  qnn_executorch_backend
  executorch_core
  extension_data_loader
  extension_flat_tensor
  extension_llm_runner
  extension_module
  extension_tensor
  gflags
  custom_ops
  quantized_ops_lib
  quantized_kernels
  tokenizers::tokenizers
)

target_include_directories(
  qnn_llama_runner PUBLIC ${EXECUTORCH_ROOT}/extension/llm/tokenizers/include
)

target_compile_options(qnn_llama_runner PUBLIC ${_common_compile_options})
set_target_properties(
  qnn_llama_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)

if(EXECUTORCH_BUILD_EDGEINFER)
  set(
    _edgeinfer_llama_runner__srcs
    ${_llama_runner__srcs}
    ${CMAKE_CURRENT_LIST_DIR}/runner/edgeinfer_fallback.h
    ${CMAKE_CURRENT_LIST_DIR}/runner/edgeinfer_prompt_processor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/runner/edgeinfer_prompt_processor.h
  )
  add_executable(
    qnn_llama_runner_edgeinfer ${_edgeinfer_llama_runner__srcs}
  )
  target_compile_definitions(
    qnn_llama_runner_edgeinfer PRIVATE EXECUTORCH_BUILD_EDGEINFER=1
  )
  target_include_directories(
    qnn_llama_runner_edgeinfer
    PUBLIC ${_common_include_directories}
           ${EXECUTORCH_ROOT}/extension/llm/tokenizers/include
  )
  target_link_libraries(
    qnn_llama_runner_edgeinfer
    qnn_executorch_backend
    executorch_core
    extension_data_loader
    extension_flat_tensor
    extension_llm_runner
    extension_llm_composable_attention_runner
    extension_module
    extension_tensor
    gflags
    custom_ops
    quantized_ops_lib
    quantized_kernels
    tokenizers::tokenizers
  )
  target_compile_options(
    qnn_llama_runner_edgeinfer PUBLIC ${_common_compile_options}
  )
  set_target_properties(
    qnn_llama_runner_edgeinfer
    PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
  )

  # Build the finite-graph, arbitrary-context Attention validation runner.
  add_executable(
    qnn_composable_attention_runner
    ${CMAKE_CURRENT_LIST_DIR}/qnn_composable_attention_runner.cpp
  )
  target_include_directories(
    qnn_composable_attention_runner PUBLIC ${_common_include_directories}
  )
  target_link_libraries(
    qnn_composable_attention_runner
    qnn_executorch_backend
    executorch_core
    extension_llm_composable_attention_runner
    extension_module
    extension_tensor
    gflags
  )
  target_compile_options(
    qnn_composable_attention_runner PUBLIC ${_common_compile_options}
  )
  set_target_properties(
    qnn_composable_attention_runner
    PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
  )

  # Build the full Llama validation runner around the same Attention portfolio.
  add_executable(
    qnn_composable_llama_runner
    ${CMAKE_CURRENT_LIST_DIR}/qnn_composable_llama_runner.cpp
  )
  target_include_directories(
    qnn_composable_llama_runner PUBLIC ${_common_include_directories}
  )
  target_link_libraries(
    qnn_composable_llama_runner
    qnn_executorch_backend
    executorch_core
    extension_llm_composable_attention_runner
    extension_module
    extension_tensor
    gflags
  )
  target_compile_options(
    qnn_composable_llama_runner PUBLIC ${_common_compile_options}
  )
  set_target_properties(
    qnn_composable_llama_runner
    PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
  )

  # Same-weight, complete-layer long-context benchmark used for migration
  # checks.
  add_executable(
    qnn_composable_layer_benchmark
    ${CMAKE_CURRENT_LIST_DIR}/qnn_composable_layer_benchmark.cpp
  )
  target_include_directories(
    qnn_composable_layer_benchmark PUBLIC ${_common_include_directories}
  )
  target_link_libraries(
    qnn_composable_layer_benchmark
    qnn_executorch_backend
    executorch_core
    extension_llm_composable_attention_runner
    extension_module
    extension_tensor
    gflags
  )
  target_compile_options(
    qnn_composable_layer_benchmark PUBLIC ${_common_compile_options}
  )
  set_target_properties(
    qnn_composable_layer_benchmark
    PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
  )
endif()

# build qnn multimodal runner preprocess qnn runner src files for multimodal
set(_multimodal_runner__srcs ${_llama_runner__srcs})
list(FILTER _multimodal_runner__srcs EXCLUDE REGEX ".*qnn_llama_runner.*")
list(FILTER _multimodal_runner__srcs EXCLUDE REGEX ".*runner/runner\.(cpp|h)")
list(FILTER _multimodal_runner__srcs EXCLUDE REGEX ".*edgeinfer_.*")
list(
  PREPEND
  _multimodal_runner__srcs
  ${CMAKE_CURRENT_LIST_DIR}/qnn_multimodal_runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_runner.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_embedding_merger.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_embedding_merger.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/utils.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/encoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/encoder.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/tok_embedding_runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/tok_embedding_runner.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/tok_embedding_processor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/tok_embedding_processor.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_prompt_processor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_prompt_processor.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_token_generator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_token_generator.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_lhd_token_generator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_lhd_token_generator.h
)

list(APPEND _multimodal_runner__srcs)

# build qnn multimodal runner
add_executable(qnn_multimodal_runner ${_multimodal_runner__srcs})
target_include_directories(
  qnn_multimodal_runner PUBLIC ${_common_include_directories}
)
target_include_directories(
  qnn_multimodal_runner
  PUBLIC ${EXECUTORCH_ROOT}/extension/llm/tokenizers/include
)
target_compile_options(qnn_multimodal_runner PUBLIC ${_common_compile_options})

target_link_libraries(
  qnn_multimodal_runner
  qnn_executorch_backend
  executorch_core
  extension_data_loader
  extension_flat_tensor
  extension_llm_runner
  extension_module
  extension_tensor
  gflags
  custom_ops
  quantized_ops_lib
  quantized_kernels
  tokenizers::tokenizers
)

set_target_properties(
  qnn_multimodal_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)
