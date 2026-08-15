# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

cmake_minimum_required(VERSION 3.19)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(_test_srcs
    test_generation_config.cpp
    test_text_llm_runner.cpp
    test_text_prefiller.cpp
    test_text_decoder_runner.cpp
    test_multimodal_input.cpp
    test_util.cpp
    test_wav_loader.cpp
)

if(EXECUTORCH_BUILD_EDGEINFER)
  list(
    APPEND
    _test_srcs
    test_composable_attention_runner.cpp
    test_composable_attention_runner_integration.cpp
    test_edgeinfer_decode_cache_transaction.cpp
    test_static_attention_planner.cpp
  )
  set(_composable_attention_test_pte
      "${CMAKE_CURRENT_BINARY_DIR}/composable_attention_test.pte"
  )
  add_custom_command(
    OUTPUT "${_composable_attention_test_pte}"
    BYPRODUCTS "${_composable_attention_test_pte}.json"
    COMMAND
      ${CMAKE_COMMAND} -E env "PYTHONPATH=${EXECUTORCH_ROOT}/src"
      ${PYTHON_EXECUTABLE} -m
      executorch.extension.llm.export.export_composable_attention --backend
      portable --dtype fp32 --query-heads 4 --kv-heads 2 --query-rows 1
      --head-dim 8 --widths 4 8 --output "${_composable_attention_test_pte}"
    DEPENDS
      "${EXECUTORCH_ROOT}/extension/llm/export/export_composable_attention.py"
      "${EXECUTORCH_ROOT}/extension/llm/modules/composable_attention.py"
  )
  add_custom_target(
    generated_composable_attention_test_file
    DEPENDS "${_composable_attention_test_pte}"
  )
  list(APPEND _test_extra_libs extension_llm_composable_attention_runner)
endif()

# Add LSan stub for Apple platforms
if(APPLE)
  list(APPEND _test_srcs lsan_stub.cpp)
endif()

et_cxx_test(
  test_runner
  SOURCES
  ${_test_srcs}
  EXTRA_LIBS
  executorch
  ${_test_extra_libs}
  extension_llm_runner
  portable_kernels
  portable_ops_lib
)

# Portable operator registration is constructor-driven. Keep the registration
# objects in this host integration-test binary even though no C++ symbol refers
# to them directly.
executorch_target_link_options_shared_lib(portable_ops_lib)

if(EXECUTORCH_BUILD_EDGEINFER)
  add_dependencies(test_runner generated_composable_attention_test_file)
  set_property(
    TEST test_runner
    APPEND
    PROPERTY ENVIRONMENT
             "ET_COMPOSABLE_ATTENTION_PTE=${_composable_attention_test_pte}"
  )
endif()

# Override sanitizer to this issue:
# https://github.com/abseil/abseil-cpp/issues/841 Root issue:
# https://github.com/llvm/llvm-project/issues/16778
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_link_options(test_runner PUBLIC --rtlib=compiler-rt)
endif()
