# EdgeInfer NPU for ExecuTorch

[中文说明](README_zh_CN.md)

This branch is an opt-in ExecuTorch extension for long-context LLM inference
on Qualcomm NPUs. It keeps every delegated Attention graph static, profiles a
finite portfolio of Query-row (`R`), internal Query-tile (`Q`), and K/V-width
(`C`) shapes, and uses dynamic programming to select the lowest measured-cost
composition for the current context.

The extension is incremental:

- the original Qualcomm `qnn_llama_runner` is still built and remains the
  default;
- EdgeInfer is compiled only with `EXECUTORCH_BUILD_EDGEINFER=ON`;
- the production path is selected only when EdgeInfer PTE paths and their
  shape contract are supplied;
- initialization and execution failures use the original decoder-PTE fallback
  before exposing partially updated K/V state.

## What It Adds

- AOT export of `attn_first_rR_cC` and `attn_merge_rR_cC` static methods.
- Exact online-softmax state composition across one or more K/V blocks.
- Measured-cost planning where one padded large graph and multi-graph
  compositions compete in the same search space.
- A persistent block-native K/V cache for Decode.
- Chunked Prefill with a causal staircase: blocks wholly above the causal
  boundary are skipped, while boundary blocks may use a small amount of
  padding.
- Independent Prefill and Decode portfolios, such as `R=32` for Prefill and
  `R=1` for Decode.
- Transactional rollback and a native ExecuTorch fallback path.

The exported portfolio has no compiled maximum-context field. Device memory,
host memory, integer ranges, and application policy remain practical limits.

## Paper and Detailed Results

The public manuscript is available here:

> [*EdgeInfer: Hardware-Aware Execution Planning for an Efficient,
> High-Performance LLM Inference Engine on Static-Graph Edge NPUs*](docs/paper/EdgeInfer_Preprint.pdf)

The setup uses one Transformer decoder block, a 40,960-token maximum
context, W16A16 IEEE FP16, and the QNN HTP backend. `Fixed` is the ExecuTorch
maximum-context baseline (`Cmax=40960`), with no online recompilation. The test
devices are a Xiaomi 13 (SM8550, HTP V73, Android 13) and a OnePlus 12 (SM8650,
HTP V75, Android 16); export and deployment use Qualcomm AI Runtime 2.43.0 and
Qualcomm AI Engine Direct 2.18.0.

| Device | Phase | Fixed | EdgeInfer | Speedup |
| --- | --- | ---: | ---: | ---: |
| Xiaomi 13 SM8550 | Decode | 2363.900 ms | 30.733 ms | 76.9x |
| Xiaomi 13 SM8550 | Prefill | 785.422 s | 20.060 s | 39.2x |
| OnePlus 12 SM8650 | Decode | 1994.460 ms | 21.343 ms | 93.4x |
| OnePlus 12 SM8650 | Prefill | 686.916 s | 14.779 s | 46.5x |

At 40K Decode, the paper reports 68-76.9x across 1-8 decoder blocks on Xiaomi
13 and 77-97x across 1-32 blocks on OnePlus 12; the 32-block OnePlus case falls
from 64.534 s to 0.829 s (78x). See Tables 1-3 and Figures 3-6 in the PDF for
the complete 1K-40K measurements, hardware-cost heatmaps, component breakdown,
and ablations. These are decoder-block/Attention measurements, not an
end-to-end result for a named full LLM.

## Prerequisites

- Linux development host.
- A supported Qualcomm Android device and working `adb` connection.
- Android NDK and a QAIRT/QNN SDK compatible with the device.
- Python environment supported by the checked-out ExecuTorch revision.
- `jq`, CMake, and a C++ build tool.
- A compatible dense Llama-family checkpoint and its parameter JSON.
- The original Qualcomm decoder PTE and tokenizer required by the standard
  ExecuTorch Llama flow.

Set up ExecuTorch and QAIRT first:

```bash
git submodule sync --recursive
git submodule update --init --recursive
./install_executorch.sh --editable

export ANDROID_NDK=/path/to/android-ndk
export QNN_SDK_ROOT=/path/to/qairt
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH"
```

The standard Qualcomm setup remains documented in
[`docs/source/llm/build-run-llama3-qualcomm-ai-engine-direct-backend.md`](docs/source/llm/build-run-llama3-qualcomm-ai-engine-direct-backend.md).

## 1. Select Candidate Widths

Candidate widths are positive powers of two. Probe them from the smallest
shape required by the workload and stop after the first target-device resource
limit. For a Qwen3-0.6B Attention geometry:

```bash
python -m executorch.extension.llm.export.probe_composable_attention_widths \
  --output-dir artifacts/probe-r1 \
  --backend qnn \
  --soc-model SM8650 \
  --dtype fp16 \
  --query-heads 16 \
  --kv-heads 8 \
  --query-rows 1 \
  --head-dim 128 \
  --start-width 1 \
  --max-width 4096
```

The probe records export time, PTE size, graph count, successful widths, and
failure provenance in `width_probe_manifest.json`. A real deployment should
also execute each successful shape on the target device before admitting it to
the portfolio.

## 2. Export Prefill and Decode Portfolios

Export the same model contract twice, with shapes appropriate to each phase.
The following widths are examples; use the successful measured set for the
target device. The original decoder PTE metadata `ar_len` must equal the
primary Prefill `--query-rows` value (128 in this example).

```bash
COMMON_ARGS=(
  --checkpoint /path/to/consolidated.00.pth
  --params /path/to/params.json
  --backend qnn
  --soc-model SM8650
  --dtype fp16
  --pre-attention-rope
)

python -m executorch.examples.models.llama.export_composable_llama \
  "${COMMON_ARGS[@]}" \
  --query-rows 128 \
  --prefill-query-rows 128 256 \
  --widths 1 2 4 8 16 32 64 128 256 512 1024 \
  --row-widths 128:1,2,4,8,16,32,64,128,256,512,1024 \
               256:1,2,4,8,16,32,64,128,256,512 \
  --row-query-tiles 128:32 256:32 \
  --output artifacts/edgeinfer_prefill.pte

python -m executorch.examples.models.llama.export_composable_llama \
  "${COMMON_ARGS[@]}" \
  --query-rows 1 \
  --widths 1024 4096 \
  --output artifacts/edgeinfer_decode_r1.pte
```

Keep each `.pte.json` file beside its PTE. It records export wall time, graph
count, PTE count and size, methods, dimensions, RoPE contract, and widths.

Validate the two contracts before deployment:

```bash
PREFILL_MANIFEST=artifacts/edgeinfer_prefill.pte.json
DECODE_MANIFEST=artifacts/edgeinfer_decode_r1.pte.json

test "$(jq -r '.query_rows' "$DECODE_MANIFEST")" = 1
for field in dim layers query_heads kv_heads head_dim vocab_size rope_theta rope_style; do
  test "$(jq -r ".$field" "$PREFILL_MANIFEST")" = \
       "$(jq -r ".$field" "$DECODE_MANIFEST")"
done

PREFILL_WIDTHS="$(jq -r '.widths | join(",")' "$PREFILL_MANIFEST")"
DECODE_WIDTHS="$(jq -r '.widths | join(",")' "$DECODE_MANIFEST")"
```

Do not pass the union of both width sets to both PTEs. Every declared width
must have matching first and merge methods in the corresponding PTE.

## 3. Build Android Libraries and Runners

Build and install the ExecuTorch Android libraries:

```bash
export ET_ANDROID_BUILD="$PWD/cmake-out-edgeinfer-android"

cmake -S . -B "$ET_ANDROID_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$ET_ANDROID_BUILD" \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-30 \
  -DEXECUTORCH_BUILD_QNN=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EDGEINFER=ON \
  -DQNN_SDK_ROOT="$QNN_SDK_ROOT"

cmake --build "$ET_ANDROID_BUILD" --target install -j
```

Build the Qualcomm examples against that installation:

```bash
cmake -S examples/qualcomm \
  -B "$ET_ANDROID_BUILD/examples/qualcomm" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-30 \
  -DCMAKE_PREFIX_PATH="$ET_ANDROID_BUILD/lib/cmake/ExecuTorch;$ET_ANDROID_BUILD/third-party/gflags;$ET_ANDROID_BUILD/lib/cmake/absl" \
  -DEXECUTORCH_BUILD_EDGEINFER=ON \
  -DQNN_SDK_ROOT="$QNN_SDK_ROOT"

cmake --build "$ET_ANDROID_BUILD/examples/qualcomm" --target \
  qnn_llama_runner qnn_llama_runner_edgeinfer \
  qnn_composable_attention_runner -j
```

Building `qnn_llama_runner` in the same tree verifies that the original path is
still available. `qnn_llama_runner_edgeinfer` is the opt-in executable.

## 4. Deploy

Create a device directory and push:

- `qnn_llama_runner_edgeinfer`;
- the original decoder PTE;
- Prefill and Decode EdgeInfer PTEs;
- tokenizer or tokenized prompt input;
- `libqnn_executorch_backend.so`;
- the QNN HTP/System stub libraries and the matching DSP skeleton required by
  the target SoC.

Example skeleton:

```bash
SERIAL=device-serial
DEVICE_DIR=/data/local/tmp/edgeinfer-npu

adb -s "$SERIAL" shell "mkdir -p '$DEVICE_DIR'"
adb -s "$SERIAL" push \
  "$ET_ANDROID_BUILD/examples/qualcomm/oss_scripts/llama/qnn_llama_runner_edgeinfer" \
  artifacts/edgeinfer_prefill.pte \
  artifacts/edgeinfer_decode_r1.pte \
  /path/to/decoder_baseline.pte \
  /path/to/tokenizer.bin \
  "$DEVICE_DIR/"
```

Follow the standard Qualcomm guide for the exact runtime and DSP library set.
Do not publish proprietary QAIRT libraries in the Git repository.

## 5. Run

Read every runner flag from the Prefill or Decode manifest instead of typing
shape metadata manually:

```bash
EDGEINFER_LAYERS="$(jq -r '.layers' "$PREFILL_MANIFEST")"
EDGEINFER_DIM="$(jq -r '.dim' "$PREFILL_MANIFEST")"
EDGEINFER_QUERY_HEADS="$(jq -r '.query_heads' "$PREFILL_MANIFEST")"
EDGEINFER_KV_HEADS="$(jq -r '.kv_heads' "$PREFILL_MANIFEST")"
EDGEINFER_HEAD_DIM="$(jq -r '.head_dim' "$PREFILL_MANIFEST")"
EDGEINFER_VOCAB_SIZE="$(jq -r '.vocab_size' "$PREFILL_MANIFEST")"
EDGEINFER_ROPE_THETA="$(jq -r '.rope_theta' "$PREFILL_MANIFEST")"
EDGEINFER_ROPE_STYLE="$(jq -r '.rope_style' "$PREFILL_MANIFEST")"

adb -s "$SERIAL" shell <<EOF
cd "$DEVICE_DIR"
export LD_LIBRARY_PATH="$DEVICE_DIR"
export ADSP_LIBRARY_PATH="$DEVICE_DIR"

./qnn_llama_runner_edgeinfer \
  --model_path=decoder_baseline.pte \
  --tokenizer_path=tokenizer.bin \
  --decoder_model_version=llama2 \
  --prompt="Once upon a time" \
  --seq_len=256 \
  --eval_mode=1 \
  --temperature=0 \
  --edgeinfer_composable_model_path=edgeinfer_prefill.pte \
  --edgeinfer_decode_model_path=edgeinfer_decode_r1.pte \
  --edgeinfer_widths="$PREFILL_WIDTHS" \
  --edgeinfer_decode_widths="$DECODE_WIDTHS" \
  --edgeinfer_layers="$EDGEINFER_LAYERS" \
  --edgeinfer_dim="$EDGEINFER_DIM" \
  --edgeinfer_query_heads="$EDGEINFER_QUERY_HEADS" \
  --edgeinfer_kv_heads="$EDGEINFER_KV_HEADS" \
  --edgeinfer_head_dim="$EDGEINFER_HEAD_DIM" \
  --edgeinfer_vocab_size="$EDGEINFER_VOCAB_SIZE" \
  --edgeinfer_rope_theta="$EDGEINFER_ROPE_THETA" \
  --edgeinfer_rope_style="$EDGEINFER_ROPE_STYLE" \
  --edgeinfer_pre_attention_rope=true \
  --edgeinfer_profile_warmup=1 \
  --edgeinfer_profile_iterations=3
EOF
```

For deterministic benchmarking, a tokenized prompt may be supplied instead of
`--prompt`. Keep the baseline and EdgeInfer runs on the same model, precision,
input, device, thermal state, warmup policy, and timing boundary.

## 6. Confirm EdgeInfer Actually Executed

Exit code zero is insufficient because a rejected portfolio intentionally
falls back to the original path. Preserve the unfiltered log and require:

```text
EdgeInfer Prefill persistent KV plan
EdgeInfer Prefill causal staircase
EdgeInfer Prefill completed through split QKV/Attention/Post graphs
EdgeInfer split Decode is enabled
EdgeInfer Decode token position=...
```

`unavailable before cache mutation` means the request used the original
decoder-PTE fallback. The sentence saying that the original path *remains* a
fallback is informational and does not mean fallback ran.

Common causes of an early fallback are:

- a width declared on the command line is absent from the matching PTE;
- Prefill and Decode manifests describe different model geometry;
- an incorrect RoPE style or model dimension was supplied;
- a required QNN runtime or DSP library is missing;
- the selected PTE is incompatible with the target SoC.

## Repository Policy

Commit source, tests, scripts, manifests without proprietary data, and concise
reproduction metadata. Do not commit QAIRT packages, model weights, generated
PTE/DLC files, Android build trees, or large raw logs. This fork retains the
upstream ExecuTorch BSD license; users are responsible for the licenses of
models and vendor SDKs.
