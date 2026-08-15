# ExecuTorch EdgeInfer 纯 NPU 分支

[English](README.md)

该分支是在 ExecuTorch 上增量实现的 Qualcomm NPU 长上下文 LLM 推理方案。
Attention 始终由静态图执行；部署前离线实测 Query 行数 `R`、内部 Query tile
`Q` 和 K/V 宽度 `C` 的有限候选集合，运行时动态规划在“单个允许填充的大图”
和“多个图组合”之间选择实测成本最低的方案。

该功能不会替换原有 ExecuTorch 路径：

- 原来的 `qnn_llama_runner` 继续保留并默认构建；
- 只有设置 `EXECUTORCH_BUILD_EDGEINFER=ON` 才编译 EdgeInfer；
- 只有显式传入 EdgeInfer PTE 和完整 shape contract 才启用新路径；
- 初始化或执行失败时，在不暴露半更新 K/V 状态的前提下回退原 decoder PTE。

## 核心功能

- 离线导出 `attn_first_rR_cC` 和 `attn_merge_rR_cC` 静态图组合库；
- 使用 online-softmax 状态完成跨块精确代数组合；
- 规划器以真机实测成本为第一目标，允许填充，也允许只选择一个大图；
- Decode 使用持久化 block-native K/V cache；
- Prefill 使用因果阶梯，只执行当前 Query tile 可见的 K/V 块，完全位于右上三角
  的块直接跳过，边界块允许少量填充；
- Prefill 和 Decode 可以分别使用不同 portfolio，例如 `R=32` 和 `R=1`；
- 具备事务回滚和原生 ExecuTorch 回退路径。

导出的 portfolio 没有预编译的最大 context 字段，但设备内存、主机内存、整数
范围和应用策略仍然是实际限制。

## 环境要求

- Linux 开发机；
- Qualcomm Android 手机和可用的 `adb`；
- 与手机匹配的 Android NDK 和 QAIRT/QNN SDK；
- 当前 ExecuTorch revision 支持的 Python 环境；
- `jq`、CMake 和 C++ 构建工具；
- 兼容的 dense Llama-family checkpoint 及参数 JSON；
- 原 Qualcomm ExecuTorch 流程所需的 decoder PTE 和 tokenizer。

```bash
git submodule sync --recursive
git submodule update --init --recursive
./install_executorch.sh --editable

export ANDROID_NDK=/path/to/android-ndk
export QNN_SDK_ROOT=/path/to/qairt
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH"
```

Qualcomm 原始导出和部署流程见
[`docs/source/llm/build-run-llama3-qualcomm-ai-engine-direct-backend.md`](../source/llm/build-run-llama3-qualcomm-ai-engine-direct-backend.md)。

## 1. 探测候选宽度

候选宽度从 1 开始按 2 的幂次方增长。下面以 Qwen3-0.6B Attention 几何为例：

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

`width_probe_manifest.json` 会记录每个 shape 的导出时间、PTE 大小、图数、成功
宽度和失败原因。正式部署还应在目标手机执行每个成功导出的 shape，只有真正可
执行的 shape 才进入性能 profile。

## 2. 分别导出 Prefill 和 Decode Portfolio

两个 PTE 使用相同 checkpoint 和模型契约，但 Query rows 和宽度可以不同。原
decoder PTE 元数据中的 `ar_len` 必须等于 Prefill primary `--query-rows`（本例为
128）：

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

这里的宽度只是示例，应替换为目标手机探测并实测成功的集合。每个 `.pte.json`
必须与对应 PTE 一起保存。

部署前验证两个 manifest：

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

不要把 Prefill 和 Decode 的宽度并集同时传给两个 PTE。命令行声明的每个宽度，
都必须在对应 PTE 中存在匹配的 first 和 merge method。

## 3. 构建 Android 库和 Runner

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

同一构建树中同时编译 `qnn_llama_runner`，用于确认原路径仍然保留；
`qnn_llama_runner_edgeinfer` 是显式启用的新入口。

## 4. 部署到手机

推送以下文件到同一个设备目录：

- `qnn_llama_runner_edgeinfer`；
- 原 decoder PTE；
- EdgeInfer Prefill PTE 和 Decode PTE；
- tokenizer 或预先 tokenized 的输入；
- `libqnn_executorch_backend.so`；
- 目标 SoC 所需的 QNN HTP/System stub 和对应 DSP skeleton。

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

具体 QNN runtime/DSP 文件组合以 ExecuTorch Qualcomm 原指南为准。不要把 QAIRT
专有库提交到 GitHub。

## 5. 完整执行命令

所有模型 shape 参数都从 Prefill manifest 读取，避免手工抄错：

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

性能对比必须保持模型、精度、输入、设备、温度、warmup 和计时边界一致。

## 6. 判断是否真正走了 EdgeInfer

退出码为 0 不足以证明新路径执行，因为 portfolio 配置错误时会安全回退。必须保留
完整日志，并确认出现：

```text
EdgeInfer Prefill persistent KV plan
EdgeInfer Prefill causal staircase
EdgeInfer Prefill completed through split QKV/Attention/Post graphs
EdgeInfer split Decode is enabled
EdgeInfer Decode token position=...
```

日志出现 `unavailable before cache mutation` 说明该请求使用了原 decoder-PTE
fallback。日志中“原路径 remains a fallback”只是说明回退能力仍在，并不表示已经
发生回退。

常见错误包括：

- 命令行声明了对应 PTE 中不存在的宽度；
- Prefill 和 Decode manifest 的模型几何不一致；
- RoPE style 或模型维度填写错误；
- 缺少 QNN runtime/DSP 库；
- PTE 与目标 SoC 不匹配。

## GitHub 仓库规则

可以提交源码、测试、脚本、去除专有数据的 manifest 和精简复现信息。不要提交
QAIRT、模型权重、PTE/DLC、Android 构建目录或大体积原始日志。该 fork 继续遵守
ExecuTorch BSD license，模型和厂商 SDK 的许可证由使用者另行负责。
