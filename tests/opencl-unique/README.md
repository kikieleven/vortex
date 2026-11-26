# opencl-unique — 移植到 Vortex 2.2 的去重 OpenCL 测试集

本目录是 `tests/opencl-unique` 测试集在 **Vortex 2.2** 上的移植版本。内容与 3.0
版本一致（84 个用例），目录自包含：`Makefile`、`common.mk`、每个用例的源码。
原始 `tests/opencl` 保持不变。

- 来源：3.0 仓库的 `tests/opencl-unique`（去重后的测试集）
- 用例数：84
- 默认扫描：**69**（15 个 EXCLUDE，见下）
- 测试框架：沿用本仓库 `tests/opencl/common.mk`（`-std=c++11`、`runtime/`），
  顶层 Makefile 使用 `TESTS`/`EXCLUDE` + 通配规则

## 使用方法

在 build 树中运行（构建是 out-of-tree 的）：

```bash
cd build            # 或 build32
. ./ci/toolchain_env.sh
make -C tests/opencl-unique run-simx
make -C tests/opencl-unique run-rtlsim

# 单个用例
make -C tests/opencl-unique/blueporcelain-vectoradd run-simx
```

改动本目录的 Makefile 后，回到 build 树重新执行 `../configure`（会读取现有
`config.mk` 的 XLEN，不会翻转 64/32）。

**注意：不要同时运行两个 build 树的测试。** 2.2 的
`kernel/scripts/vxbin.py:57` 把临时文件硬编码为 `/tmp/temp_kernel.bin`，两个
build 树并发执行时可能互相覆盖，产生假失败。最终验证是串行执行的。

## EXCLUDE 列表

| 用例 | 原因 |
|---|---|
| `parboil-tpacf` `parboil-histo` `parboil-mri-gridding` `atomicreduce` | `tpacf` 可通过（移植不含原子内建）；其余三个因 2.2 的 OpenCL C 1.2 内核库缺少 `_cl_atomic_*` 而构建失败，见 RESULTS.md |
| `image_copy` `image_int` `image_hwsw` | 2.2 PoCL 无 image 支持；`image_hwsw` 还依赖 TEX/RTU |
| `copybuf` | 2.2 上 host 段错误（3.0 可运行） |
| `blueporcelain-nqueen` | 2.2 报 `divergent branch` 并中止 |
| `shoc-sort` | 2.2 卡死 |
| `btree` | 2.2 卡死 |
| `ventus-softmax` | 结果错误（64 位还会卡死） |
| `cfd` | host 段错误 |
| `rodinia-srad-v2` | 64 位结果错误（32 位 PASS） |
| `nw` | 32 位结果错误（64 位 PASS） |

前两组在 3.0 中也属于特殊配置/环境限制；其余是 **2.2 模型层面的不兼容**，
保留目录便于以后修复后重新启用。`rodinia-srad-v2` 与 `nw` 只在单一 XLEN 失败，
可用对应 XLEN 手动运行验证。

## 与 3.0 版本的差异

- 测试框架：2.2 的 `tests/opencl/common.mk` 使用 `runtime/`、`kernel/`、
  `-DXLEN_*` 和自带 PoCL；3.0 使用 `sw/`、`VX_CFG_*` 和 ocl-icd。
- 2.2 的 `tests/Makefile` 中 `opencl` 在 XLEN=64 时默认不构建；本目录的
  `opencl-unique` 目标在两种 XLEN 下都会构建。
- 3.0 版本的 `EXCLUDE` 为 8 个；本版本为 15 个（7 个 2.2 模型不兼容用例）。

## 验证结果

详细的逐用例结果见 [`RESULTS.md`](RESULTS.md)，机器可读数据见
[`results/v2.2.json`](results/v2.2.json)。摘要：

- `build/`（XLEN=64）与 `build32/`（XLEN=32）上，默认扫描 69 个用例：
  **simx 与 rtlsim 在四种组合下全部 PASS（69/69 × 4）**；
- 验证按 64 simx → 64 rtlsim → 32 simx → 32 rtlsim **串行**执行，以避免
  `/tmp/temp_kernel.bin` 冲突；
- 2.2 的 simx/rtlsim `instrs` 不保证相等，结果中保留两侧数值供跨版本对比；
- `rodinia-leukocyte`、`myocyte` 在 2.2 rtlsim 上较慢（分别约数分钟和约 18 分钟），
  但均在超时内完成。

## 已知限制

- 只验证默认参数；未覆盖非默认尺寸、opae/xrt 后端、fp64。
- 已重建 EXT_A runtime 尝试原子用例：仅 `tpacf` 通过；`parboil-histo`/
  `parboil-mri-gridding`/`atomicreduce` 受 OpenCL 1.2 内核库限制，详见 RESULTS.md。
- **分类结果**（"特殊配置已验证" vs "本版本有问题的测试"）见 RESULTS.md。
- `common.mk` 中 `.depend` 对多源文件用例的依赖生成问题沿用 2.2 原实现，未修改。