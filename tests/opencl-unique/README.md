# opencl-unique — 移植到 Vortex 2.3 的去重 OpenCL 测试集

本目录是 `tests/opencl-unique` 测试集在 **Vortex 2.3** 上的移植版本。内容与 3.0
版本一致（84 个用例），目录自包含：`Makefile`、`common.mk`、每个用例的源码。
原始 `tests/opencl` 保持不变。

- 来源：3.0 仓库的 `tests/opencl-unique`（去重后的测试集）
- 用例数：84
- 默认扫描：**71**（13 个 EXCLUDE，见下）
- 测试框架：沿用本仓库 `tests/opencl/common.mk`；顶层 Makefile 使用
  `TESTS`/`EXCLUDE` + 通配规则（`build-%`、`run-%`、`clean-%`）

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

## EXCLUDE 列表

| 用例 | 原因 |
|---|---|
| `parboil-tpacf` `parboil-histo` `parboil-mri-gridding` `atomicreduce` | `tpacf` 可通过（移植不含原子内建）；其余三个因 2.3 的 OpenCL C 1.2 内核库缺少 `_cl_atomic_*` 而构建失败，见 RESULTS.md |
| `image_copy` `image_int` `image_hwsw` | 2.3 PoCL 报 `Device reports no image support`；`image_hwsw` 还依赖 TEX/RTU，而 2.3 没有这些单元 |
| `copybuf` | 2.3 上 host 段错误（3.0 上可运行） |
| `blueporcelain-nqueen` | 2.3 simx 报 `divergent branch!` 并中止（3.0 可运行） |
| `shoc-sort` | 2.3 simx 卡死（`-n 64` 也同样卡死） |
| `btree` | 2.3 simx 卡死 |
| `ventus-softmax` | `softmax_rows` 结果错误，随后 `softmax_chw` 触发 2.3 simobject 端口断言 |
| `cfd` | host 段错误（3.0 可运行） |

前两组在 3.0 中也被 EXCLUDE；最后五个是 **2.3 模型层面的不兼容**，不是测试
移植本身的问题，保留目录便于以后在 2.3 修复后重新启用。

## 与 3.0 版本的差异

- 测试框架：2.3 的 `tests/opencl/common.mk` 使用 `runtime/`、`kernel/`、
  `EXT_*` 宏和独立的 PoCL（自带 `libOpenCL.so`）；3.0 使用 `sw/`、`VX_CFG_*`
  和 ocl-icd。本目录直接使用 2.3 的 `common.mk`。
- 3.0 版本的 `EXCLUDE` 为 8 个；本版本增加上面 5 个 2.3 不兼容用例，共 13 个。
- 3.0 的去重策略与 57 个被删用例的详细清单见 3.0 仓库的
  `tests/opencl-unique/README.md`；本目录只承载去重后的 84 个用例。

## 验证结果

详细的逐用例结果见 [`RESULTS.md`](RESULTS.md)，机器可读数据见
[`results/v2.3.json`](results/v2.3.json)。摘要：

- `build/`（XLEN=64）与 `build32/`（XLEN=32）上，默认扫描 71 个用例：
  **rtlsim 均为 71/71 PASS**；
- simx 在两个树中各有 1 次**间歇性中止**（64 位 `blueporcelain-nn-op-lstm`、
  32 位 `blueporcelain-nn-op-gru`），单独重复 3 次均 PASS；
- 13 个 EXCLUDE 用例的结果与原因见 RESULTS.md：`tpacf` 为特殊配置且通过；
  其余包括 3 个 OpenCL 1.2 内核库限制、3 个 image 能力缺失、`copybuf` 运行失败，
  以及 5 个 2.3 模型不兼容用例；
- **分类结果**（"特殊配置已验证" vs "本版本有问题的测试"）见 RESULTS.md；
- 2.3 的 simx/rtlsim `instrs` 不保证相等，结果中保留两侧数值供跨版本对比。

## 已知限制

- 只验证默认参数；未覆盖非默认尺寸、opae/xrt 后端、fp64。
- 已重建 EXT_A runtime 尝试原子用例：仅 `tpacf` 通过；`parboil-histo`/
  `parboil-mri-gridding`/`atomicreduce` 受 OpenCL 1.2 内核库限制，详见 RESULTS.md。
- 2.3 没有 TEX/RTU 单元，`image_hwsw` 无法运行；`image_copy`/`image_int` 取决于
  PoCL 的 image 支持。
- `common.mk` 中 `.depend` 对多源文件用例的依赖生成问题沿用 2.3 原实现，未修改。