# opencl-unique — 去重后的 OpenCL 功能覆盖测试集

本目录是 [`tests/opencl`](../opencl) 的**自包含去重副本**，用于在不改动原始测试集的
前提下，提供一份"一个功能点一个代表用例"的测试集。原始 `tests/opencl` 完整保留，
本目录中的用例可以直接构建、运行，也可以按需继续增删。

- 来源：`tests/opencl`（vortex-3.0 基线，含 `blueporcelain-nn-op-*` 用例）
- 原始用例数：141
- 保留：84（默认扫描 76）
- 移除：57（24 个严格重复 + 31 个算法族合并 + 2 个在 3.0 上已知失败）

## 使用方法

在 build 树中运行（构建是 out-of-tree 的）：

```bash
cd build            # 或 build32
make -C tests/opencl-unique run-simx
make -C tests/opencl-unique run-rtlsim

# 单个用例
make -C tests/opencl-unique/blueporcelain-vectoradd run-simx
```

需要 A 扩展的原子用例要显式传 `CONFIGS`（见下方 EXCLUDE 列表）：

```bash
make -C tests/opencl-unique/parboil-histo run-simx \
     CONFIGS=-DVX_CFG_EXT_A_ENABLE
```

改了本目录的 Makefile 后，回到 build 树重新 `../configure --xlen=<32|64>`。

## 去重规则

1. **严格重复**：同一算法、同一数据类型、同一判据（不看实现）只保留一个代表。
2. **算法族合并**：对 PolyBench 线性代数、stencil/迭代求解、reduction/scan、
   排序、image、DP、聚类、粒子/势场这些族，每族保留 1–3 个代表。
3. **移除已知失败**：`lbm` 和 `codegen` 在 Vortex 3.0 上失败，原始
   `tests/opencl` 中同样失败，因此不纳入本测试集：
   - `lbm`：`clEnqueueNDRangeKernel` 错误
   - `codegen`：`misaligned memory access`

## 移除的 57 个用例及保留代表

| 功能组 | 移除 | 保留代表 |
|---|---|---|
| 普通 GEMM | `sgemm` `sgemm2` `sgemm3` `parboil-sgemm` `shoc-gemm` `polybench-gemm` | `ventus-gemm`（另有 `ventus-gemm-fused`） |
| 向量加 | `vecadd` `ventus-add` | `blueporcelain-vectoradd`（float/int/int64） |
| 点积 | `dotproduct` | `blueporcelain-dot-product` |
| 矩阵转置 | `transpose` | `blueporcelain-matrix-transpose` |
| BFS | `bfs` `parboil-bfs` | `shoc-bfs` |
| 2D 卷积 | `conv3` `sfilter` `blueporcelain-nn-op-conv2d` `polybench-convolution-2d` | `ventus-conv` |
| 3D 卷积 | `polybench-convolution-3d` | `blueporcelain-nn-op-conv3d` |
| MNIST 卷积链 | `ventus-mnist-conv-tiny` | `ventus-mnist-conv-small` |
| BatchNorm | `blueporcelain-nn-op-batchnorm2d` | `ventus-batchnorm2d`（`batchnorm3d` 保留） |
| ReLU | `blueporcelain-nn-op-relu` | `ventus-relu` |
| Softmax | `blueporcelain-nn-op-softmax` | `ventus-softmax` |
| Pooling | `blueporcelain-nn-op-maxpool2d` | `ventus-pool2d`（`adaptiveavgpool2d` 保留） |
| 直方图 | `histogram` | `parboil-histo` |
| 基数排序 | `blueporcelain-radix-sort` | `shoc-sort` |
| PolyBench 线性代数/矩阵 | `polybench-2mm` `polybench-atax` `polybench-bicg` `polybench-gesummv` `polybench-gramschmidt` `polybench-mvt` `polybench-syrk` `polybench-correlation` `lud` `guassian` | `polybench-3mm` `polybench-syr2k` `polybench-gemver` `polybench-covariance` `polybench-doitgen` `polybench-lu`（另有 `blueporcelain-matrix-inversion`） |
| Stencil/迭代求解 | `stencil` `parboil-stencil` `polybench-jacobi-1d-imper` `polybench-jacobi-2d-imper` `polybench-fdtd-2d` `hotspot` | `shoc-stencil2d` `hotspot3D` `polybench-adi` |
| Reduction/Scan | `psum` `blueporcelain-reduction-sum` `blueporcelain-prefix-sum` | `shoc-reduction` `blueporcelain-reduction-max` `shoc-scan` |
| 排序 | `psort` `blueporcelain-merge-sort` `hybridsort` | `blueporcelain-bitonic-sort` `shoc-sort` |
| Image | `image_get` `image_linear` `image_ff` `image_ff_linear` `image_ff_bgra` | `image_copy` `image_int` `image_hwsw` |
| DP | `pathfinder` | `nw` |
| 聚类 | `streamcluster` | `kmeans` |
| 粒子/势场 | `lavaMD` `parboil-cutcp` | `blueporcelain-nbody` `rodinia-particlefilter-single` |
| 已知失败 | `lbm` `codegen` | —（原始 `tests/opencl` 中保留） |

## 默认扫描排除项（EXCLUDE）

这些用例不是重复，而是需要特殊运行条件：

| 用例 | 原因 |
|---|---|
| `parboil-tpacf` `parboil-histo` `parboil-mri-gridding` `atomicreduce` | 需要 `CONFIGS=-DVX_CFG_EXT_A_ENABLE` |
| `image_copy` `image_int` `image_hwsw` | 需要 image-enabled 的 PoCL 构建 |
| `copybuf` | 只测试 buffer copy API，不启动 kernel（`instrs=0`） |

## 验证结果（XLEN=64 和 XLEN=32）

在本仓库的 `build/`（XLEN=64）和 `build32/`（XLEN=32）上：

- 默认扫描：**76/76 PASSED**；
- 同一配置下 simx 与 rtlsim 的 `PERF: instrs=` **76/76 完全一致**；
- 聚合目标 `make -C tests/opencl-unique run-simx` 在两个树中均通过；
- 8 个 EXCLUDE 用例手工补跑全部通过（原子用例带 `EXT_A`，image 三类，`copybuf`），
  simx/rtlsim 指令数也一致。

## 已知限制

- 只验证了各用例的默认参数；未覆盖非默认尺寸、opae/xrt 后端、fp64。
- `common.mk` 的 `.depend` 规则对多源文件用例（`nearn`/`blackscholes`/`kmeans`/
  `parboil-*`）生成失败：`-MT $(OBJS)` 只对第一个目标生效。编译和链接不受影响，
  但头文件依赖不完整（修改 `common.h` 可能不会触发重编）。该问题来自原始
  `tests/opencl/common.mk`，本目录未修改。
- 本目录是原始用例的副本；若 `tests/opencl` 中的对应用例后续修复，需要手动同步。
- 为控制仓库体积，`spmv` 的备用数据集 `Dubcova3.*`（约 60 MB）未复制；默认参数使用的
  `1138_bus.*` 已包含，可以直接运行。需要 Dubcova3 时请从 `tests/opencl/spmv/` 取用。
